"""
Experiment A — Gated Height Patch + Smart Validation Layer
============================================================
Two improvements over the previous height-patch experiment:

Improvement 1 — Gated height feature:
  Instead of feeding zero patches for non-profil layers
  (which is misleading — zeros imply a flat surface),
  we gate the height branch output with a binary scalar:
    h_feat = HeightPatchCNN(patch) * has_height
  where has_height=1 for L248-L252 and has_height=0 otherwise.
  The gate scalar is also concatenated explicitly so the
  classifier knows whether height data is available:
    fused = [lstm, frame_cnn, h_feat * gate, pv, gate]
  This makes zero contribution truly zero and unambiguous.

Improvement 2 — Smart validation layer selection:
  Instead of random val layer, always pick the layer
  most similar in (P, V) space to the test layer.
  Reduces checkpoint selection noise from mismatched
  process conditions.

No edge trimming (confirmed negative result).
No architecture changes beyond the gate scalar.

Output: extended_analysis/lolo_experiment_a/
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.interpolate import interp1d

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Output dirs ───────────────────────────────────────────────────
OUTPUT_DIR  = Path("extended_analysis")
LOLO_DIR    = OUTPUT_DIR / "lolo_experiment_len10"
LOLO_DIR.mkdir(parents=True, exist_ok=True)

# Baseline to compare against
BASELINE_CSV = OUTPUT_DIR / "lolo_PV" / "lolo_PV_predictions.csv"

# ── Reproducibility ───────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cpu":
    print("  WARNING: Running on CPU — training will be slow.")
    print("  If you have a GPU, check that PyTorch CUDA is installed.")
    import torch
    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")

# ── Experiment config ─────────────────────────────────────────────
ALL_LAYERS   = list(range(226, 232)) + list(range(245, 253))
SEQ_LEN      = 10
IMAGE_SIZE   = (128, 128)
BATCH_SIZE   = 8
MAX_EPOCHS   = 60
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-3
MAX_TRAIN_AUC = 0.95

# ── Refinement 1: edge trim ───────────────────────────────────────
TRIM_START = 0   # no trimming (confirmed negative result)
TRIM_END   = 0

# ── Refinement 2: profilometry height ────────────────────────────
PROFIL_CSV    = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
TRACK_CSV_DIR = Path(
    r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2"
    r"\Erfan_balling_data_updated"
)
PX_X_MM = 0.00789
PX_Y_MM = 0.01250

# Per-layer auto-aligned offsets + row centers (from xcorr alignment)
PROFIL_LAYER_CONFIG = {
    248: {"x_offset": 25.5440, "row_center": 62,  "half_band": 20},
    249: {"x_offset": 25.5740, "row_center": 143, "half_band": 20},
    250: {"x_offset": 25.4540, "row_center": 221, "half_band": 20},
    251: {"x_offset": 25.4400, "row_center": 301, "half_band": 20},
    252: {"x_offset": 25.5140, "row_center": 379, "half_band": 20},
}
PROFIL_LAYERS = set(PROFIL_LAYER_CONFIG.keys())

# ── 2D height patch config ────────────────────────────────────────
# Each camera frame corresponds to a 2D region in the profilometry.
# We extract a patch centered at (x_frame, row_center) covering
# the melt pool footprint in both directions.
#
# X (along scan):   frame spacing ~0.040mm, 6-frame window = 0.240mm
#                   patch = ±0.200mm = ±25 cols each side
# Y (cross-track):  track width ~0.200mm
#                   patch = ±0.200mm = ±16 rows each side (at 12.5um/px)
#
# Patch shape: (2*PATCH_HALF_Y + 1) x (2*PATCH_HALF_X + 1)
#            = 33 rows x 51 cols
# Downsampled to PATCH_OUT x PATCH_OUT for CNN input
PATCH_HALF_X = 25   # cols  = 25 * 7.89um = 0.197mm each side
PATCH_HALF_Y = 16   # rows  = 16 * 12.5um = 0.200mm each side
PATCH_OUT    = 16   # downsample patch to 16x16 for CNN


# ─────────────────────────────────────────────
# PROFILOMETRY HEIGHT LOOKUP TABLE
# ─────────────────────────────────────────────

def build_height_lookup():
    """
    Load profilometry Z matrix and build per-layer height lookup.

    For each profilometry layer returns:
      - Z matrix (shared, same object)
      - per-layer normalization stats (mean, std)
      - frame -> (col_center, row_center) mapping

    Returns:
      Z: the full height matrix
      lookup: dict layer -> {x_offset, row_center, h_global_mean, h_global_std}
      frame_xy: dict (layer, frame) -> (col, row) pixel coords in Z
    """
    if not PROFIL_CSV.exists():
        print("  WARNING: profilometry CSV not found")
        return None, {}, {}

    print("Loading profilometry CSV...")
    Z = pd.read_csv(PROFIL_CSV, header=None).values.astype(float)
    n_rows, n_cols = Z.shape
    print(f"  Shape: {n_rows}x{n_cols}")

    lookup   = {}
    frame_xy = {}

    for layer, cfg in PROFIL_LAYER_CONFIG.items():
        rc       = cfg["row_center"]
        x_offset = cfg["x_offset"]

        # Normalization: use stats from the track band
        hb   = cfg["half_band"]
        r_lo = max(0, rc - hb)
        r_hi = min(n_rows - 1, rc + hb)
        band = Z[r_lo:r_hi+1, :]
        h_mean = float(np.nanmean(band))
        h_std  = float(np.nanstd(band))
        if h_std < 1e-8: h_std = 1.0

        lookup[layer] = {
            "x_offset"    : x_offset,
            "row_center"  : rc,
            "h_mean"      : h_mean,
            "h_std"       : h_std,
            "n_rows"      : n_rows,
            "n_cols"      : n_cols,
        }
        print(f"  L{layer}: rc={rc}  "
              f"h_mean={h_mean*1000:.1f}um  h_std={h_std*1000:.1f}um")

        # Build frame -> pixel coordinate mapping
        csv_path = TRACK_CSV_DIR / f"L0{layer}.csv"
        if not csv_path.exists():
            continue
        df_t = pd.read_csv(csv_path)
        df_t.columns = [c.strip() for c in df_t.columns]

        for _, row in df_t.iterrows():
            fn    = int(row["frame_number"])
            xm    = float(row["x(mm)"])
            x_p   = xm - x_offset
            col_c = int(round(x_p / PX_X_MM))
            frame_xy[(layer, fn)] = (col_c, rc)

    return Z, lookup, frame_xy


def extract_height_patch(Z, col_c, row_c, h_mean, h_std,
                          n_rows, n_cols):
    """
    Extract a 2D height patch centered at (col_c, row_c) from Z.

    Patch covers:
      X: col_c - PATCH_HALF_X  to  col_c + PATCH_HALF_X
         = +/-0.197mm along scan direction
      Y: row_c - PATCH_HALF_Y  to  row_c + PATCH_HALF_Y
         = +/-0.200mm cross-track (left/right/above/below)

    Returns a (PATCH_OUT, PATCH_OUT) float32 tensor, z-score normalized.
    Returns zeros if out of bounds.
    """
    import torch.nn.functional as F_nn

    r_lo = row_c - PATCH_HALF_Y
    r_hi = row_c + PATCH_HALF_Y + 1
    c_lo = col_c - PATCH_HALF_X
    c_hi = col_c + PATCH_HALF_X + 1

    # Clamp to valid range
    if r_lo < 0 or r_hi > n_rows or c_lo < 0 or c_hi > n_cols:
        return torch.zeros(1, PATCH_OUT, PATCH_OUT, dtype=torch.float32)

    patch = Z[r_lo:r_hi, c_lo:c_hi].copy()
    patch = np.where(np.isnan(patch), h_mean, patch)

    # Z-score normalize
    patch = (patch - h_mean) / h_std

    # Downsample to PATCH_OUT x PATCH_OUT
    p_t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
    p_t = F_nn.interpolate(p_t, size=(PATCH_OUT, PATCH_OUT),
                           mode="bilinear", align_corners=False)
    return p_t.squeeze(0)   # shape: (1, PATCH_OUT, PATCH_OUT)


# ─────────────────────────────────────────────
# MODEL — process branch takes [P, V, H]
# ─────────────────────────────────────────────

class HeightPatchCNN(nn.Module):
    """
    Small CNN that encodes a 2D profilometry height patch
    (PATCH_OUT x PATCH_OUT) into a feature vector.

    Input:  (B, 1, PATCH_OUT, PATCH_OUT) — normalized height patch
    Output: (B, height_dim) — feature vector

    The patch covers the melt pool footprint in 2D:
      X: +/-0.197mm along scan  (left and right of frame position)
      Y: +/-0.200mm cross-track (above and below track centerline)
    """
    def __init__(self, height_dim=16, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, height_dim),
            nn.ReLU(True), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.encoder(x)


class DiffFrameCNN(nn.Module):
    def __init__(self, feature_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(True), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.proj(self.encoder(x))


class EnhancedDetector(nn.Module):
    """
    DiffImageDetectorPV + 2D profilometry height patch branch.

    For each 6-frame camera window, extracts a 2D height patch
    from the profilometry centered at the window's x position
    and track row_center. The patch covers:
      X: +/-0.197mm along scan (left and right of frame)
      Y: +/-0.200mm cross-track (above and below centerline)

    This gives the model spatial context about the bead height
    in all directions around the melt pool footprint.

    For layers without profilometry (9 of 14), the patch is
    all zeros and the HeightPatchCNN learns to output zeros.

    Architecture:
      DiffFrameCNN (shared) -> LSTM
      FrameCNN (last raw frame)
      HeightPatchCNN (2D profil patch)  <-- NEW
      PV branch [P_norm, V_norm]
      -> concat -> classifier
    """
    def __init__(self, feature_dim=64, lstm_hidden=64,
                 n_layers=2, pv_hidden=16,
                 height_dim=16, dropout=0.35):
        super().__init__()
        self.diff_cnn  = DiffFrameCNN(feature_dim, dropout)
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(True), nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(True), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, 32), nn.ReLU(True), nn.Dropout(dropout),
        )
        self.height_cnn = HeightPatchCNN(height_dim, dropout)
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.pv_branch = nn.Sequential(
            nn.Linear(2, pv_hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        fused = lstm_hidden * 2 + 32 + height_dim + pv_hidden + 1  # +1 for gate
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, diffs, last_frame, pv, height_patch,
                has_height):
        """
        diffs:        (B, SEQ_LEN-1, 1, H, W)
        last_frame:   (B, 1, H, W)
        pv:           (B, 2)    [P_norm, V_norm]
        height_patch: (B, 1, PATCH_OUT, PATCH_OUT)
        has_height:   (B, 1)   1.0 if profil available, 0.0 otherwise

        Gated height branch:
          h_feat = HeightPatchCNN(patch) * has_height
          fused  = [lstm, frame_cnn, h_feat, pv, has_height]
        The gate (has_height) is also concatenated explicitly so
        the classifier knows whether height data is real or absent.
        This makes zeros unambiguous — they mean "no data",
        not "flat surface".
        """
        B, T, C, H, W = diffs.shape
        f   = self.diff_cnn(diffs.view(B*T,C,H,W)).view(B,T,-1)
        out, (h, _) = self.lstm(f)
        ff  = self.frame_cnn(last_frame)
        # Gate: zero out height features for non-profil layers
        hf  = self.height_cnn(height_patch) * has_height
        pvf = self.pv_branch(pv)
        # Concatenate gated height + explicit gate flag
        fused = torch.cat([h[-1], out.mean(1), ff, hf, pvf,
                           has_height], dim=1)
        return self.classifier(fused)


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class EnhancedDataset(Dataset):
    def __init__(self, samples_df, source_df,
                 p_mean, p_std, v_mean, v_std,
                 Z, height_lookup, frame_xy,
                 augment=False):
        self.df          = samples_df.reset_index(drop=True)
        self.p_mean      = float(p_mean)
        self.p_std       = float(p_std)  if p_std  != 0 else 1.
        self.v_mean      = float(v_mean)
        self.v_std       = float(v_std)  if v_std  != 0 else 1.
        self.Z           = Z              # full profilometry matrix
        self.hlookup     = height_lookup  # layer -> {h_mean, h_std, ...}
        self.frame_xy    = frame_xy       # (layer,frame) -> (col, row)
        self.augment     = augment
        self.fmap        = {}
        for layer, g in source_df.groupby("layer"):
            g = g.sort_values("frame")
            self.fmap[int(layer)] = {
                int(r.frame): str(r.image_path)
                for r in g.itertuples(index=False)
            }

    def _load(self, path):
        img = Image.open(path)
        x   = TF.pil_to_tensor(img).float()
        x   = torch.clamp(x, 0, 65535) / 65535.
        x   = TF.resize(x, list(IMAGE_SIZE), antialias=True)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return x

    def _get_height_patch(self, layer, center_frame):
        """
        Extract 2D height patch centered at (col, row) for
        this layer and frame. Returns zeros if no profil data.
        """
        if self.Z is None or layer not in self.hlookup:
            return torch.zeros(1, PATCH_OUT, PATCH_OUT,
                               dtype=torch.float32)
        key = (layer, center_frame)
        if key not in self.frame_xy:
            return torch.zeros(1, PATCH_OUT, PATCH_OUT,
                               dtype=torch.float32)
        col_c, row_c = self.frame_xy[key]
        cfg = self.hlookup[layer]
        return extract_height_patch(
            self.Z, col_c, row_c,
            cfg["h_mean"], cfg["h_std"],
            cfg["n_rows"], cfg["n_cols"]
        )

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        layer = int(row["layer"])
        start = int(row["start_frame"])
        fmap  = self.fmap[layer]
        all_f = sorted(fmap.keys())
        mn, mx = all_f[0], all_f[-1]

        frames = [max(mn, min(mx, start+i)) for i in range(SEQ_LEN)]
        imgs   = [self._load(fmap[f]) for f in frames]
        diffs  = torch.stack(
            [torch.abs(imgs[t]-imgs[t-1]) for t in range(1, SEQ_LEN)],
            dim=0
        )
        d_max = diffs.max()
        if d_max > 1e-8: diffs = diffs / d_max
        last = (imgs[-1] - 0.5) / 0.5

        if self.augment and random.random() > 0.5:
            diffs = torch.flip(diffs, dims=[3])
            last  = torch.flip(last,  dims=[2])

        p_n = (float(row["P"]) - self.p_mean) / self.p_std
        v_n = (float(row["V"]) - self.v_mean) / self.v_std
        pv  = torch.tensor([p_n, v_n], dtype=torch.float32)

        # 2D height patch centered on window's center frame
        center_f     = frames[SEQ_LEN // 2]
        height_patch = self._get_height_patch(layer, center_f)

        # Gate scalar: 1.0 if this layer has real profil data
        has_h = torch.tensor(
            [[1.0]] if layer in PROFIL_LAYERS else [[0.0]],
            dtype=torch.float32
        ).squeeze(0)   # shape: (1,)

        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return diffs, last, pv, height_patch, has_h, y


# ─────────────────────────────────────────────
# SAMPLE BUILDERS WITH EDGE TRIMMING
# ─────────────────────────────────────────────

def get_valid_frames(g):
    """
    Apply edge trimming: remove first TRIM_START and last TRIM_END frames.
    Returns the set of valid frame numbers.
    """
    all_f = sorted(g["frame"].astype(int).tolist())
    if len(all_f) <= TRIM_START + TRIM_END:
        return set(all_f)   # too few frames — keep all
    valid = all_f[TRIM_START : len(all_f) - TRIM_END]
    return set(valid)


def build_samples(split_df, neg_ratio=1.0, seed=42):
    rng  = random.Random(seed)
    rows = []
    for layer, g in split_df.groupby("layer"):
        g     = g.sort_values("frame").reset_index(drop=True)
        valid = get_valid_frames(g)
        all_f = sorted(valid)
        if len(all_f) < SEQ_LEN:
            continue
        mf = all_f[0]; xf = all_f[-1]
        p  = float(g["P"].iloc[0]); v = float(g["V"].iloc[0])
        ball = set(g[g["balling"]==1]["frame"].astype(int)) & valid
        half = SEQ_LEN // 2
        pos  = set()
        for fn in ball:
            ws = max(mf, min(fn-half, xf-SEQ_LEN+1))
            seq = range(ws, ws+SEQ_LEN)
            if all(f in valid for f in seq) and \
               any(f in ball for f in seq):
                pos.add(ws)
        for ws in pos:
            rows.append({"layer":int(layer),"start_frame":ws,
                          "label":1,"P":p,"V":v})
        negs = [s for s in range(mf, xf-SEQ_LEN+2)
                if all(f in valid for f in range(s, s+SEQ_LEN)) and
                   all(f not in ball for f in range(s, s+SEQ_LEN))]
        n_neg = max(int(round(len(pos)*neg_ratio)), 1)
        if negs:
            ch = rng.sample(negs, k=min(n_neg, len(negs)))
            while len(ch) < n_neg: ch.append(rng.choice(negs))
            for ws in ch:
                rows.append({"layer":int(layer),"start_frame":ws,
                              "label":0,"P":p,"V":v})
    return pd.DataFrame(rows).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def build_full_samples(layer_df):
    rows = []
    for layer, g in layer_df.groupby("layer"):
        g     = g.sort_values("frame").reset_index(drop=True)
        valid = get_valid_frames(g)
        all_f = sorted(valid)
        if len(all_f) < SEQ_LEN:
            continue
        mf = all_f[0]; xf = all_f[-1]
        p  = float(g["P"].iloc[0]); v = float(g["V"].iloc[0])
        ball = set(g[g["balling"]==1]["frame"].astype(int)) & valid
        for start in range(mf, xf-SEQ_LEN+2):
            if not all(f in valid for f in range(start, start+SEQ_LEN)):
                continue
            label = int(any(f in ball for f in range(start, start+SEQ_LEN)))
            cf    = start + SEQ_LEN // 2
            cr    = g[g["frame"]==cf]
            if cr.empty:
                cr = g.iloc[[(g["frame"]-cf).abs().argmin()]]
            r = cr.iloc[0]
            rows.append({
                "layer"       : int(layer),
                "start_frame" : start,
                "center_frame": int(cf),
                "label"       : label,
                "P": p, "V": v,
                "bead_type"   : r.get("bead_type"),
                "bead_size"   : r.get("bead_size_vis"),
                "bead_size_mm": float(r["bead_size_mm"])
                                if pd.notna(r.get("bead_size_mm"))
                                else np.nan,
                "n_beads"     : int(r.get("n_beads", 0)),
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def best_threshold(probs, targets):
    best_t, best_f1 = 0.5, -1.
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(targets, (probs>=t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, float(t)
    return best_t, best_f1


def run_epoch(model, loader, optimizer=None, desc=""):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    all_p, all_t = [], []
    total_loss = 0.
    pbar = tqdm(loader, desc=desc, leave=False,
                ncols=80, disable=not is_train)
    for diffs, lf, pv, hp, hg, y in pbar:
        diffs = diffs.to(device); lf = lf.to(device)
        pv    = pv.to(device);    hp = hp.to(device)
        hg    = hg.to(device)
        y_dev = y.to(device).unsqueeze(1)
        if is_train: optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(diffs, lf, pv, hp, hg)
            loss   = F.binary_cross_entropy_with_logits(logits, y_dev)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.3f}")
        total_loss += loss.item() * diffs.size(0)
        all_p.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_t.extend((y.numpy()>=0.5).astype(int))
    pa, ta = np.array(all_p), np.array(all_t)
    try:    auc = roc_auc_score(ta, pa)
    except: auc = float("nan")
    return {"probs":pa, "targets":ta, "auc":auc,
            "f1":f1_score(ta,(pa>=0.5).astype(int),zero_division=0),
            "loss":total_loss/max(len(loader.dataset),1)}


def evaluate_strata(pred_df):
    clean = pred_df[pred_df["label"]==0]
    strata = {
        "Overall"    : pred_df["label"]==1,
        "Type=Middle": pred_df["bead_type"]=="Middle",
        "Type=Left"  : pred_df["bead_type"]=="Left",
        "Type=Right" : pred_df["bead_type"]=="Right",
        "Size=L"     : pred_df["bead_size"]=="L",
        "Size=M"     : pred_df["bead_size"]=="M",
        "Size=S"     : pred_df["bead_size"]=="S",
    }
    results = {}
    for name, mask in strata.items():
        pos = pred_df[mask & (pred_df["label"]==1)]
        if len(pos) < 2:
            results[name] = {"n_pos":len(pos),"auc":np.nan}
            continue
        ev = pd.concat([pos, clean], ignore_index=True)
        try:
            auc = roc_auc_score(
                (ev["label"]==1).astype(int).values,
                ev["pred_prob"].values
            )
        except: auc = np.nan
        results[name] = {"n_pos":len(pos),"auc":auc}
    return results


# ─────────────────────────────────────────────
# ONE LOLO FOLD
# ─────────────────────────────────────────────

def train_fold(held_out, enriched_df, Z_profil, height_lookup, frame_xy):
    train_layers = [l for l in ALL_LAYERS if l != held_out]
    train_df = enriched_df[
        enriched_df["layer"].isin(train_layers)
    ].reset_index(drop=True)
    test_df  = enriched_df[
        enriched_df["layer"]==held_out
    ].reset_index(drop=True)

    p_mean = float(train_df["P"].mean())
    p_std  = float(train_df["P"].std())
    v_mean = float(train_df["V"].mean())
    v_std  = float(train_df["V"].std())
    for s, n in [(p_std,"P"),(v_std,"V")]:
        if pd.isna(s) or s==0:
            if n=="P": p_std=1.
            else:      v_std=1.

    # Smart val: pick layer most similar in (P, V) to test layer
    test_p = float(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
    test_v = float(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])

    def pv_dist(l):
        lp = float(enriched_df[enriched_df["layer"]==l]["P"].iloc[0])
        lv = float(enriched_df[enriched_df["layer"]==l]["V"].iloc[0])
        # Normalize: P range ~140W, V range ~200mm/s
        return ((lp - test_p)/140)**2 + ((lv - test_v)/200)**2

    val_layer = min(train_layers, key=pv_dist)
    pure_tr   = [l for l in train_layers if l != val_layer]
    print(f"    Val layer: L{val_layer} "
          f"(closest P/V to L{held_out})")

    tr_df = enriched_df[
        enriched_df["layer"].isin(pure_tr)
    ].reset_index(drop=True)
    vl_df = enriched_df[
        enriched_df["layer"]==val_layer
    ].reset_index(drop=True)

    tr_samp = build_samples(tr_df, seed=42)
    vl_samp = build_samples(vl_df, seed=43)

    kw = dict(p_mean=p_mean, p_std=p_std,
              v_mean=v_mean, v_std=v_std,
              Z=Z_profil, height_lookup=height_lookup,
              frame_xy=frame_xy)

    tr_ds = EnhancedDataset(tr_samp, tr_df, **kw, augment=True)
    vl_ds = EnhancedDataset(vl_samp, vl_df, **kw, augment=False)
    tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl_loader = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = EnhancedDetector(
        feature_dim=64, lstm_hidden=64, n_layers=2,
        pv_hidden=16, height_dim=16, dropout=0.35
    ).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_val_auc, no_imp = -1., 0
    ckpt_path = LOLO_DIR / f"fold_{held_out}.pth"
    torch.save({"state":model.state_dict(),"thresh":0.5}, ckpt_path)

    print(f"    Training: {len(tr_samp)} samples  "
          f"Val: {len(vl_samp)} samples  "
          f"Val layer: L{val_layer}")
    for epoch in range(1, MAX_EPOCHS+1):
        tr_m = run_epoch(model, tr_loader, opt,
                         desc=f"Ep{epoch:02d} train")
        vl_m = run_epoch(model, vl_loader,
                         desc=f"Ep{epoch:02d} val  ")
        t, _ = best_threshold(vl_m["probs"], vl_m["targets"])
        sched.step(vl_m["auc"])

        improved = (not np.isnan(vl_m["auc"]) and
                    vl_m["auc"] > best_val_auc)
        marker = " *" if improved else ""
        print(f"    Ep{epoch:02d}: "
              f"tr_auc={tr_m['auc']:.3f} "
              f"vl_auc={vl_m['auc']:.3f} "
              f"vl_f1={vl_m['f1']:.3f} "
              f"no_imp={no_imp}{marker}",
              flush=True)

        if improved:
            best_val_auc, no_imp = vl_m["auc"], 0
            torch.save({"state":model.state_dict(),"thresh":t}, ckpt_path)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

        if not np.isnan(tr_m["auc"]) and tr_m["auc"] >= MAX_TRAIN_AUC:
            print(f"    Train AUC ceiling reached at epoch {epoch}")
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])

    full    = build_full_samples(test_df)
    full_ds = EnhancedDataset(full, test_df, **kw, augment=False)
    fl_load = DataLoader(full_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval(); probs = []
    with torch.no_grad():
        for di, lf, pv, hp, hg, _ in fl_load:
            logits = model(di.to(device), lf.to(device),
                           pv.to(device), hp.to(device),
                           hg.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())

    full["pred_prob"] = probs
    try:    overall_auc = roc_auc_score(full["label"], full["pred_prob"])
    except: overall_auc = float("nan")

    strata  = evaluate_strata(full)
    p_val   = int(test_df["P"].iloc[0])
    v_val   = int(test_df["V"].iloc[0])
    n_ball  = int((test_df["balling"]==1).sum())
    has_h   = held_out in PROFIL_LAYERS

    print(f"  L{held_out} P={p_val}W V={v_val} "
          f"n_ball={n_ball} height={'YES' if has_h else 'no'}: "
          f"AUC={overall_auc:.3f}")
    for s, r in strata.items():
        if not np.isnan(r["auc"]) and r["n_pos"] >= 2:
            print(f"    {s:<18} n={r['n_pos']:>3}  AUC={r['auc']:.3f}")

    return {
        "held_out"   : held_out,
        "P": p_val, "V": v_val,
        "overall_auc": overall_auc,
        "has_height" : has_h,
        "strata"     : strata,
        "predictions": full,
    }


# ─────────────────────────────────────────────
# COMPARISON
# ─────────────────────────────────────────────

def compare_results(new_results, baseline_csv):
    print("\n" + "="*70)
    print("COMPARISON: Enhanced vs Baseline (lolo_PV)")
    print("Experiment A: gated height patch + smart val selection")
    print("="*70)

    if not baseline_csv.exists():
        print(f"  Baseline not found: {baseline_csv}")
        return

    old_df  = pd.read_csv(baseline_csv)
    old_auc = {}
    for layer in ALL_LAYERS:
        sub = old_df[old_df["held_out"]==layer]
        if sub.empty: continue
        clean = sub[sub["label"]==0]; ball = sub[sub["label"]==1]
        if len(ball)<2 or len(clean)<2: continue
        ev = pd.concat([ball,clean], ignore_index=True)
        try:
            old_auc[layer] = roc_auc_score(
                (ev["label"]==1).astype(int).values,
                ev["pred_prob"].values
            )
        except: old_auc[layer] = np.nan

    print(f"\n  {'Layer':>7} {'P':>5} {'V':>6} {'Height':>7} "
          f"{'Baseline':>10} {'Enhanced':>10} {'Delta':>8}")
    print("  " + "-"*60)

    deltas = []; deltas_h = []; deltas_nh = []
    for r in sorted(new_results, key=lambda x: x["held_out"]):
        layer   = r["held_out"]
        new_auc = r["overall_auc"]
        old_a   = old_auc.get(layer, np.nan)
        delta   = new_auc - old_a if not (
            np.isnan(new_auc) or np.isnan(old_a)
        ) else np.nan
        h_flag  = "YES" if r["has_height"] else " no"
        old_s   = f"{old_a:.3f}" if not np.isnan(old_a) else "  n/a"
        new_s   = f"{new_auc:.3f}" if not np.isnan(new_auc) else "  n/a"
        dlt_s   = f"{delta:+.3f}" if not np.isnan(delta) else "  n/a"
        marker  = " <-- profil" if r["has_height"] else ""
        print(f"  {layer:>7} {r['P']:>5} {r['V']:>6} {h_flag:>7} "
              f"{old_s:>10} {new_s:>10} {dlt_s:>8}{marker}")
        if not np.isnan(delta):
            deltas.append(delta)
            if r["has_height"]: deltas_h.append(delta)
            else:               deltas_nh.append(delta)

    print(f"\n  SUMMARY:")
    print(f"    All layers   : mean delta = {np.mean(deltas):+.3f}  "
          f"({sum(d>0 for d in deltas)}/{len(deltas)} improved)")
    if deltas_h:
        print(f"    With height  : mean delta = {np.mean(deltas_h):+.3f}  "
              f"({sum(d>0 for d in deltas_h)}/{len(deltas_h)} improved)")
    if deltas_nh:
        print(f"    Without height: mean delta = {np.mean(deltas_nh):+.3f}  "
              f"({sum(d>0 for d in deltas_nh)}/{len(deltas_nh)} improved)")

    # Figure
    layers_s  = sorted(new_results, key=lambda x: x["held_out"])
    x         = np.arange(len(layers_s))
    new_aucs  = [r["overall_auc"] for r in layers_s]
    old_aucs  = [old_auc.get(r["held_out"], np.nan) for r in layers_s]
    xlabels   = [f"L{r['held_out']}\n{r['P']}W"
                  + ("\n[H]" if r["has_height"] else "")
                  for r in layers_s]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "Experiment A vs Baseline\n"
        "Gated height patch (has_height gate) + smart val layer\n"
        "No edge trimming  |  [H] = layer has profilometry data",
        fontsize=10, fontweight="bold"
    )
    w = 0.35
    ax = axes[0]
    ax.bar(x-w/2, old_aucs, w, label="Baseline (lolo_PV)",
           color="#1565C0", alpha=0.8, edgecolor="black", linewidth=0.7)
    ax.bar(x+w/2, new_aucs, w, label="Enhanced",
           color="#E53935", alpha=0.8, edgecolor="black", linewidth=0.7)
    ax.axhline(0.5,  color="black", linewidth=1.2, linestyle="--")
    ax.axhline(0.65, color="green", linewidth=1,   linestyle=":")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("AUC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Per-layer AUC"); ax.legend(fontsize=9)

    ax = axes[1]
    delta_vals = [n-o if not (np.isnan(n) or np.isnan(o)) else 0
                  for n, o in zip(new_aucs, old_aucs)]
    colors = ["#2E7D32" if d>0.01 else
              "#C62828" if d<-0.01 else "#888888"
              for d in delta_vals]
    ax.bar(x, delta_vals, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Delta AUC (Enhanced - Baseline)")
    ax.set_title("Improvement\nGreen=better  Red=worse  Gray=similar")

    plt.tight_layout()
    path = OUTPUT_DIR / "lolo_enhanced_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading enriched_df...")
    enriched_df = pd.read_csv(
        r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv",
        encoding="utf-8-sig", low_memory=False
    )
    for col in ["frame","balling","event_id","bead_id_rich",
                "n_beads","layer","P","V"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(
                enriched_df[col], errors="coerce"
            ).astype("Int64")
    for col in ["bead_size_mm","start_pixel","end_pixel","size_actual"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(
                enriched_df[col], errors="coerce"
            ).astype(float)
    for col in ["bead_type","bead_size_vis","image_path"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].where(
                enriched_df[col].astype(str) != "nan", other=None
            )
    enriched_df["is_multi_bead"] = enriched_df["is_multi_bead"].astype(bool)

    total_frames = len(enriched_df)
    trimmed = 0
    for layer, g in enriched_df.groupby("layer"):
        valid = get_valid_frames(g)
        trimmed += len(g) - len(valid)

    print(f"  {total_frames} total frames")
    print(f"  Edge trim: first {TRIM_START} + last {TRIM_END} per layer")
    print(f"  Frames trimmed: {trimmed} ({100*trimmed/total_frames:.1f}%)")
    print(f"  Frames remaining: {total_frames - trimmed}")

    # Build profilometry height lookup
    print("\nBuilding profilometry 2D height patch feature...")
    Z_profil, height_lookup, frame_xy = build_height_lookup()
    n_frames_with_patch = len(frame_xy)
    print(f"  Frame->pixel mappings built: {n_frames_with_patch}")
    print(f"  Patch size: {PATCH_HALF_Y*2+1}x{PATCH_HALF_X*2+1} pixels")
    print(f"    X: +/-{PATCH_HALF_X}px = +/-{PATCH_HALF_X*PX_X_MM*1000:.0f}um along scan")
    print(f"    Y: +/-{PATCH_HALF_Y}px = +/-{PATCH_HALF_Y*PX_Y_MM*1000:.0f}um cross-track")
    print(f"  Downsampled to: {PATCH_OUT}x{PATCH_OUT} for CNN input")
    print(f"  Profilometry layers: {sorted(PROFIL_LAYERS)}")
    print(f"  Zero patch layers:   {sorted(set(ALL_LAYERS)-PROFIL_LAYERS)}")

    # Run LOLO
    print("\n" + "="*65)
    print("ENHANCED LOLO — 14 folds")
    print(f"  Improvement 1: gated height patch (has_height gate scalar)")
    print(f"  Improvement 2: smart validation layer selection")
    print(f"  Refinement 2: profilometry height for L248-L252")
    print("="*65)

    all_results = []
    for held_out in sorted(ALL_LAYERS):
        p = int(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
        v = int(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])
        n = int((enriched_df[enriched_df["layer"]==held_out]["balling"]==1).sum())
        print(f"\nFold: L{held_out} P={p}W V={v} n_ball={n}")
        result = train_fold(held_out, enriched_df, Z_profil, height_lookup, frame_xy)
        all_results.append(result)
        pd.to_pickle(all_results, LOLO_DIR / "intermediate.pkl")

    # Save predictions
    all_preds = pd.concat(
        [r["predictions"].assign(held_out=r["held_out"])
         for r in all_results],
        ignore_index=True
    )
    all_preds.to_csv(LOLO_DIR / "predictions.csv", index=False)

    # Compare
    compare_results(all_results, BASELINE_CSV)

    # Mean AUC summary
    aucs = [r["overall_auc"] for r in all_results
            if not np.isnan(r["overall_auc"])]
    print(f"\n  Mean AUC (enhanced): {np.mean(aucs):.3f}")
    print(f"  Baseline mean AUC  : 0.673")
    print(f"  Net improvement    : {np.mean(aucs)-0.673:+.3f}")
    print(f"\nOutputs: {LOLO_DIR}/")
