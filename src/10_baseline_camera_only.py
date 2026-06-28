"""
Level 1: Add V as additional scalar input alongside P
======================================================
Change from previous model:
  - Process branch input: Linear(1, 8) → Linear(2, 8)
  - Both P and V are normalized and passed as [P_norm, V_norm]
  - Everything else identical to the LOLO experiment

This gives the model an explicit signal about scan speed regime,
allowing it to learn different decision boundaries for V=1800 vs V=2000
without fundamentally changing the architecture.

Expected benefit:
  - Better generalization across V values in LOLO
  - Model can down-weight features that are V-specific
  - Cleaner separation for layers where V is the key variable

Output: lolo_PV/ directory with full LOLO results
Comparison printed at end: LOLO-P vs LOLO-PV per layer AUC
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("extended_analysis")
LOLO_DIR   = OUTPUT_DIR / "lolo_PV"
LOLO_DIR.mkdir(parents=True, exist_ok=True)

# Previous LOLO results (P only) for comparison
PREV_LOLO_CSV = OUTPUT_DIR / "lolo_14" / "lolo_14_predictions.csv"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ALL_LAYERS = list(range(226, 232)) + list(range(245, 253))

SEQ_LEN      = 6   # updated from 8 — P75 of event durations is 6 frames
IMAGE_SIZE   = (128, 128)
BATCH_SIZE   = 8
MAX_EPOCHS   = 60
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-3


# ─────────────────────────────────────────────
# MODEL — process branch now takes [P, V]
# ─────────────────────────────────────────────

class DiffFrameCNN(nn.Module):
    def __init__(self, feature_dim=64, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.proj(self.encoder(x))


class DiffImageDetectorPV(nn.Module):
    """
    Same as DiffImageDetector but process branch takes [P_norm, V_norm]
    instead of just [P_norm].
    Key change: nn.Linear(1, pv_hidden) → nn.Linear(2, pv_hidden)
    """
    def __init__(self, feature_dim=64, lstm_hidden=64,
                 n_layers=2, pv_hidden=16, dropout=0.35):
        super().__init__()
        self.diff_cnn = DiffFrameCNN(feature_dim, dropout)
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(inplace=True), nn.MaxPool2d(4),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32*4*4, 32),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(feature_dim, lstm_hidden, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)

        # KEY CHANGE: input is 2 (P and V) instead of 1 (P only)
        self.pv_branch = nn.Sequential(
            nn.Linear(2, pv_hidden),   # ← was Linear(1, p_hidden)
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        fused = lstm_hidden * 2 + 32 + pv_hidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, diffs, last_frame, pv):
        """
        pv: (B, 2) — [P_norm, V_norm]
        """
        B, T, C, H, W = diffs.shape
        f  = self.diff_cnn(diffs.view(B*T, C, H, W)).view(B, T, -1)
        out, (h, _) = self.lstm(f)
        frame_feat  = self.frame_cnn(last_frame)
        pv_feat     = self.pv_branch(pv)
        fused = torch.cat([h[-1], out.mean(1), frame_feat, pv_feat], dim=1)
        return self.classifier(fused)


# ─────────────────────────────────────────────
# DATASET — now returns [P_norm, V_norm]
# ─────────────────────────────────────────────

class DiffDatasetPV(Dataset):
    def __init__(self, samples_df, source_df,
                 p_mean=0.0, p_std=1.0,
                 v_mean=0.0, v_std=1.0,
                 augment=False):
        self.samples_df = samples_df.reset_index(drop=True)
        self.p_mean = float(p_mean)
        self.p_std  = float(p_std)  if p_std  != 0 else 1.0
        self.v_mean = float(v_mean)
        self.v_std  = float(v_std)  if v_std  != 0 else 1.0
        self.augment = augment
        self.fmap = {}
        for layer, g in source_df.groupby("layer"):
            g = g.sort_values("frame")
            self.fmap[int(layer)] = {
                int(r.frame): str(r.image_path)
                for r in g.itertuples(index=False)
            }

    def _load(self, path):
        img = Image.open(path)
        x   = TF.pil_to_tensor(img).float()
        x   = torch.clamp(x, 0, 65535) / 65535.0
        x   = TF.resize(x, list(IMAGE_SIZE), antialias=True)
        if x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)
        return x

    def __len__(self):
        return len(self.samples_df)

    def __getitem__(self, idx):
        row   = self.samples_df.iloc[idx]
        layer = int(row["layer"])
        start = int(row["start_frame"])
        fmap  = self.fmap[layer]
        all_f = sorted(fmap.keys())
        mn, mx = all_f[0], all_f[-1]

        frames = [max(mn, min(mx, start+i)) for i in range(SEQ_LEN)]
        imgs   = [self._load(fmap[f]) for f in frames]
        diffs  = torch.stack([torch.abs(imgs[t]-imgs[t-1])
                               for t in range(1, SEQ_LEN)], dim=0)
        d_max = diffs.max()
        if d_max > 1e-8:
            diffs = diffs / d_max
        last = (imgs[-1] - 0.5) / 0.5

        if self.augment and random.random() > 0.5:
            diffs = torch.flip(diffs, dims=[3])
            last  = torch.flip(last,  dims=[2])

        p_val = float(row["P"])
        v_val = float(row["V"])

        # KEY CHANGE: return 2D process vector [P_norm, V_norm]
        pv = torch.tensor([
            (p_val - self.p_mean) / self.p_std,
            (v_val - self.v_mean) / self.v_std,
        ], dtype=torch.float32)

        return diffs, last, pv, torch.tensor(
            float(row["label"]), dtype=torch.float32
        )


# ─────────────────────────────────────────────
# SAMPLE BUILDERS (identical to before)
# ─────────────────────────────────────────────

def build_samples(split_df, neg_ratio=1.0, seed=42):
    rng = random.Random(seed)
    rows = []
    for layer, g in split_df.groupby("layer"):
        g  = g.sort_values("frame").reset_index(drop=True)
        mf = int(g["frame"].min())
        xf = int(g["frame"].max())
        p  = float(g["P"].iloc[0])
        v  = float(g["V"].iloc[0])
        ball = set(g[g["balling"]==1]["frame"].astype(int))
        half = SEQ_LEN // 2
        pos  = set()
        for fn in ball:
            ws = max(mf, min(fn-half, xf-SEQ_LEN+1))
            if any(f in ball for f in range(ws, ws+SEQ_LEN)):
                pos.add(ws)
        for ws in pos:
            rows.append({"layer": int(layer), "start_frame": ws,
                          "label": 1, "P": p, "V": v})
        negs = [s for s in range(mf, xf-SEQ_LEN+2)
                if all(f not in ball for f in range(s, s+SEQ_LEN))]
        n_neg = max(int(round(len(pos)*neg_ratio)), 1)
        if negs:
            ch = rng.sample(negs, k=min(n_neg, len(negs)))
            while len(ch) < n_neg:
                ch.append(rng.choice(negs))
            for ws in ch:
                rows.append({"layer": int(layer), "start_frame": ws,
                              "label": 0, "P": p, "V": v})
    return pd.DataFrame(rows).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)


def build_full_samples(layer_df):
    rows = []
    for layer, g in layer_df.groupby("layer"):
        g  = g.sort_values("frame").reset_index(drop=True)
        mf = int(g["frame"].min())
        xf = int(g["frame"].max())
        p  = float(g["P"].iloc[0])
        v  = float(g["V"].iloc[0])
        ball = set(g[g["balling"]==1]["frame"].astype(int))
        for start in range(mf, xf-SEQ_LEN+2):
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
                "P"           : p,
                "V"           : v,
                "bead_type"   : r.get("bead_type"),
                "bead_size"   : r.get("bead_size_vis"),
                "bead_size_mm": float(r["bead_size_mm"])
                                if pd.notna(r.get("bead_size_mm")) else np.nan,
                "n_beads"     : int(r.get("n_beads", 0)),
            })
    return pd.DataFrame(rows).reset_index(drop=True)


# ─────────────────────────────────────────────
# TRAINING UTILITIES
# ─────────────────────────────────────────────

def best_threshold(probs, targets):
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(targets, (probs >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_t, best_f1


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    all_p, all_t = [], []
    for diffs, lf, pv, y in loader:
        diffs = diffs.to(device)
        lf    = lf.to(device)
        pv    = pv.to(device)
        y_dev = y.to(device).unsqueeze(1)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(diffs, lf, pv)
            loss   = F.binary_cross_entropy_with_logits(logits, y_dev)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        all_p.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel())
        all_t.extend((y.numpy() >= 0.5).astype(int))
    pa, ta = np.array(all_p), np.array(all_t)
    metrics = {"probs": pa, "targets": ta}
    try:
        metrics["auc"] = roc_auc_score(ta, pa)
    except Exception:
        metrics["auc"] = float("nan")
    metrics["f1"] = f1_score(
        ta, (pa >= 0.5).astype(int), zero_division=0
    )
    return metrics


def evaluate_strata(pred_df):
    clean = pred_df[pred_df["label"] == 0]
    strata = [
        ("Overall",     pred_df["label"] == 1),
        ("Type=Middle", pred_df["bead_type"] == "Middle"),
        ("Type=Left",   pred_df["bead_type"] == "Left"),
        ("Type=Right",  pred_df["bead_type"] == "Right"),
        ("Size=L",      pred_df["bead_size"] == "L"),
        ("Size=M",      pred_df["bead_size"] == "M"),
        ("Size=S",      pred_df["bead_size"] == "S"),
        ("Multi-bead",  pred_df["n_beads"] > 1),
    ]
    results = {}
    for name, mask in strata:
        pos = pred_df[mask & (pred_df["label"] == 1)]
        if len(pos) < 2:
            results[name] = {"n_pos": len(pos), "auc": np.nan}
            continue
        ev  = pd.concat([pos, clean], ignore_index=True)
        y_t = (ev["label"]==1).astype(int).values
        y_p = ev["pred_prob"].values
        try:
            auc = roc_auc_score(y_t, y_p)
        except Exception:
            auc = np.nan
        results[name] = {"n_pos": len(pos), "auc": auc}
    return results


# ─────────────────────────────────────────────
# LOLO — one fold
# ─────────────────────────────────────────────

def train_fold_pv(held_out, df):
    train_layers = [l for l in ALL_LAYERS if l != held_out]
    train_df = df[df["layer"].isin(train_layers)].reset_index(drop=True)
    test_df  = df[df["layer"] == held_out].reset_index(drop=True)

    p_mean = float(train_df["P"].mean())
    p_std  = float(train_df["P"].std())
    v_mean = float(train_df["V"].mean())
    v_std  = float(train_df["V"].std())
    for std_val, name in [(p_std, "P"), (v_std, "V")]:
        if pd.isna(std_val) or std_val == 0:
            if name == "P": p_std = 1.0
            else:           v_std = 1.0

    val_layer = random.choice(train_layers)
    pure_tr   = [l for l in train_layers if l != val_layer]

    tr_df = df[df["layer"].isin(pure_tr)].reset_index(drop=True)
    vl_df = df[df["layer"] == val_layer].reset_index(drop=True)

    tr_samp = build_samples(tr_df, seed=42)
    vl_samp = build_samples(vl_df, seed=43)

    tr_ds = DiffDatasetPV(tr_samp, tr_df, p_mean, p_std,
                           v_mean, v_std, augment=True)
    vl_ds = DiffDatasetPV(vl_samp, vl_df, p_mean, p_std,
                           v_mean, v_std, augment=False)

    tr_loader = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    vl_loader = DataLoader(vl_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model = DiffImageDetectorPV(
        feature_dim=64, lstm_hidden=64, n_layers=2,
        pv_hidden=16, dropout=0.35
    ).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, "max", factor=0.5, patience=5, min_lr=1e-5
    )

    best_f1, best_thresh, no_imp = -1.0, 0.5, 0
    best_path = LOLO_DIR / f"fold_{held_out}.pth"

    for epoch in range(1, MAX_EPOCHS + 1):
        run_epoch(model, tr_loader, opt)
        vm   = run_epoch(model, vl_loader)
        t, f = best_threshold(vm["probs"], vm["targets"])
        sched.step(f)
        if f > best_f1:
            best_f1, best_thresh, no_imp = f, t, 0
            torch.save({
                "state" : model.state_dict(),
                "thresh": t,
                "p_mean": p_mean, "p_std": p_std,
                "v_mean": v_mean, "v_std": v_std,
            }, best_path)
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])

    full    = build_full_samples(test_df)
    full_ds = DiffDatasetPV(full, test_df, p_mean, p_std,
                             v_mean, v_std, augment=False)
    fl_load = DataLoader(full_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    probs = []
    with torch.no_grad():
        for di, lf, pv, _ in fl_load:
            logits = model(di.to(device), lf.to(device), pv.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().ravel())

    full["pred_prob"] = probs
    try:
        overall_auc = roc_auc_score(full["label"], full["pred_prob"])
    except Exception:
        overall_auc = np.nan

    strata  = evaluate_strata(full)
    p_val   = int(test_df["P"].iloc[0])
    v_val   = int(test_df["V"].iloc[0])
    n_ball  = int((test_df["balling"]==1).sum())

    print(f"  L{held_out} P={p_val}W V={v_val} "
          f"n_ball={n_ball}: AUC={overall_auc:.3f}")
    for s, r in strata.items():
        if not np.isnan(r["auc"]) and r["n_pos"] >= 2:
            print(f"    {s:<18} n={r['n_pos']:>3}  AUC={r['auc']:.3f}")

    return {
        "held_out"   : held_out,
        "P"          : p_val,
        "V"          : v_val,
        "overall_auc": overall_auc,
        "strata"     : strata,
        "predictions": full,
    }


# ─────────────────────────────────────────────
# COMPARISON PLOT: P-only vs P+V
# ─────────────────────────────────────────────

def compare_results(new_results, prev_csv_path):
    """
    Compare per-layer AUC: new model (P+V) vs old model (P only).
    """
    print("\n" + "="*65)
    print("COMPARISON: P+V model vs P-only model")
    print("="*65)

    # Load old results
    if not prev_csv_path.exists():
        print(f"  Previous results not found: {prev_csv_path}")
        return

    old_df = pd.read_csv(prev_csv_path)
    old_auc = {}
    for layer in ALL_LAYERS:
        sub = old_df[old_df["held_out"] == layer]
        if sub.empty:
            continue
        clean = sub[sub["label"] == 0]
        ball  = sub[sub["label"] == 1]
        if len(ball) < 2 or len(clean) < 2:
            continue
        ev = pd.concat([ball, clean], ignore_index=True)
        try:
            old_auc[layer] = roc_auc_score(
                (ev["label"]==1).astype(int).values,
                ev["pred_prob"].values
            )
        except Exception:
            old_auc[layer] = np.nan

    print(f"\n  {'Layer':>7} {'P':>5} {'V':>6}  "
          f"{'P-only AUC':>12}  {'P+V AUC':>10}  "
          f"{'Delta':>8}  {'Better?'}")
    print("  " + "-" * 65)

    deltas = []
    for r in sorted(new_results, key=lambda x: (x["V"], x["P"])):
        layer   = r["held_out"]
        new_auc = r["overall_auc"]
        old_a   = old_auc.get(layer, np.nan)
        delta   = new_auc - old_a if not (
            np.isnan(new_auc) or np.isnan(old_a)
        ) else np.nan
        better  = "✓" if not np.isnan(delta) and delta > 0.01 \
                  else ("✗" if not np.isnan(delta) and delta < -0.01
                        else "≈")
        old_str = f"{old_a:.3f}" if not np.isnan(old_a) else "  n/a"
        new_str = f"{new_auc:.3f}" if not np.isnan(new_auc) else "  n/a"
        dlt_str = f"{delta:+.3f}" if not np.isnan(delta) else "  n/a"
        print(f"  {layer:>7} {r['P']:>5} {r['V']:>6}  "
              f"{old_str:>12}  {new_str:>10}  "
              f"{dlt_str:>8}  {better}")
        if not np.isnan(delta):
            deltas.append(delta)

    if deltas:
        print(f"\n  Mean delta  : {np.mean(deltas):+.3f}")
        print(f"  Layers improved : "
              f"{sum(1 for d in deltas if d > 0.01)} / {len(deltas)}")
        print(f"  Layers hurt     : "
              f"{sum(1 for d in deltas if d < -0.01)} / {len(deltas)}")

    # Bar chart comparison
    layers_sorted = sorted(new_results, key=lambda x: (x["V"], x["P"]))
    x        = np.arange(len(layers_sorted))
    new_aucs = [r["overall_auc"] for r in layers_sorted]
    old_aucs = [old_auc.get(r["held_out"], np.nan) for r in layers_sorted]
    labels   = [f"L{r['held_out']}\n{r['P']}W\nV={r['V']}"
                for r in layers_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        "P+V model vs P-only model: per-layer AUC comparison\n"
        "Does adding V as explicit input improve generalization?",
        fontsize=11, fontweight="bold"
    )

    ax = axes[0]
    w  = 0.35
    ax.bar(x - w/2, old_aucs, w, label="P-only",
           color="#1565C0", alpha=0.8,
           edgecolor="black", linewidth=0.7)
    ax.bar(x + w/2, new_aucs, w, label="P+V",
           color="#E53935", alpha=0.8,
           edgecolor="black", linewidth=0.7)
    ax.axhline(0.5,  color="black", linewidth=1.2, linestyle="--")
    ax.axhline(0.65, color="green", linewidth=1,   linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Overall AUC (LOLO)", fontsize=10)
    ax.set_title("Per-layer AUC: P-only vs P+V", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0.3, 1.0)

    ax = axes[1]
    delta_vals = [n - o if not (np.isnan(n) or np.isnan(o)) else 0
                  for n, o in zip(new_aucs, old_aucs)]
    colors = ["#2E7D32" if d > 0.01 else
              "#C62828"  if d < -0.01 else
              "#888888"  for d in delta_vals]
    ax.bar(x, delta_vals, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--")
    ax.axhline(0.05,  color="green", linewidth=1, linestyle=":",
               alpha=0.5, label="+0.05 threshold")
    ax.axhline(-0.05, color="red",   linewidth=1, linestyle=":",
               alpha=0.5, label="-0.05 threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Delta AUC (P+V minus P-only)", fontsize=10)
    ax.set_title("Improvement from adding V\n"
                 "Green=improved, Red=hurt, Gray=similar", fontsize=10)
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / "comparison_P_vs_PV.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {path.name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Loading enriched_df from CSV...")
    enriched_df = pd.read_csv(
        r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv",
        encoding="utf-8-sig", low_memory=False
    )
    for col in ["frame", "balling", "event_id", "bead_id_rich",
                "n_beads", "layer", "P", "V"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(
                enriched_df[col], errors="coerce"
            ).astype("Int64")
    for col in ["bead_size_mm", "start_pixel", "end_pixel", "size_actual"]:
        if col in enriched_df.columns:
            enriched_df[col] = pd.to_numeric(
                enriched_df[col], errors="coerce"
            ).astype(float)
    for col in ["bead_type", "bead_size_vis", "image_path"]:
        if col in enriched_df.columns:
            enriched_df[col] = enriched_df[col].where(
                enriched_df[col].astype(str) != "nan", other=None
            )
    enriched_df["is_multi_bead"] = enriched_df["is_multi_bead"].astype(bool)

    print(f"Loaded: {len(enriched_df)} frames, "
          f"{enriched_df['layer'].nunique()} layers")
    print(f"V values: {sorted(enriched_df['V'].dropna().unique().tolist())}")

    # ── Run LOLO with P+V model ───────────────────────────────────
    print("\n" + "="*65)
    print("LOLO with P+V model — 14 folds (~40 min)")
    print("="*65)

    all_results = []
    for held_out in sorted(ALL_LAYERS):
        p = int(enriched_df[enriched_df["layer"]==held_out]["P"].iloc[0])
        v = int(enriched_df[enriched_df["layer"]==held_out]["V"].iloc[0])
        n = int((enriched_df[enriched_df["layer"]==held_out]["balling"]==1).sum())
        print(f"\nFold: hold out L{held_out} "
              f"(P={p}W, V={v}mm/s, {n} balling frames)")
        result = train_fold_pv(held_out, enriched_df)
        all_results.append(result)
        pd.to_pickle(all_results, LOLO_DIR / "lolo_PV_intermediate.pkl")

    # Save predictions
    all_preds = pd.concat(
        [r["predictions"].assign(held_out=r["held_out"])
         for r in all_results],
        ignore_index=True
    )
    all_preds.to_csv(LOLO_DIR / "lolo_PV_predictions.csv", index=False)
    print(f"\nSaved predictions: {LOLO_DIR / 'lolo_PV_predictions.csv'}")

    # ── Compare with previous P-only results ─────────────────────
    compare_results(all_results, PREV_LOLO_CSV)

    # ── Type × size final table ───────────────────────────────────
    print("\n" + "="*65)
    print("TYPE × SIZE AUC TABLE (P+V model)")
    print("="*65)

    all_preds_df = all_preds.copy()
    clean_all    = all_preds_df[all_preds_df["label"] == 0]

    print(f"\n  {'Type':<10} {'Size':>5} {'N':>6} "
          f"{'AUC':>8}  {'Detectable?'}")
    print("  " + "-" * 45)
    for btype in ["Middle", "Left", "Right"]:
        for bsize in ["L", "M", "S"]:
            pos = all_preds_df[
                (all_preds_df["label"] == 1) &
                (all_preds_df["bead_type"] == btype) &
                (all_preds_df["bead_size"] == bsize)
            ]
            if len(pos) < 3:
                continue
            ev  = pd.concat([pos, clean_all], ignore_index=True)
            y_t = (ev["label"]==1).astype(int).values
            y_p = ev["pred_prob"].values
            try:
                auc = roc_auc_score(y_t, y_p)
            except Exception:
                auc = np.nan
            det = "YES ✓" if not np.isnan(auc) and auc > 0.65 \
                  else "NO ✗"
            print(f"  {btype:<10} {bsize:>5} {len(pos):>6} "
                  f"{auc:>8.3f}  {det}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
