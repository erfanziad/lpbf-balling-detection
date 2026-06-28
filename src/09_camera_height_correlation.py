"""
Profilometry ↔ Camera Label Correlation
=========================================
Joins surface height from profilometry (along-scan CSVs)
with per-frame camera balling labels from enriched_df.

Join key: x(mm) position — each camera frame has a known
x coordinate from the track CSV (x(mm) column).
The profilometry along-scan CSV has height at each x position.

Alignment parameters (solved in separate chat):
  X_OFFSET = 25.6180 mm  (machine x → profil x)
  Y_OFFSET =  5.2320 mm  (machine y → profil y)
  px_x     =  7.890 µm   (X pixel size)
  px_y     = 12.500 µm   (Y pixel size)

Layer-to-row mapping (verified):
  L248: row_center=41,  rows 1–81
  L249: row_center=141, rows 121–161
  L250: row_center=221, rows 201–241
  L251: row_center=301, rows 281–321
  L252: row_center=381, rows 361–401

Research questions:
  1. Do labeled balling frames have higher local bead height
     than clean frames?
  2. Does bead height correlate with bead type (L/M/R)?
  3. Does bead height correlate with bead size (S/M/L)?
  4. Do frames with high bead height correspond to higher
     model prediction probability (pred_prob from LOLO)?
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────
PROFIL_DIR  = Path("lpbf_outputs")   # where along_scan CSVs are
ENRICHED_CSV = Path(
    r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv"
)
TRACK_CSV_DIR = Path(
    r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2"
    r"\Erfan_balling_data_updated"
)
LOLO_PREDS = Path("extended_analysis/lolo_PV/lolo_PV_predictions.csv")
OUTPUT_DIR  = Path("extended_analysis/profilometry_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alignment constants ───────────────────────────────────────────
X_OFFSET = 25.6180   # mm
Y_OFFSET =  5.2320   # mm
PX_X_MM  =  0.00789  # mm per pixel in X
PX_Y_MM  =  0.01250  # mm per pixel in Y
MERGE_TOL_MM = 0.004  # ±4 µm matching tolerance (half a pixel)

# ── Layer mapping ─────────────────────────────────────────────────
LAYER_MAP = {
    248: {"row_center": 41,  "row_lo": 1,   "row_hi": 81,  "P": 380},
    249: {"row_center": 141, "row_lo": 121, "row_hi": 161, "P": 400},
    250: {"row_center": 221, "row_lo": 201, "row_hi": 241, "P": 420},
    251: {"row_center": 301, "row_lo": 281, "row_hi": 321, "P": 440},
    252: {"row_center": 381, "row_lo": 361, "row_hi": 401, "P": 460},
}

TARGET_LAYERS = sorted(LAYER_MAP.keys())


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading enriched_df...")
enriched_df = pd.read_csv(
    ENRICHED_CSV, encoding="utf-8-sig", low_memory=False
)
for col in ["frame","balling","layer","P","V","n_beads"]:
    if col in enriched_df.columns:
        enriched_df[col] = pd.to_numeric(
            enriched_df[col], errors="coerce"
        ).astype("Int64")
for col in ["bead_size_mm"]:
    if col in enriched_df.columns:
        enriched_df[col] = pd.to_numeric(
            enriched_df[col], errors="coerce"
        ).astype(float)
for col in ["bead_type","bead_size_vis","image_path"]:
    if col in enriched_df.columns:
        enriched_df[col] = enriched_df[col].where(
            enriched_df[col].astype(str) != "nan", other=None
        )
print(f"  {len(enriched_df)} frames")

# Load LOLO predictions if available
lolo_preds = None
if LOLO_PREDS.exists():
    lolo_preds = pd.read_csv(LOLO_PREDS)
    print(f"  LOLO predictions: {len(lolo_preds)} rows")


# ─────────────────────────────────────────────
# LOAD TRACK X COORDINATES
# ─────────────────────────────────────────────

print("\nLoading track x-coordinates...")
track_xy = {}
for layer in TARGET_LAYERS:
    csv_path = TRACK_CSV_DIR / f"L0{layer}.csv"
    if not csv_path.exists():
        print(f"  L{layer}: not found at {csv_path}")
        continue
    df_t = pd.read_csv(csv_path)
    df_t.columns = [c.strip().lower().replace("(","").replace(")","").replace(" ","_")
                    for c in df_t.columns]
    # Rename to standard names
    x_col = next((c for c in df_t.columns if "x" in c and "mm" in c), None) or \
            next((c for c in df_t.columns if c.startswith("x")), None)
    fn_col = next((c for c in df_t.columns
                   if "frame" in c), None)
    pw_col = next((c for c in df_t.columns
                   if "power" in c or c.startswith("p")), None)

    if x_col:
        df_t = df_t.rename(columns={x_col: "x_mm"})
    if fn_col:
        df_t = df_t.rename(columns={fn_col: "frame"})

    track_xy[layer] = df_t[
        [c for c in ["frame","x_mm"] if c in df_t.columns]
    ]
    print(f"  L{layer}: {len(df_t)} frames  "
          f"x={df_t['x_mm'].min():.3f}–{df_t['x_mm'].max():.3f}mm")


# ─────────────────────────────────────────────
# LOAD ALONG-SCAN PROFILOMETRY
# ─────────────────────────────────────────────

print("\nLoading along-scan profilometry CSVs...")
profil_data = {}
for layer in TARGET_LAYERS:
    csv_path = PROFIL_DIR / f"L0{layer}_along_scan.csv"
    if not csv_path.exists():
        print(f"  L{layer}: {csv_path} not found")
        continue
    df_p = pd.read_csv(csv_path)
    # Expected columns: x_mm, h_mean_mm, h_max_mm
    print(f"  L{layer}: {len(df_p)} rows  "
          f"cols={list(df_p.columns)}")
    profil_data[layer] = df_p


# ─────────────────────────────────────────────
# JOIN: CAMERA FRAME → PROFILOMETRY HEIGHT
# ─────────────────────────────────────────────

print("\nJoining camera labels with profilometry height...")

all_joined = []

for layer in TARGET_LAYERS:
    if layer not in profil_data or layer not in track_xy:
        print(f"  L{layer}: missing data — skipping")
        continue

    # Camera labels for this layer
    cam = enriched_df[
        enriched_df["layer"] == layer
    ].copy().reset_index(drop=True)

    # Add x_mm from track coordinates
    xy  = track_xy[layer]
    cam = cam.merge(xy, on="frame", how="left")

    # Convert machine x → profilometry x
    cam["x_profil_mm"] = cam["x_mm"] - X_OFFSET

    # Merge with profilometry on x_profil_mm
    prof = profil_data[layer].copy()

    # Nearest-neighbor merge within tolerance
    merged_rows = []
    for _, row in cam.iterrows():
        xp = row["x_profil_mm"]
        if pd.isna(xp):
            continue
        # Find closest profil row within tolerance
        dists = (prof["x_mm"] - xp).abs()
        idx   = dists.idxmin()
        if dists[idx] <= MERGE_TOL_MM:
            prow = prof.loc[idx]
            merged_rows.append({
                "layer"       : int(layer),
                "frame"       : int(row["frame"]),
                "x_mm"        : float(row["x_mm"]),
                "x_profil_mm" : float(xp),
                "balling"     : int(row["balling"]),
                "bead_type"   : row.get("bead_type"),
                "bead_size"   : row.get("bead_size_vis"),
                "bead_size_mm": row.get("bead_size_mm"),
                "P"           : int(row["P"]),
                "V"           : int(row["V"]),
                "h_mean_mm"   : float(prow.get("h_mean_mm", np.nan)),
                "h_max_mm"    : float(prow.get("h_max_mm", np.nan)),
            })

    if not merged_rows:
        print(f"  L{layer}: no matches found")
        continue

    df_joined = pd.DataFrame(merged_rows)

    # Add LOLO pred_prob if available
    if lolo_preds is not None:
        col = "held_out" if "held_out" in lolo_preds.columns else "layer"
        lolo_layer = lolo_preds[
            lolo_preds[col].astype(int) == layer
        ][["center_frame","pred_prob"]].rename(
            columns={"center_frame":"frame"}
        )
        df_joined = df_joined.merge(lolo_layer, on="frame", how="left")

    n_match = len(df_joined)
    n_ball  = int(df_joined["balling"].sum())
    print(f"  L{layer}: {n_match} frames matched  "
          f"({n_ball} balling)")
    all_joined.append(df_joined)

if not all_joined:
    print("ERROR: No data joined. Check paths and column names.")
    exit(1)

df = pd.concat(all_joined, ignore_index=True)
df.to_csv(OUTPUT_DIR / "profilometry_camera_joined.csv", index=False)
print(f"\nJoined dataset: {len(df)} rows saved")


# ─────────────────────────────────────────────
# ANALYSIS 1 — BALLING vs CLEAN HEIGHT
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("ANALYSIS 1: Bead height — balling vs clean")
print("="*60)

for layer in TARGET_LAYERS:
    sub = df[df["layer"] == layer]
    if len(sub) == 0:
        continue
    ball_h = sub[sub["balling"]==1]["h_max_mm"].dropna()
    clean_h = sub[sub["balling"]==0]["h_max_mm"].dropna()
    if len(ball_h) < 2 or len(clean_h) < 2:
        continue
    t, p = stats.ttest_ind(ball_h, clean_h)
    d    = (ball_h.mean() - clean_h.mean()) / \
           np.sqrt((ball_h.std()**2 + clean_h.std()**2)/2)
    print(f"  L{layer} P={LAYER_MAP[layer]['P']}W: "
          f"ball_h={ball_h.mean()*1000:.1f}µm  "
          f"clean_h={clean_h.mean()*1000:.1f}µm  "
          f"Cohen's d={d:.3f}  p={p:.4f}")


# ─────────────────────────────────────────────
# ANALYSIS 2 — HEIGHT BY BEAD TYPE
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("ANALYSIS 2: Bead height by type (Left/Middle/Right)")
print("="*60)

clean_h_all = df[df["balling"]==0]["h_max_mm"].dropna()
for btype in ["Left","Middle","Right"]:
    sub = df[(df["balling"]==1) & (df["bead_type"]==btype)]["h_max_mm"].dropna()
    if len(sub) < 2:
        continue
    d = (sub.mean() - clean_h_all.mean()) / \
        np.sqrt((sub.std()**2 + clean_h_all.std()**2)/2)
    print(f"  {btype:>8}: n={len(sub):>4}  "
          f"mean={sub.mean()*1000:.1f}µm  "
          f"Cohen's d vs clean={d:.3f}")


# ─────────────────────────────────────────────
# ANALYSIS 3 — HEIGHT BY BEAD SIZE
# ─────────────────────────────────────────────

print("\n" + "="*60)
print("ANALYSIS 3: Bead height by size (S/M/L)")
print("="*60)

for bsize in ["L","M","S"]:
    sub = df[(df["balling"]==1) & (df["bead_size"]==bsize)]["h_max_mm"].dropna()
    if len(sub) < 2:
        continue
    d = (sub.mean() - clean_h_all.mean()) / \
        np.sqrt((sub.std()**2 + clean_h_all.std()**2)/2)
    print(f"  Size={bsize}: n={len(sub):>4}  "
          f"mean={sub.mean()*1000:.1f}µm  "
          f"Cohen's d vs clean={d:.3f}")


# ─────────────────────────────────────────────
# ANALYSIS 4 — HEIGHT vs MODEL SCORE
# ─────────────────────────────────────────────

if "pred_prob" in df.columns:
    print("\n" + "="*60)
    print("ANALYSIS 4: Profilometry height vs model prediction score")
    print("="*60)
    valid = df[["h_max_mm","pred_prob"]].dropna()
    r, p  = stats.pearsonr(valid["h_max_mm"], valid["pred_prob"])
    print(f"  Pearson r (height vs pred_prob): {r:.3f}  p={p:.4f}")
    print(f"  N={len(valid)}")
    print(f"  Interpretation: {'positive' if r>0 else 'negative'} correlation — "
          f"{'higher beads → higher model score' if r>0 else 'higher beads → lower model score'}")


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────

print("\nGenerating figures...")

# Figure 1: Height distributions by label
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Bead height from profilometry vs camera labels\n"
    "Height = max height in 0.5mm cross-track band at frame X position",
    fontsize=10, fontweight="bold"
)

# Panel 1: balling vs clean
ax = axes[0]
clean_h = df[df["balling"]==0]["h_max_mm"].dropna() * 1000
ball_h  = df[df["balling"]==1]["h_max_mm"].dropna() * 1000
ax.hist(clean_h, bins=40, alpha=0.6, color="steelblue",
        density=True, label=f"Clean (n={len(clean_h)})")
ax.hist(ball_h,  bins=40, alpha=0.7, color="orange",
        density=True, label=f"Balling (n={len(ball_h)})")
ax.set_xlabel("Max height (µm)")
ax.set_ylabel("Density")
ax.set_title("Height: balling vs clean")
ax.legend(fontsize=9)

# Panel 2: by bead type
ax = axes[1]
colors_type = {"Left":"#2E7D32","Middle":"#E65100","Right":"#C62828"}
ax.hist(clean_h, bins=30, alpha=0.4, color="steelblue",
        density=True, label="Clean")
for btype, color in colors_type.items():
    sub = df[(df["balling"]==1) &
             (df["bead_type"]==btype)]["h_max_mm"].dropna() * 1000
    if len(sub) > 1:
        ax.hist(sub, bins=20, alpha=0.65, color=color,
                density=True, label=f"{btype} (n={len(sub)})")
ax.set_xlabel("Max height (µm)")
ax.set_ylabel("Density")
ax.set_title("Height by bead type")
ax.legend(fontsize=8)

# Panel 3: by bead size
ax = axes[2]
colors_size = {"L":"#1A237E","M":"#1565C0","S":"#42A5F5"}
ax.hist(clean_h, bins=30, alpha=0.4, color="steelblue",
        density=True, label="Clean")
for bsize, color in colors_size.items():
    sub = df[(df["balling"]==1) &
             (df["bead_size"]==bsize)]["h_max_mm"].dropna() * 1000
    if len(sub) > 1:
        ax.hist(sub, bins=20, alpha=0.65, color=color,
                density=True, label=f"Size={bsize} (n={len(sub)})")
ax.set_xlabel("Max height (µm)")
ax.set_ylabel("Density")
ax.set_title("Height by bead size")
ax.legend(fontsize=8)

plt.tight_layout()
path = OUTPUT_DIR / "height_distributions.png"
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")


# Figure 2: Height vs model score scatter
if "pred_prob" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Profilometry height vs CNN-LSTM model score\n"
        "Key question: does higher bead height → higher model confidence?",
        fontsize=10, fontweight="bold"
    )
    colors_label = {0: "steelblue", 1: "orange"}

    ax = axes[0]
    for ball_val, color in colors_label.items():
        sub = df[df["balling"]==ball_val][
            ["h_max_mm","pred_prob"]
        ].dropna()
        label = "Balling" if ball_val==1 else "Clean"
        ax.scatter(sub["h_max_mm"]*1000, sub["pred_prob"],
                   c=color, s=10, alpha=0.4, label=label)
    valid = df[["h_max_mm","pred_prob"]].dropna()
    r, _  = stats.pearsonr(valid["h_max_mm"], valid["pred_prob"])
    # Trend line
    z = np.polyfit(valid["h_max_mm"]*1000, valid["pred_prob"], 1)
    xfit = np.linspace(valid["h_max_mm"].min()*1000,
                       valid["h_max_mm"].max()*1000, 100)
    ax.plot(xfit, np.polyval(z, xfit), "r-", linewidth=2,
            label=f"Trend (r={r:.3f})")
    ax.set_xlabel("Max bead height (µm)")
    ax.set_ylabel("Model prediction probability")
    ax.set_title("All frames")
    ax.legend(fontsize=8)

    ax = axes[1]
    # Same but color by bead type
    for btype, color in colors_type.items():
        sub = df[(df["balling"]==1) & (df["bead_type"]==btype)][
            ["h_max_mm","pred_prob"]
        ].dropna()
        if len(sub) > 1:
            ax.scatter(sub["h_max_mm"]*1000, sub["pred_prob"],
                       c=color, s=15, alpha=0.6, label=btype)
    ax.set_xlabel("Max bead height (µm)")
    ax.set_ylabel("Model prediction probability")
    ax.set_title("Balling frames by bead type")
    ax.legend(fontsize=8)

    plt.tight_layout()
    path = OUTPUT_DIR / "height_vs_model_score.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path.name}")


# Figure 3: Along-scan height profiles with balling labels overlaid
fig, axes = plt.subplots(len(TARGET_LAYERS), 1,
                          figsize=(16, 4*len(TARGET_LAYERS)),
                          sharex=False)
fig.suptitle(
    "Along-scan height profile with balling labels overlaid\n"
    "Orange spans = labeled balling frames | "
    "Green/Red/Blue = Left/Middle/Right bead type",
    fontsize=10, fontweight="bold"
)

type_colors = {"Left":"#2E7D32","Middle":"#E65100","Right":"#C62828"}

for ax, layer in zip(axes, TARGET_LAYERS):
    if layer not in profil_data:
        ax.set_visible(False)
        continue

    prof = profil_data[layer]
    sub  = df[df["layer"]==layer].copy()

    ax.plot(prof["x_mm"], prof["h_max_mm"]*1000,
            color="steelblue", linewidth=1, alpha=0.8,
            label="Max height (µm)")
    ax.fill_between(prof["x_mm"],
                    prof["h_mean_mm"]*1000,
                    prof["h_max_mm"]*1000,
                    alpha=0.2, color="steelblue")

    # Overlay balling frame positions
    ball_sub = sub[sub["balling"]==1]
    for _, row in ball_sub.iterrows():
        xp   = row["x_profil_mm"]
        btype = row.get("bead_type")
        color = type_colors.get(btype, "orange")
        ax.axvline(xp, color=color, alpha=0.4,
                   linewidth=1.5, ymin=0.85)

    ax.set_ylabel("Height (µm)")
    ax.set_xlabel("X position in profilometry (mm)")
    ax.set_title(
        f"L{layer}  P={LAYER_MAP[layer]['P']}W  V=2000mm/s",
        fontsize=9
    )
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = OUTPUT_DIR / "along_scan_with_labels.png"
plt.savefig(path, dpi=120, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")

print(f"\nAll outputs: {OUTPUT_DIR}/")
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name}")

print(f"""
KEY FINDINGS TO LOOK FOR:

  height_distributions.png:
    - Do balling frames have clearly higher max height than clean?
    - Is the distribution shift large (Cohen's d > 0.5) or subtle?
    - Do Large beads have higher height than Small beads?
      (This validates the size labels from microscopy)
    - Do Left/Right beads have different height profiles
      than Middle beads? (Physical mechanism validation)

  height_vs_model_score.png:
    - Positive correlation → model detects higher beads better
    - If correlation is weak → height is not the only camera signal
    - Do Right-type beads cluster differently from Left/Middle?

  along_scan_with_labels.png:
    - Do the vertical colored lines (balling labels) align with
      height peaks in the profilometry profile?
    - If yes → labels are well-aligned with actual bead locations
    - If labels appear in valleys → possible frame offset issue
""")
