"""
Auto-Align Balling Labels to Profilometry Height Peaks
=======================================================
Computes height profile directly from raw Z matrix using
your current LAYER_MAP row_center values and a configurable
band width. No dependency on pre-computed along_scan CSVs.

For each layer:
  1. Extract a band of rows around row_center from Z
  2. Compute max/mean height per column across that band
  3. Build a balling indicator signal from camera labels
  4. Search for X offset that maximizes height at balling
     frames vs height at clean frames (separation score)
  5. Report optimal per-layer and global X offset

Outputs:
  - xcorr_alignment.png  — score curves + before/after plots
  - alignment_constants.txt  — paste into profilometry_overlay.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
PROFIL_CSV    = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
ENRICHED_CSV  = Path(r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv")
TRACK_CSV_DIR = Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\Erfan_balling_data_updated")
OUTPUT_DIR    = Path("extended_analysis/profilometry_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Current alignment constants (your manually tuned values) ──────
X_OFFSET_CURRENT = 25.5680   # mm
Y_OFFSET         =  5.2320   # mm
PX_X_MM          =  0.00789  # mm per pixel X
PX_Y_MM          =  0.01250  # mm per pixel Y

LAYER_X_FINETUNE_CURRENT = {
    248: -0.05,
    249:  0.00,
    250:  0.05,
    251:  0.00,
    252: -0.05,
}

# ── Layer mapping — YOUR current manually tuned row centers ───────
LAYER_MAP = {
    248: {"row_center": 62,  "row_lo": 1,   "row_hi": 81,  "P": 380},
    249: {"row_center": 143, "row_lo": 121, "row_hi": 161, "P": 400},
    250: {"row_center": 221, "row_lo": 201, "row_hi": 241, "P": 420},
    251: {"row_center": 301, "row_lo": 281, "row_hi": 321, "P": 440},
    252: {"row_center": 379, "row_lo": 361, "row_hi": 401, "P": 460},
}
TARGET_LAYERS = sorted(LAYER_MAP.keys())

# ── Band width options to try ─────────────────────────────────────
# How many pixels on each side of row_center to include in band
# 0  = single center row only
# 10 = 21 rows = 0.25mm band
# 20 = 41 rows = 0.50mm band
# 40 = 81 rows = full track row range
HALF_BAND_PX = 20   # start with 0.25mm half-band = 0.50mm total

# ── Search parameters ─────────────────────────────────────────────
SEARCH_RANGE_MM = 0.5    # search +/- 0.5mm around current offset
SEARCH_STEP_MM  = 0.002  # 2µm step — finer than one pixel (7.89µm)


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading profilometry CSV...")
Z = pd.read_csv(PROFIL_CSV, header=None).values.astype(float)
n_rows, n_cols = Z.shape
col_mm = np.arange(n_cols) * PX_X_MM
print(f"  Shape: {n_rows}x{n_cols}  "
      f"({n_rows*PX_Y_MM:.2f}mm x {n_cols*PX_X_MM:.2f}mm)")

print("Loading enriched_df...")
enriched_df = pd.read_csv(
    ENRICHED_CSV, encoding="utf-8-sig", low_memory=False
)
for col in ["frame","balling","layer","P","V"]:
    if col in enriched_df.columns:
        enriched_df[col] = pd.to_numeric(
            enriched_df[col], errors="coerce"
        ).astype("Int64")

print("Loading track CSVs...")
track_xy = {}
for layer in TARGET_LAYERS:
    p = TRACK_CSV_DIR / f"L0{layer}.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        track_xy[layer] = df
        print(f"  L{layer}: {len(df)} frames")


# ─────────────────────────────────────────────
# EXTRACT HEIGHT PROFILE FROM RAW Z
# ─────────────────────────────────────────────

def get_height_profile(layer, half_band=HALF_BAND_PX):
    """
    Extract column-wise max height from a band of rows
    centered on row_center for this layer.

    Returns (x_mm_array, h_max_array, h_mean_array)
    """
    rc = LAYER_MAP[layer]["row_center"]
    r_lo = max(0, rc - half_band)
    r_hi = min(n_rows - 1, rc + half_band)

    band = Z[r_lo:r_hi+1, :]   # shape: (band_height, n_cols)

    h_max  = np.nanmax(band,  axis=0)
    h_mean = np.nanmean(band, axis=0)

    return col_mm, h_max, h_mean


# ─────────────────────────────────────────────
# BUILD BALLING SIGNAL
# ─────────────────────────────────────────────

def build_balling_signal(layer, x_offset, finetune=0.0):
    """
    For each camera frame in this layer, compute:
      x_profil = x_machine - x_offset - finetune
      label    = balling (1) or clean (0)

    Returns (x_profil_array, balling_array) sorted by x.
    Only laser-on frames (x_profil >= 0) included.
    """
    cam = enriched_df[
        enriched_df["layer"] == layer
    ].copy().reset_index(drop=True)

    if layer not in track_xy:
        return None, None

    xy = track_xy[layer][["frame_number","x(mm)"]].copy()
    xy.columns = ["frame","x_mm"]
    xy["frame"] = xy["frame"].astype(int)
    cam["frame"] = cam["frame"].astype(int)
    cam = cam.merge(xy, on="frame", how="left")
    cam = cam.dropna(subset=["x_mm"])

    cam["x_profil"] = cam["x_mm"] - x_offset - finetune
    cam = cam[cam["x_profil"] >= 0].sort_values("x_profil")

    if len(cam) == 0:
        return None, None

    return cam["x_profil"].values, cam["balling"].astype(float).values


# ─────────────────────────────────────────────
# ALIGNMENT SCORE FUNCTION
# ─────────────────────────────────────────────

def alignment_score(layer, x_offset, finetune, h_interp):
    """
    Score = mean height at balling frames - mean height at clean frames.
    Higher = balling labels better aligned with height peaks.
    """
    x_arr, bal_arr = build_balling_signal(layer, x_offset, finetune)
    if x_arr is None or len(x_arr) < 5:
        return 0.0

    h_at_frames = h_interp(x_arr)
    ball_mask   = bal_arr > 0.5
    clean_mask  = ~ball_mask

    if ball_mask.sum() < 2 or clean_mask.sum() < 2:
        return 0.0

    return float(h_at_frames[ball_mask].mean() -
                 h_at_frames[clean_mask].mean())


# ─────────────────────────────────────────────
# RUN ALIGNMENT SEARCH
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"ALIGNMENT SEARCH  (band half-width = {HALF_BAND_PX}px "
      f"= {HALF_BAND_PX*PX_Y_MM*1000:.0f}um per side)")
print(f"{'='*60}")

offsets = np.arange(
    -SEARCH_RANGE_MM,
     SEARCH_RANGE_MM + SEARCH_STEP_MM,
     SEARCH_STEP_MM
)

fig, axes = plt.subplots(
    len(TARGET_LAYERS), 2,
    figsize=(18, 4.5 * len(TARGET_LAYERS))
)
fig.suptitle(
    f"Auto-alignment: balling labels vs profilometry height\n"
    f"Band: +/-{HALF_BAND_PX}px = +/-{HALF_BAND_PX*PX_Y_MM*1000:.0f}um "
    f"around row_center  |  "
    f"Score = mean_height(balling) - mean_height(clean)",
    fontsize=10, fontweight="bold"
)

results = {}

for i, layer in enumerate(TARGET_LAYERS):
    finetune  = LAYER_X_FINETUNE_CURRENT.get(layer, 0.0)
    x_mm, h_max, h_mean = get_height_profile(layer)

    # Normalize for scoring
    h_range = h_max.max() - h_max.min()
    if h_range < 1e-8:
        print(f"  L{layer}: flat height profile — skipping")
        continue
    h_norm   = (h_max - h_max.min()) / h_range
    h_interp = interp1d(x_mm, h_norm,
                        bounds_error=False, fill_value=0.0)

    # Score at each candidate offset
    scores = []
    for delta in offsets:
        s = alignment_score(
            layer, X_OFFSET_CURRENT + delta, finetune, h_interp
        )
        scores.append(s)
    scores = np.array(scores)

    best_idx   = int(np.argmax(scores))
    best_delta = float(offsets[best_idx])
    best_score = float(scores[best_idx])
    new_xoff   = X_OFFSET_CURRENT + best_delta

    results[layer] = {
        "best_delta_mm": best_delta,
        "new_x_offset" : new_xoff,
        "best_score"   : best_score,
        "scores"       : scores,
    }

    print(f"\n  L{layer} P={LAYER_MAP[layer]['P']}W "
          f"rc={LAYER_MAP[layer]['row_center']}:")
    print(f"    Band rows: {LAYER_MAP[layer]['row_center']-HALF_BAND_PX}"
          f" to {LAYER_MAP[layer]['row_center']+HALF_BAND_PX}")
    print(f"    Best delta: {best_delta:+.3f}mm  "
          f"(new X_OFFSET = {new_xoff:.4f}mm)")
    print(f"    Score: {best_score:.5f}")

    # ── Left panel: score curve ───────────────────────────────
    ax = axes[i, 0]
    ax.plot(offsets, scores, color="steelblue", linewidth=1.5)
    ax.axvline(best_delta, color="red", linewidth=2,
               linestyle="--",
               label=f"Best: {best_delta:+.3f}mm")
    ax.axvline(0, color="gray", linewidth=1.2,
               linestyle=":", label="Current (delta=0)")
    ax.fill_between(offsets, scores,
                    where=scores >= scores.max() * 0.95,
                    alpha=0.2, color="red",
                    label="Top 5% region")
    ax.set_xlabel("X offset correction (mm)")
    ax.set_ylabel("Alignment score\n(ball_height - clean_height)")
    ax.set_title(f"L{layer} — score vs X correction", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right panel: height + labels before/after ─────────────
    ax = axes[i, 1]
    ax.plot(x_mm, h_max * 1000, color="steelblue",
            linewidth=1, alpha=0.7, label="Max height")
    ax.fill_between(x_mm, h_mean * 1000, h_max * 1000,
                    alpha=0.15, color="steelblue")

    # Before: current offset
    x_b, bal_b = build_balling_signal(
        layer, X_OFFSET_CURRENT, finetune
    )
    if x_b is not None:
        mask_b = bal_b > 0.5
        h_b    = np.interp(x_b[mask_b], x_mm, h_max) * 1000
        ax.scatter(x_b[mask_b], h_b, c="orange", s=25,
                   alpha=0.6, zorder=4,
                   label=f"Before (delta=0)")

    # After: best offset
    x_a, bal_a = build_balling_signal(
        layer, new_xoff, finetune
    )
    if x_a is not None:
        mask_a = bal_a > 0.5
        h_a    = np.interp(x_a[mask_a], x_mm, h_max) * 1000
        ax.scatter(x_a[mask_a], h_a, c="red", s=25,
                   alpha=0.8, zorder=5,
                   label=f"After ({best_delta:+.3f}mm)")

    ax.set_xlabel("X profilometry (mm)")
    ax.set_ylabel("Max height (um)")
    ax.set_title(
        f"L{layer} — balling frames on height profile\n"
        f"Band: row_center={LAYER_MAP[layer]['row_center']} "
        f"+/-{HALF_BAND_PX}px = "
        f"+/-{HALF_BAND_PX*PX_Y_MM*1000:.0f}um",
        fontsize=9
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = OUTPUT_DIR / "xcorr_alignment.png"
plt.savefig(path, dpi=120, bbox_inches="tight")
plt.close()
print(f"\nSaved: {path.name}")


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")

deltas       = [r["best_delta_mm"] for r in results.values()]
global_delta = float(np.mean(deltas))
new_global   = X_OFFSET_CURRENT + global_delta

print(f"\n  Per-layer best X corrections:")
for layer, r in results.items():
    print(f"    L{layer}: {r['best_delta_mm']:+.4f}mm  "
          f"score={r['best_score']:.5f}")

print(f"\n  Global mean correction: {global_delta:+.4f}mm")
print(f"  New X_OFFSET: {new_global:.4f}mm")

new_finetuning = {}
for layer, r in results.items():
    old_ft   = LAYER_X_FINETUNE_CURRENT.get(layer, 0.0)
    deviation = r["best_delta_mm"] - global_delta
    new_ft   = old_ft - deviation
    new_finetuning[layer] = round(new_ft, 4)

print(f"\n  Per-layer fine-tunes (deviation from global mean):")
for layer, ft in new_finetuning.items():
    print(f"    {layer}: {ft:+.4f}mm")

print(f"""
PASTE INTO profilometry_overlay.py:

X_OFFSET  = {new_global:.4f}   # mm (auto-aligned, band={HALF_BAND_PX}px)

LAYER_X_FINETUNE = {{""")
for layer, ft in new_finetuning.items():
    print(f"    {layer}:  {ft:+.4f},")
print("}")

# Save constants
with open(OUTPUT_DIR / "alignment_constants.txt",
          "w", encoding="utf-8") as f:
    f.write(f"AUTO-ALIGNED CONSTANTS\n")
    f.write(f"Band half-width: {HALF_BAND_PX}px = "
            f"{HALF_BAND_PX*PX_Y_MM*1000:.0f}um per side\n")
    f.write("="*40 + "\n\n")
    f.write(f"X_OFFSET  = {new_global:.4f}   # mm\n\n")
    f.write("LAYER_X_FINETUNE = {\n")
    for layer, ft in new_finetuning.items():
        f.write(f"    {layer}:  {ft:+.4f},\n")
    f.write("}\n\n")
    f.write("Per-layer details:\n")
    for layer, r in results.items():
        f.write(f"  L{layer}: delta={r['best_delta_mm']:+.4f}mm  "
                f"score={r['best_score']:.5f}\n")

print(f"\nSaved: {OUTPUT_DIR}/alignment_constants.txt")
print(f"Diagnostic figure: {OUTPUT_DIR}/xcorr_alignment.png")
print(f"""
NEXT STEPS:
  1. Look at xcorr_alignment.png
     Left panels: score curve should have a clear single peak
     Right panels: red dots (after) should sit higher on height
     peaks than orange dots (before)
  2. If score curves are noisy or flat -> the band may be wrong
     Try HALF_BAND_PX = 0 (single row), 10, 20, 40
  3. Paste the constants into profilometry_overlay.py
  4. Regenerate the overlay and visually verify
""")
