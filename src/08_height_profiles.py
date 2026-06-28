"""
Along-Scan Label Spans with Auto-Aligned X Offsets
====================================================
Uses the X offsets found by lpbf_align_xcorr.py to draw
the along-scan height profile with balling label spans.

Auto-aligned X offsets (from xcorr alignment):
  L248: X_OFFSET = 25.5440mm  (delta = -0.024mm)
  L249: X_OFFSET = 25.5740mm  (delta = +0.006mm)
  L250: X_OFFSET = 25.4540mm  (delta = -0.114mm)
  L251: X_OFFSET = 25.4400mm  (delta = -0.128mm)
  L252: X_OFFSET = 25.5140mm  (delta = -0.054mm)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────
PROFIL_CSV    = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
ENRICHED_CSV  = Path(r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv")
TRACK_CSV_DIR = Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\Erfan_balling_data_updated")
OUTPUT_DIR    = Path("extended_analysis/profilometry_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pixel sizes ───────────────────────────────────────────────────
PX_X_MM = 0.00789
PX_Y_MM = 0.01250

# ── Per-layer auto-aligned X offsets from xcorr ───────────────────
LAYER_CONFIG = {
    248: {"x_offset": 25.5440, "row_center": 62,  "P": 380,
          "half_band": 20},
    249: {"x_offset": 25.5740, "row_center": 143, "P": 400,
          "half_band": 20},
    250: {"x_offset": 25.4540, "row_center": 221, "P": 420,
          "half_band": 20},
    251: {"x_offset": 25.4400, "row_center": 301, "P": 440,
          "half_band": 20},
    252: {"x_offset": 25.5140, "row_center": 379, "P": 460,
          "half_band": 20},
}
TARGET_LAYERS = sorted(LAYER_CONFIG.keys())

TYPE_COLORS = {
    "Left"  : "#00CC44",
    "Middle": "#FF8C00",
    "Right" : "#FF2222",
}


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading profilometry CSV...")
Z = pd.read_csv(PROFIL_CSV, header=None).values.astype(float)
n_rows, n_cols = Z.shape
col_mm = np.arange(n_cols) * PX_X_MM
print(f"  {n_rows}x{n_cols}")

print("Loading enriched_df...")
enriched_df = pd.read_csv(
    ENRICHED_CSV, encoding="utf-8-sig", low_memory=False
)
for col in ["frame","balling","layer","P","V"]:
    if col in enriched_df.columns:
        enriched_df[col] = pd.to_numeric(
            enriched_df[col], errors="coerce"
        ).astype("Int64")
for col in ["bead_size_mm"]:
    if col in enriched_df.columns:
        enriched_df[col] = pd.to_numeric(
            enriched_df[col], errors="coerce"
        ).astype(float)
for col in ["bead_type","bead_size_vis"]:
    if col in enriched_df.columns:
        enriched_df[col] = enriched_df[col].where(
            enriched_df[col].astype(str) != "nan", other=None
        )

print("Loading track CSVs...")
track_xy = {}
for layer in TARGET_LAYERS:
    p = TRACK_CSV_DIR / f"L0{layer}.csv"
    if p.exists():
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        track_xy[layer] = df


# ─────────────────────────────────────────────
# HEIGHT PROFILE FROM RAW Z
# ─────────────────────────────────────────────

def get_height_profile(layer):
    cfg  = LAYER_CONFIG[layer]
    rc   = cfg["row_center"]
    hb   = cfg["half_band"]
    r_lo = max(0, rc - hb)
    r_hi = min(n_rows - 1, rc + hb)
    band = Z[r_lo:r_hi+1, :]
    return col_mm, np.nanmax(band, axis=0), np.nanmean(band, axis=0)


# ─────────────────────────────────────────────
# BALLING ANNOTATIONS
# ─────────────────────────────────────────────

def get_balling_frames(layer):
    cfg = LAYER_CONFIG[layer]
    cam = enriched_df[
        enriched_df["layer"] == layer
    ].copy().reset_index(drop=True)

    if layer not in track_xy:
        return pd.DataFrame()

    xy = track_xy[layer][["frame_number","x(mm)"]].copy()
    xy.columns = ["frame","x_mm"]
    xy["frame"] = xy["frame"].astype(int)
    cam["frame"] = cam["frame"].astype(int)
    cam = cam.merge(xy, on="frame", how="left")
    cam = cam.dropna(subset=["x_mm"])
    cam["x_profil"] = cam["x_mm"] - cfg["x_offset"]
    cam = cam[cam["x_profil"] >= 0].sort_values("x_profil")
    return cam


# ─────────────────────────────────────────────
# FIGURE — ALONG-SCAN PROFILES
# ─────────────────────────────────────────────

print("\nGenerating along-scan label spans figure...")

fig, axes = plt.subplots(
    len(TARGET_LAYERS), 1,
    figsize=(22, 4 * len(TARGET_LAYERS)),
    sharex=False
)
fig.suptitle(
    "Along-scan height profile with balling label spans\n"
    "X offsets: auto-aligned per layer via cross-correlation\n"
    "Colored spans = labeled balling events  |  "
    "Green=Left  Orange=Middle  Red=Right",
    fontsize=11, fontweight="bold"
)

for ax, layer in zip(axes, TARGET_LAYERS):
    cfg  = LAYER_CONFIG[layer]
    x_mm, h_max, h_mean = get_height_profile(layer)

    # Height profile
    ax.plot(x_mm, h_max * 1000, color="steelblue",
            linewidth=1.2, alpha=0.9, label="Max height")
    ax.fill_between(x_mm, h_mean * 1000, h_max * 1000,
                    alpha=0.2, color="steelblue")
    ax.plot(x_mm, h_mean * 1000, color="steelblue",
            linewidth=0.8, alpha=0.5, linestyle="--",
            label="Mean height")

    # Balling frames
    cam  = get_balling_frames(layer)
    ball = cam[cam["balling"] == 1].copy()

    if len(ball) > 0:
        ball = ball.sort_values("x_profil").copy()
        ball["dx"]    = ball["x_profil"].diff().fillna(0)
        ball["event"] = (ball["dx"] > 0.08).cumsum()

        for _, ev in ball.groupby("event"):
            x_start  = ev["x_profil"].min() - 0.02
            x_end    = ev["x_profil"].max() + 0.02
            btype    = ev["bead_type"].mode().iloc[0] \
                       if len(ev) > 0 else None
            bsize    = ev["bead_size_vis"].mode().iloc[0] \
                       if len(ev) > 0 else "?"
            bsize_mm = ev["bead_size_mm"].mean() \
                       if "bead_size_mm" in ev.columns else np.nan
            color    = TYPE_COLORS.get(btype, "#AAAAAA")

            ax.axvspan(x_start, x_end,
                       alpha=0.30, color=color, zorder=2)

            # Label
            mid_x = (x_start + x_end) / 2
            lbl   = f"{btype[0] if btype else '?'}-{bsize}"
            if pd.notna(bsize_mm):
                lbl += f"\n{bsize_mm:.2f}"
            ax.text(mid_x, 0.97, lbl,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=4.5, color=color,
                    fontweight="bold", rotation=90)

    # Also scatter individual balling frame x positions
    # as dots on the height profile — easier to see alignment
    ball_x = ball["x_profil"].values if len(ball) > 0 else []
    if len(ball_x) > 0:
        h_at_ball = np.interp(ball_x, x_mm, h_max) * 1000
        ax.scatter(ball_x, h_at_ball, c="red", s=8,
                   zorder=5, alpha=0.6,
                   label="Balling frame position")

    n_ball  = (cam["balling"] == 1).sum()
    n_clean = (cam["balling"] == 0).sum()

    ax.set_ylabel("Height (um)")
    ax.set_xlabel("X profilometry (mm)")
    ax.set_title(
        f"L{layer}  P={cfg['P']}W  V=2000mm/s  |  "
        f"{n_ball} balling  {n_clean} clean  |  "
        f"X_OFFSET={cfg['x_offset']:.4f}mm  "
        f"row_center={cfg['row_center']} "
        f"(band +/-{cfg['half_band']}px = "
        f"+/-{cfg['half_band']*PX_Y_MM*1000:.0f}um)",
        fontsize=9, fontweight="bold"
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = OUTPUT_DIR / "along_scan_label_spans_aligned.png"
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

print("""
WHAT TO CHECK:
  - Do the colored spans land on height peaks (not valleys)?
  - Do the red dots (individual balling frame positions) sit
    near the tops of the height peaks?
  - Are there height peaks WITHOUT any colored span?
    (would suggest unlabeled balling events)
  - Are there colored spans in clear valleys?
    (would suggest label-profilometry misalignment)
""")
