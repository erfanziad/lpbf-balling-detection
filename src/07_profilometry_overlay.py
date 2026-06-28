"""
Microscopy Label Overlay on Profilometry Height Map
====================================================
For each layer (L248-L252), generates a zoomed height map
showing the track region with balling events overlaid as
colored markers.

Color coding:
  Bead type:  Green=Left  Orange=Middle  Red=Right
  Bead size:  marker size proportional to bead_size_mm
  Clean frames: no marker

Also generates a second figure showing the along-scan
height profile with colored spans for each balling event.

Alignment constants (solved and verified):
  X_OFFSET = 25.3680 mm (25.6180 - 0.25mm global X correction)
  Y_OFFSET =  5.2320 mm
  px_x     =  7.890 um
  px_y     = 12.500 um
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── Paths ─────────────────────────────────────────────────────────
PROFIL_CSV   = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
ENRICHED_CSV = Path(r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv")
TRACK_CSV_DIR = Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\Erfan_balling_data_updated")
PROFIL_DIR   = Path("lpbf_outputs")
OUTPUT_DIR   = Path("extended_analysis/profilometry_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Alignment ─────────────────────────────────────────────────────
X_OFFSET  = 25.3680   # mm  (25.6180 - 0.25mm global rightward shift)
Y_OFFSET  =  5.2320   # mm
PX_X_MM   =  0.00789  # mm per pixel X
PX_Y_MM   =  0.01250  # mm per pixel Y

# Per-layer X fine-tune (mm) — add on top of global X_OFFSET
# L252 had the largest residual offset (~0.10mm extra)
# Per-layer fine-tune ON TOP of global X_OFFSET correction
# Positive = shift further right, Negative = shift back left
LAYER_X_FINETUNE = {
    248: -0.05,   # needs 0.30mm total, global gives 0.25mm -> extra 0.05mm
    249:  0.00,   # needs 0.25mm -> global is exact
    250:  0.05,   # needs 0.20mm -> pull back 0.05mm from global
    251:  0.00,   # needs 0.25mm -> global is exact
    252: -0.05,   # needs 0.30mm total -> extra 0.05mm
}

# ── Layer mapping ─────────────────────────────────────────────────
# Row centers adjusted based on visual alignment assessment:
# Original centers were slightly above the actual track ridge peaks
# +2 pixels = +0.025mm downward shift to center on ridge
LAYER_MAP = {
    248: {"row_center": 55,  "row_lo": 1,   "row_hi": 81,  "P": 380},  # was 41->43, now 55 (+12px = +0.15mm down)
    249: {"row_center": 143, "row_lo": 121, "row_hi": 161, "P": 400},  # was 141, shifted down +2px
    250: {"row_center": 221, "row_lo": 201, "row_hi": 241, "P": 420},  # kept — best aligned
    251: {"row_center": 301, "row_lo": 281, "row_hi": 321, "P": 440},  # kept — good alignment
    252: {"row_center": 379, "row_lo": 361, "row_hi": 401, "P": 460},  # was 381, shifted up -2px (Red was too low)
}

TARGET_LAYERS = sorted(LAYER_MAP.keys())

# ── Visual settings ───────────────────────────────────────────────
TYPE_COLORS = {
    "Left"  : "#00CC44",   # green
    "Middle": "#FF8C00",   # orange
    "Right" : "#FF2222",   # red
    None    : "#FFFFFF",
}
MIN_MARKER = 60    # minimum marker size
MAX_MARKER = 300   # maximum marker size


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading profilometry CSV...")
Z = pd.read_csv(PROFIL_CSV, header=None).values.astype(float)
n_rows, n_cols = Z.shape
col_mm = np.arange(n_cols) * PX_X_MM   # X in profilometry mm
row_mm = np.arange(n_rows) * PX_Y_MM   # Y in profilometry mm
print(f"  Shape: {n_rows}×{n_cols}  "
      f"({row_mm[-1]:.2f}mm × {col_mm[-1]:.2f}mm)")

Z_display = np.where(np.isnan(Z), np.nanmean(Z), Z)
vmin = np.nanpercentile(Z, 2)
vmax = np.nanpercentile(Z, 98)

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
for col in ["bead_type","bead_size_vis","image_path"]:
    if col in enriched_df.columns:
        enriched_df[col] = enriched_df[col].where(
            enriched_df[col].astype(str) != "nan", other=None
        )

print("Loading track x-coordinates...")
track_xy = {}
for layer in TARGET_LAYERS:
    csv_path = TRACK_CSV_DIR / f"L0{layer}.csv"
    if not csv_path.exists():
        continue
    df_t = pd.read_csv(csv_path)
    df_t.columns = [c.strip() for c in df_t.columns]
    track_xy[layer] = df_t

print("Loading along-scan profilometry...")
profil_data = {}
for layer in TARGET_LAYERS:
    csv_path = PROFIL_DIR / f"L0{layer}_along_scan.csv"
    if csv_path.exists():
        profil_data[layer] = pd.read_csv(csv_path)


# ─────────────────────────────────────────────
# BUILD PER-FRAME ANNOTATION TABLE
# ─────────────────────────────────────────────

def build_annotations(layer):
    """
    For each frame in this layer, compute:
      x_profil_mm: profilometry X coordinate
      y_profil_mm: profilometry Y coordinate (fixed per layer)
      balling, bead_type, bead_size_vis, bead_size_mm
    """
    cam = enriched_df[
        enriched_df["layer"] == layer
    ].copy().reset_index(drop=True)

    # Add x from track CSV
    if layer in track_xy:
        xy = track_xy[layer][["frame_number","x(mm)"]].copy()
        xy.columns = ["frame","x_mm"]
        xy["frame"] = xy["frame"].astype(int)
        cam["frame"] = cam["frame"].astype(int)
        cam = cam.merge(xy, on="frame", how="left")
    else:
        cam["x_mm"] = np.nan

    # Convert to profilometry coordinates
    # Apply global offset + per-layer fine-tune
    finetune = LAYER_X_FINETUNE.get(layer, 0.0)
    cam["x_profil_mm"] = cam["x_mm"] - X_OFFSET - finetune
    y_machine = LAYER_MAP[layer]["row_center"] * PX_Y_MM + Y_OFFSET
    cam["y_profil_mm"] = LAYER_MAP[layer]["row_center"] * PX_Y_MM

    # Marker size proportional to bead_size_mm
    size_mm = cam["bead_size_mm"].fillna(0.05)
    size_mm = size_mm.clip(lower=0.01)
    # Normalize to marker size range
    s_min, s_max = size_mm[cam["balling"]==1].min() \
        if (cam["balling"]==1).any() else 0.01, \
        size_mm[cam["balling"]==1].max() \
        if (cam["balling"]==1).any() else 0.3
    if s_max > s_min:
        cam["marker_size"] = MIN_MARKER + (MAX_MARKER - MIN_MARKER) * \
            (size_mm - s_min) / (s_max - s_min)
    else:
        cam["marker_size"] = (MIN_MARKER + MAX_MARKER) / 2

    return cam


# ─────────────────────────────────────────────
# FIGURE 1 — FULL HEIGHT MAP WITH ALL LABELS
# ─────────────────────────────────────────────

print("\nGenerating full overlay figure...")

fig, ax = plt.subplots(figsize=(20, 8))
im = ax.imshow(
    Z_display,
    aspect="auto",
    origin="upper",
    extent=[0, col_mm[-1], row_mm[-1], 0],
    cmap="RdYlBu_r",
    vmin=vmin, vmax=vmax,
    alpha=0.9
)
plt.colorbar(im, ax=ax, label="Height (mm)", shrink=0.6)

# Draw track band rectangles
for layer, info in LAYER_MAP.items():
    y_lo = info["row_lo"] * PX_Y_MM
    y_hi = info["row_hi"] * PX_Y_MM
    ax.axhspan(y_lo, y_hi, alpha=0.08, color="white")
    ax.text(0.02, (y_lo + y_hi)/2,
            f"L{layer}\nP={info['P']}W",
            va="center", ha="left", fontsize=7,
            color="white", fontweight="bold",
            bbox=dict(fc="black", alpha=0.5,
                      boxstyle="round,pad=0.2"))

# Overlay balling markers
legend_handles = []
for layer in TARGET_LAYERS:
    ann = build_annotations(layer)
    ball = ann[ann["balling"] == 1].dropna(subset=["x_profil_mm"])

    for btype, color in TYPE_COLORS.items():
        if btype is None:
            continue
        sub = ball[ball["bead_type"] == btype]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x_profil_mm"],
            sub["y_profil_mm"],
            c=color,
            s=sub["marker_size"],
            alpha=0.75,
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

# Legend for bead type
for btype, color in TYPE_COLORS.items():
    if btype is None:
        continue
    legend_handles.append(
        mpatches.Patch(color=color, label=f"{btype} bead")
    )
# Legend for marker size
for size_label, size_val in [("Large (0.15mm)", MAX_MARKER),
                               ("Medium (0.10mm)", (MIN_MARKER+MAX_MARKER)//2),
                               ("Small (0.05mm)", MIN_MARKER)]:
    legend_handles.append(
        plt.scatter([], [], c="gray", s=size_val,
                    label=size_label, alpha=0.7)
    )

ax.legend(handles=legend_handles, loc="upper right",
          fontsize=8, facecolor="black",
          labelcolor="white", framealpha=0.8)
ax.set_xlabel("X position in profilometry (mm) — along scan direction")
ax.set_ylabel("Y position in profilometry (mm) — cross-track direction")
ax.set_title(
    "Surface height map with microscopy balling labels overlaid\n"
    "X-offset corrected: +0.175mm global + per-layer fine-tune\n"
    "Green=Left  Orange=Middle  Red=Right  |  Marker size = bead diameter",
    fontsize=10, fontweight="bold"
)

plt.tight_layout()
path = OUTPUT_DIR / "full_overlay.png"
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# FIGURE 2 — PER-LAYER ZOOMED HEIGHT MAP
# ─────────────────────────────────────────────

print("Generating per-layer zoomed overlays...")

fig, axes = plt.subplots(
    len(TARGET_LAYERS), 1,
    figsize=(20, 4 * len(TARGET_LAYERS))
)
fig.suptitle(
    "Per-layer zoomed height map with balling labels\n"
    "Green=Left  Orange=Middle  Red=Right  |  "
    "Marker size = bead diameter (mm)",
    fontsize=11, fontweight="bold"
)

for ax, layer in zip(axes, TARGET_LAYERS):
    info = LAYER_MAP[layer]
    row_lo = info["row_lo"]
    row_hi = info["row_hi"]

    # Extract track band from height map
    Z_band = Z_display[row_lo:row_hi+1, :]
    y_lo_mm = row_lo * PX_Y_MM
    y_hi_mm = row_hi * PX_Y_MM

    im = ax.imshow(
        Z_band,
        aspect="auto",
        origin="upper",
        extent=[0, col_mm[-1], y_hi_mm, y_lo_mm],
        cmap="RdYlBu_r",
        vmin=vmin, vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Height (mm)",
                 shrink=0.8, pad=0.01)

    # Track centerline
    y_center_mm = info["row_center"] * PX_Y_MM
    ax.axhline(y_center_mm, color="white", linewidth=1,
               linestyle="--", alpha=0.5)

    # Balling markers
    ann = build_annotations(layer)
    ball = ann[ann["balling"] == 1].dropna(subset=["x_profil_mm"])

    for btype, color in TYPE_COLORS.items():
        if btype is None:
            continue
        sub = ball[ball["bead_type"] == btype]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x_profil_mm"],
            [y_center_mm] * len(sub),
            c=color,
            s=sub["marker_size"],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=f"{btype} (n={len(sub)})"
        )

        # Annotate size in mm for each marker
        for _, row in sub.iterrows():
            if pd.notna(row.get("bead_size_mm")):
                ax.annotate(
                    f"{row['bead_size_mm']:.2f}",
                    (row["x_profil_mm"], y_center_mm),
                    textcoords="offset points",
                    xytext=(0, 12),
                    fontsize=4.5,
                    ha="center",
                    color=color,
                    fontweight="bold"
                )

    n_ball = (ann["balling"] == 1).sum()
    n_clean = (ann["balling"] == 0).sum()
    ax.set_title(
        f"L{layer}  P={info['P']}W  V=2000mm/s  |  "
        f"{n_ball} balling frames  {n_clean} clean frames",
        fontsize=9, fontweight="bold"
    )
    ax.set_xlabel("X position (mm) — along scan")
    ax.set_ylabel("Y (mm)")
    ax.legend(fontsize=7, loc="upper right",
              facecolor="black", labelcolor="white",
              framealpha=0.7)

plt.tight_layout()
path = OUTPUT_DIR / "per_layer_zoomed_overlay.png"
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────
# FIGURE 3 — ALONG-SCAN PROFILE + LABELS
# ─────────────────────────────────────────────

print("Generating along-scan profile with label spans...")

fig, axes = plt.subplots(
    len(TARGET_LAYERS), 1,
    figsize=(20, 3.5 * len(TARGET_LAYERS)),
    sharex=False
)
fig.suptitle(
    "Along-scan height profile with balling event spans\n"
    "Colored spans = labeled balling events  |  "
    "Height = max in 0.5mm cross-track band",
    fontsize=11, fontweight="bold"
)

for ax, layer in zip(axes, TARGET_LAYERS):
    info = LAYER_MAP[layer]

    # Height profile
    if layer in profil_data:
        prof = profil_data[layer]
        ax.plot(prof["x_mm"], prof["h_max_mm"] * 1000,
                color="steelblue", linewidth=1.2,
                alpha=0.9, label="Max height")
        ax.fill_between(
            prof["x_mm"],
            prof["h_mean_mm"] * 1000,
            prof["h_max_mm"] * 1000,
            alpha=0.2, color="steelblue"
        )
        ax.plot(prof["x_mm"], prof["h_mean_mm"] * 1000,
                color="steelblue", linewidth=0.8,
                alpha=0.5, linestyle="--",
                label="Mean height")

    # Balling event spans
    ann = build_annotations(layer)
    ball = ann[ann["balling"] == 1].dropna(
        subset=["x_profil_mm"]
    ).sort_values("x_profil_mm")

    # Find contiguous events
    if len(ball) > 0:
        ball["dx"] = ball["x_profil_mm"].diff().fillna(0)
        ball["event"] = (ball["dx"] > 0.1).cumsum()

        for event_id, ev_frames in ball.groupby("event"):
            x_start = ev_frames["x_profil_mm"].min() - 0.02
            x_end   = ev_frames["x_profil_mm"].max() + 0.02
            btype   = ev_frames["bead_type"].mode().iloc[0] \
                if len(ev_frames) > 0 else None
            bsize   = ev_frames["bead_size_vis"].mode().iloc[0] \
                if len(ev_frames) > 0 else "?"
            bsize_mm = ev_frames["bead_size_mm"].mean()
            color   = TYPE_COLORS.get(btype, "#AAAAAA")

            ax.axvspan(x_start, x_end,
                       alpha=0.3, color=color, zorder=2)
            # Label at top of span
            y_top = ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 \
                    else 30
            mid_x = (x_start + x_end) / 2
            label_str = f"{btype[0] if btype else '?'}-{bsize}"
            if pd.notna(bsize_mm):
                label_str += f"\n{bsize_mm:.2f}mm"
            ax.text(
                mid_x, 0.97,
                label_str,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top",
                fontsize=4.5, color=color,
                fontweight="bold",
                rotation=90
            )

    ax.set_ylabel("Height (um)")
    ax.set_xlabel("X profilometry (mm)")
    ax.set_title(
        f"L{layer}  P={info['P']}W  |  "
        f"Spans: Green=Left  Orange=Middle  Red=Right",
        fontsize=9
    )
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
path = OUTPUT_DIR / "along_scan_label_spans.png"
plt.savefig(path, dpi=130, bbox_inches="tight")
plt.close()
print(f"  Saved: {path.name}")


print(f"\nAll outputs: {OUTPUT_DIR}/")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

print("""
WHAT TO CHECK IN EACH FIGURE:

full_overlay.png
  The big picture — all 5 tracks with all labels.
  Do the markers cluster where you see height anomalies
  (bright/dark patches) in the height map?
  Right-type (red) markers should be on the right side
  of each bead event. Left-type (green) on the left side.

per_layer_zoomed_overlay.png
  Zoomed into each track band. Markers sit on the track
  centerline (dashed white). Size in mm annotated above.
  Key check: do markers land on height peaks or valleys?
  If markers are consistently in valleys -> frame offset.

along_scan_label_spans.png
  Height profile with colored spans showing balling events.
  Each span = one contiguous balling event.
  Label shows type abbreviation and size in mm.
  Key check: do height peaks align with span positions?
  If yes -> profilometry and camera labels are well-aligned.
""")
