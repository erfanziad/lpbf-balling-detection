"""
Load Rich Balling Labels from Excel and Merge into final_df
============================================================
The Excel file has one sheet per layer with columns:
  Balling, Type, Size (visual), Size (mm), Start (Frame), End (Frame)

This script:
  1. Reads all sheets from the Excel file
  2. Parses bead type (Left/Middle/Right), size (L/M/S), size_mm, frame ranges
  3. Merges into final_df adding columns:
       bead_type      : Left / Middle / Right / None
       bead_size_vis  : L / M / S / None
       bead_size_mm   : float / NaN
       bead_id        : int (which bead event this frame belongs to) / -1
       n_beads        : number of overlapping bead events for this frame
                        (>1 means multi-bead frame per Zhuo's guidance)
  4. Prints a summary of the enriched dataset
  5. Answers Zhuo's research question:
       Which bead types/sizes are detectable from melt pool images?

Key insight from Zhuo:
  - Labels are from post-build microscopy, NOT melt pool camera
  - Early buffer frames before melt pool visible are INTENTIONAL
  - Multi-bead = any frame overlapping with more than one bead event
  - "Middle" beads may be hardest to detect (pool looks normal)
  - "Left/Right" beads may be easier (lateral asymmetry possible)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

excel_path = Path(
    r"C:\Users\erfan\Downloads\single_track_balling_data-20260326T220819Z-1-001"
    r"\single_track_balling_data\Balling Labeling Result.xlsx"
)

# Map sheet names to layer numbers
# Adjust if sheet names differ from "Layer 245" etc.
LAYER_NUMBERS = [245, 246, 247, 248, 249, 250, 251, 252]


# ─────────────────────────────────────────────
# STEP 1: Read all sheets
# ─────────────────────────────────────────────

def load_all_sheets(excel_path):
    """
    Load all sheets from the Excel file.
    Returns dict: {layer_num: dataframe}
    Try both "Layer 245" and "245" as sheet name formats.
    """
    print(f"Loading Excel file: {excel_path.name}")
    all_sheets = pd.read_excel(excel_path, sheet_name=None, header=None)

    print(f"  Found {len(all_sheets)} sheets: {list(all_sheets.keys())}")

    layer_dfs = {}
    for sheet_name, df_raw in all_sheets.items():
        # Try to identify layer number from sheet name
        layer_num = None
        for ln in LAYER_NUMBERS:
            if str(ln) in str(sheet_name):
                layer_num = ln
                break

        if layer_num is None:
            print(f"  Could not identify layer for sheet '{sheet_name}' — skipping")
            continue

        layer_dfs[layer_num] = (sheet_name, df_raw)
        print(f"  Sheet '{sheet_name}' → Layer {layer_num}  "
              f"({len(df_raw)} raw rows)")

    return layer_dfs


# ─────────────────────────────────────────────
# STEP 2: Parse each sheet into clean bead events
# ─────────────────────────────────────────────

def parse_sheet(layer_num, df_raw):
    """
    Parse raw sheet into a clean dataframe of bead events.

    Expected columns (possibly with header row):
      Balling | Type | Size (visual) | Size (mm) | Start (Frame) | End (Frame)

    Returns DataFrame with columns:
      bead_id, bead_type, bead_size_vis, bead_size_mm, start_frame, end_frame
    """
    # Find the header row (look for "Start" or "Frame" in the data)
    header_row = None
    for i, row in df_raw.iterrows():
        row_str = " ".join(str(v) for v in row.values).lower()
        if "start" in row_str or "frame" in row_str or "type" in row_str:
            header_row = i
            break

    if header_row is not None:
        df = df_raw.iloc[header_row + 1:].reset_index(drop=True)
        df.columns = df_raw.iloc[header_row].tolist()
    else:
        df = df_raw.copy()

    # Normalize column names
    col_map = {}
    for col in df.columns:
        col_str = str(col).lower().strip()
        if "balling" in col_str or col_str in ["#", "no", "id"]:
            col_map[col] = "bead_id"
        elif col_str == "type":
            col_map[col] = "bead_type"
        elif "visual" in col_str or col_str == "size":
            col_map[col] = "bead_size_vis"
        elif "mm" in col_str:
            col_map[col] = "bead_size_mm"
        elif "start" in col_str:
            col_map[col] = "start_frame"
        elif "end" in col_str:
            col_map[col] = "end_frame"

    df = df.rename(columns=col_map)

    # Keep only relevant columns
    keep = [c for c in ["bead_id", "bead_type", "bead_size_vis",
                         "bead_size_mm", "start_frame", "end_frame"]
            if c in df.columns]
    df = df[keep].copy()

    # Drop rows with no start/end frame
    if "start_frame" not in df.columns or "end_frame" not in df.columns:
        print(f"  Layer {layer_num}: could not find start/end frame columns")
        print(f"  Available columns: {list(df.columns)}")
        return None

    df = df.dropna(subset=["start_frame", "end_frame"])
    df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce")
    df["end_frame"]   = pd.to_numeric(df["end_frame"],   errors="coerce")
    df = df.dropna(subset=["start_frame", "end_frame"])
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"]   = df["end_frame"].astype(int)

    if "bead_size_mm" in df.columns:
        df["bead_size_mm"] = pd.to_numeric(df["bead_size_mm"], errors="coerce")

    if "bead_id" not in df.columns:
        df.insert(0, "bead_id", range(1, len(df) + 1))

    df["layer"] = layer_num
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# STEP 3: Merge into final_df
# ─────────────────────────────────────────────

def merge_bead_metadata(final_df, all_bead_events):
    """
    For each frame in final_df, find which bead event(s) it falls into
    and add metadata columns.

    Columns added:
      bead_type     : type of bead (Left/Middle/Right) — first matching event
      bead_size_vis : visual size (L/M/S) — first matching event
      bead_size_mm  : bead size in mm — first matching event
      bead_id       : id of first matching bead event (-1 if none)
      n_beads       : number of bead events overlapping this frame
                      (per Zhuo: n_beads > 1 = multi-bead frame)
    """
    df = final_df.copy()

    # Initialize new columns
    df["bead_type"]     = None
    df["bead_size_vis"] = None
    df["bead_size_mm"]  = np.nan
    df["bead_id_rich"]  = -1
    df["n_beads"]       = 0

    for layer_num, bead_df in all_bead_events.items():
        if bead_df is None or bead_df.empty:
            continue

        layer_mask = df["layer"] == layer_num
        layer_rows = df[layer_mask].copy()

        for idx, row in layer_rows.iterrows():
            fn = int(row["frame"])

            # Find all bead events that include this frame
            matching = bead_df[
                (bead_df["start_frame"] <= fn) &
                (bead_df["end_frame"]   >= fn)
            ]

            n = len(matching)
            df.at[idx, "n_beads"] = n

            if n > 0:
                first = matching.iloc[0]
                df.at[idx, "bead_id_rich"]  = int(first["bead_id"]) \
                    if "bead_id" in first.index else -1
                df.at[idx, "bead_type"]     = str(first["bead_type"]) \
                    if "bead_type" in first.index else None
                df.at[idx, "bead_size_vis"] = str(first["bead_size_vis"]) \
                    if "bead_size_vis" in first.index else None
                df.at[idx, "bead_size_mm"]  = float(first["bead_size_mm"]) \
                    if "bead_size_mm" in first.index and \
                       pd.notna(first.get("bead_size_mm")) else np.nan

    # Multi-bead flag
    df["is_multi_bead"] = df["n_beads"] > 1

    return df


# ─────────────────────────────────────────────
# STEP 4: Summary and signal analysis by type
# ─────────────────────────────────────────────

def summarize_enriched_df(df):
    print("\n" + "=" * 60)
    print("ENRICHED DATASET SUMMARY")
    print("=" * 60)

    total     = len(df)
    n_balling = (df["balling"] == 1).sum()
    n_clean   = (df["balling"] == 0).sum()
    print(f"Total frames : {total}")
    print(f"  Balling    : {n_balling}")
    print(f"  Clean      : {n_clean}")

    print(f"\nBalling frames by bead TYPE:")
    ball_df = df[df["balling"] == 1]
    type_counts = ball_df["bead_type"].value_counts(dropna=False)
    for t, n in type_counts.items():
        pct = 100 * n / len(ball_df)
        print(f"  {str(t):<12} : {n:4d} frames  ({pct:.1f}%)")

    print(f"\nBalling frames by bead SIZE (visual):")
    size_counts = ball_df["bead_size_vis"].value_counts(dropna=False)
    for s, n in size_counts.items():
        pct = 100 * n / len(ball_df)
        print(f"  {str(s):<12} : {n:4d} frames  ({pct:.1f}%)")

    print(f"\nMulti-bead frames: {df['is_multi_bead'].sum()}")

    print(f"\nBead size statistics (mm):")
    print(f"  Mean   : {df['bead_size_mm'].mean():.4f} mm")
    print(f"  Median : {df['bead_size_mm'].median():.4f} mm")
    print(f"  Min    : {df['bead_size_mm'].min():.4f} mm")
    print(f"  Max    : {df['bead_size_mm'].max():.4f} mm")

    print(f"\nPer-layer breakdown:")
    for layer in sorted(df["layer"].unique()):
        sub = df[df["layer"] == layer]
        n_b = (sub["balling"] == 1).sum()
        types = sub[sub["balling"] == 1]["bead_type"].value_counts().to_dict()
        print(f"  Layer {layer}: {n_b} balling frames  types={types}")


def analyze_signal_by_type(df, fmap_dict):
    """
    For each bead type (Left/Middle/Right) and size (L/M/S),
    compute Cohen's d of consecutive frame differences.
    This directly answers Zhuo's question:
      "Which types of balling can the melt pool camera predict?"
    """
    import cv2

    def load_img(path):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        img_f = img.astype(np.float32)
        if img.dtype == np.uint16:
            img_f /= 65535.0
        else:
            img_f /= 255.0
        if img_f.ndim == 3:
            img_f = img_f.mean(axis=2)
        return img_f

    print("\n" + "=" * 60)
    print("SIGNAL ANALYSIS BY BEAD TYPE")
    print("Answering: which types can melt pool camera detect?")
    print("=" * 60)
    print("(Computing consecutive frame differences — this may take a minute)")

    # Compute diff for every frame
    diff_scores = {}
    for layer in sorted(df["layer"].unique()):
        layer_df  = df[df["layer"] == layer].sort_values("frame")
        all_frames = sorted(layer_df["frame"].astype(int).tolist())
        fmap = {int(r.frame): str(r.image_path) for _, r in layer_df.iterrows()}

        for i in range(1, len(all_frames)):
            fn      = all_frames[i]
            prev_fn = all_frames[i - 1]
            if fn not in fmap or prev_fn not in fmap:
                continue
            img_c = load_img(fmap[fn])
            img_p = load_img(fmap[prev_fn])
            if img_c is None or img_p is None:
                continue
            diff = np.abs(img_c - img_p)
            # Only count within-class pairs (both same label)
            row_curr = layer_df[layer_df["frame"] == fn].iloc[0]
            row_prev = layer_df[layer_df["frame"] == prev_fn].iloc[0]
            if row_curr["balling"] != row_prev["balling"]:
                continue  # skip boundary frames
            diff_scores[(layer, fn)] = {
                "mean_diff"    : float(diff.mean()),
                "balling"      : int(row_curr["balling"]),
                "bead_type"    : row_curr.get("bead_type"),
                "bead_size_vis": row_curr.get("bead_size_vis"),
                "n_beads"      : int(row_curr.get("n_beads", 0)),
            }

    scores_df = pd.DataFrame(diff_scores.values())
    if scores_df.empty:
        print("  No diff scores computed.")
        return

    clean_scores  = scores_df[scores_df["balling"] == 0]["mean_diff"].values
    clean_mean    = clean_scores.mean()
    clean_std     = clean_scores.std()

    print(f"\n{'Category':<25} {'N ball':>7} {'Ball mean':>10} "
          f"{'Clean mean':>10} {'Cohen d':>8} {'Detectable?':>12}")
    print("-" * 75)

    # Overall
    ball_scores = scores_df[scores_df["balling"] == 1]["mean_diff"].values
    if len(ball_scores) > 0:
        pooled = np.sqrt((ball_scores.std()**2 + clean_std**2) / 2)
        d = (ball_scores.mean() - clean_mean) / (pooled + 1e-10)
        det = "YES ✓" if d > 0.2 else ("marginal" if d > 0 else "NO ✗")
        print(f"{'ALL BALLING':<25} {len(ball_scores):>7} "
              f"{ball_scores.mean():>10.5f} {clean_mean:>10.5f} "
              f"{d:>8.3f} {det:>12}")

    # By bead type
    print()
    for btype in ["Left", "Middle", "Right"]:
        sub = scores_df[
            (scores_df["balling"] == 1) &
            (scores_df["bead_type"] == btype)
        ]["mean_diff"].values
        if len(sub) < 5:
            continue
        pooled = np.sqrt((sub.std()**2 + clean_std**2) / 2)
        d = (sub.mean() - clean_mean) / (pooled + 1e-10)
        det = "YES ✓" if d > 0.2 else ("marginal" if d > 0 else "NO ✗")
        print(f"  Type={btype:<20} {len(sub):>7} "
              f"{sub.mean():>10.5f} {clean_mean:>10.5f} "
              f"{d:>8.3f} {det:>12}")

    # By size
    print()
    for bsize in ["L", "M", "S"]:
        sub = scores_df[
            (scores_df["balling"] == 1) &
            (scores_df["bead_size_vis"] == bsize)
        ]["mean_diff"].values
        if len(sub) < 5:
            continue
        pooled = np.sqrt((sub.std()**2 + clean_std**2) / 2)
        d = (sub.mean() - clean_mean) / (pooled + 1e-10)
        det = "YES ✓" if d > 0.2 else ("marginal" if d > 0 else "NO ✗")
        print(f"  Size={bsize:<20} {len(sub):>7} "
              f"{sub.mean():>10.5f} {clean_mean:>10.5f} "
              f"{d:>8.3f} {det:>12}")

    # Multi-bead vs single-bead
    print()
    for is_multi, label in [(True, "Multi-bead"), (False, "Single-bead")]:
        sub = scores_df[
            (scores_df["balling"] == 1) &
            (scores_df["n_beads"] > 1) == is_multi
        ]["mean_diff"].values
        if len(sub) < 5:
            continue
        pooled = np.sqrt((sub.std()**2 + clean_std**2) / 2)
        d = (sub.mean() - clean_mean) / (pooled + 1e-10)
        det = "YES ✓" if d > 0.2 else ("marginal" if d > 0 else "NO ✗")
        print(f"  {label:<25} {len(sub):>7} "
              f"{sub.mean():>10.5f} {clean_mean:>10.5f} "
              f"{d:>8.3f} {det:>12}")

    return scores_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # Step 1: Load all sheets
    raw_sheets = load_all_sheets(excel_path)

    # Step 2: Parse each sheet
    print("\nParsing bead event tables...")
    all_bead_events = {}
    for layer_num, (sheet_name, df_raw) in raw_sheets.items():
        print(f"\n  Layer {layer_num} (sheet '{sheet_name}'):")
        print(f"  Raw shape: {df_raw.shape}")
        print(f"  First few rows:\n{df_raw.head(8).to_string()}")
        bead_df = parse_sheet(layer_num, df_raw)
        if bead_df is not None and not bead_df.empty:
            print(f"  Parsed: {len(bead_df)} bead events")
            print(bead_df.to_string(index=False))
            all_bead_events[layer_num] = bead_df
        else:
            print(f"  WARNING: could not parse sheet for Layer {layer_num}")

    # Step 3: Merge into final_df
    print("\nMerging bead metadata into final_df...")
    enriched_df = merge_bead_metadata(final_df, all_bead_events)
    print(f"  Done. enriched_df shape: {enriched_df.shape}")
    print(f"  New columns: bead_type, bead_size_vis, bead_size_mm, "
          f"bead_id_rich, n_beads, is_multi_bead")

    # Step 4: Summarize
    summarize_enriched_df(enriched_df)

    # Step 5: Signal analysis by type (answers Zhuo's question)
    fmap_dict = {}  # not needed for this function signature
    scores_df = analyze_signal_by_type(enriched_df, fmap_dict)

    # Make enriched_df available in notebook environment
    print("\n" + "=" * 60)
    print("enriched_df is ready. Key new columns:")
    print("  bead_type     : Left / Middle / Right / None")
    print("  bead_size_vis : L / M / S / None")
    print("  bead_size_mm  : float (measured bead diameter)")
    print("  n_beads       : number of overlapping bead events")
    print("  is_multi_bead : True if n_beads > 1")
    print("\nUse enriched_df instead of final_df for all further analysis.")
    print("=" * 60)
