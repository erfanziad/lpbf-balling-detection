"""
Save and Load enriched_df
==========================
Saves the combined 14-layer dataset in two formats:

1. Parquet (recommended) — binary, fast, preserves dtypes exactly,
   handles None/NaN correctly, no encoding issues, small file size.
   Load with: pd.read_parquet("enriched_df.parquet")

2. CSV (fallback) — human-readable, universally compatible,
   but requires careful dtype restoration on load.
   Load with: load_enriched_csv("enriched_df.csv")

Run the SAVE section after load_all_layers.py has built enriched_df.
Run the LOAD section at the top of any future analysis notebook.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SAVE_DIR = Path(r"C:\Users\erfan\Downloads\balling_dataset")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = SAVE_DIR / "enriched_df.parquet"
CSV_PATH     = SAVE_DIR / "enriched_df.csv"


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_enriched_df(df, save_parquet=True, save_csv=True):
    """
    Save enriched_df with dtype validation and integrity checks.
    """
    print("Validating before save...")

    # 1. Strip any remaining whitespace in string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().replace("nan", np.nan)

    # 2. Enforce correct dtypes
    int_cols   = ["frame", "balling", "event_id", "bead_id_rich",
                  "n_beads", "layer", "P", "V"]
    float_cols = ["bead_size_mm", "start_pixel", "end_pixel",
                  "size_actual"]
    bool_cols  = ["is_multi_bead"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # 3. Print pre-save summary
    print(f"  Shape        : {df.shape}")
    print(f"  Layers       : {sorted(df['layer'].unique().tolist())}")
    print(f"  Total frames : {len(df)}")
    print(f"  Balling      : {df['balling'].sum()}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")

    # 4. Save
    if save_parquet:
        df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")
        size_mb = PARQUET_PATH.stat().st_size / 1024**2
        print(f"\nSaved parquet: {PARQUET_PATH}  ({size_mb:.2f} MB)")

    if save_csv:
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        size_mb = CSV_PATH.stat().st_size / 1024**2
        print(f"Saved CSV    : {CSV_PATH}  ({size_mb:.2f} MB)")

    # 5. Integrity check — reload parquet and verify
    if save_parquet:
        check = pd.read_parquet(PARQUET_PATH)
        assert len(check) == len(df), "Row count mismatch after reload"
        assert set(check.columns) == set(df.columns), \
            "Column mismatch after reload"
        print(f"\nIntegrity check PASSED — parquet reloads correctly")

    return df


# ─────────────────────────────────────────────
# LOAD (use this at the top of analysis notebooks)
# ─────────────────────────────────────────────

def load_enriched_parquet(path=None):
    """
    Load enriched_df from parquet. Fastest and most reliable.
    Usage:
        enriched_df = load_enriched_parquet()
        final_df    = enriched_df[enriched_df["V"] == 2000].reset_index(drop=True)
    """
    p = Path(path) if path else PARQUET_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {p}\n"
            f"Run save_enriched_df(enriched_df) first."
        )
    df = pd.read_parquet(p)
    print(f"Loaded enriched_df from parquet: {len(df)} frames, "
          f"{df['layer'].nunique()} layers")
    return df


def load_enriched_csv(path=None):
    """
    Load enriched_df from CSV with correct dtypes restored.
    Use only if parquet is not available.
    """
    p = Path(path) if path else CSV_PATH
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)

    # Restore dtypes
    int_cols   = ["frame", "balling", "event_id", "bead_id_rich",
                  "n_beads", "layer", "P", "V"]
    float_cols = ["bead_size_mm", "start_pixel", "end_pixel",
                  "size_actual"]
    bool_cols  = ["is_multi_bead"]
    str_cols   = ["bead_type", "bead_size_vis", "image_path"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    for col in str_cols:
        if col in df.columns:
            # Convert 'nan' strings back to actual NaN
            df[col] = df[col].where(df[col] != "nan", other=np.nan)

    print(f"Loaded enriched_df from CSV: {len(df)} frames, "
          f"{df['layer'].nunique()} layers")
    return df


# ─────────────────────────────────────────────
# QUICK VERIFY (run after loading to sanity-check)
# ─────────────────────────────────────────────

def verify_enriched_df(df):
    """Print a quick sanity check of the loaded dataframe."""
    print("\nDataset verification:")
    print(f"  Shape        : {df.shape}")
    print(f"  Layers       : {sorted(df['layer'].dropna().unique().tolist())}")
    print(f"  V values     : {sorted(df['V'].dropna().unique().tolist())}")
    print(f"  Total frames : {len(df)}")
    print(f"  Balling      : {int(df['balling'].sum())} "
          f"({100*df['balling'].mean():.1f}%)")
    print(f"  Clean        : {int((df['balling']==0).sum())}")

    print(f"\n  Per-layer:")
    print(f"  {'Layer':>7} {'P':>5} {'V':>6} "
          f"{'Frames':>8} {'Ball%':>7}")
    for layer in sorted(df["layer"].dropna().unique()):
        sub   = df[df["layer"] == layer]
        p_val = float(sub["P"].iloc[0])
        v_val = float(sub["V"].iloc[0])
        pct   = 100 * sub["balling"].mean()
        print(f"  {int(layer):>7} {p_val:>5.0f} {v_val:>6.0f} "
              f"{len(sub):>8} {pct:>6.1f}%")

    # Check for whitespace in string columns
    for col in ["bead_type", "bead_size_vis"]:
        if col not in df.columns:
            continue
        bad = [v for v in df[col].dropna().unique()
               if str(v) != str(v).strip()]
        if bad:
            print(f"\n  WARNING: whitespace in {col}: {bad}")

    # Check image paths exist (sample 5)
    if "image_path" in df.columns:
        from pathlib import Path as P
        sample = df[df["image_path"].notna()].sample(
            min(5, len(df)), random_state=42
        )
        missing = [r for r in sample["image_path"]
                   if not P(r).exists()]
        if missing:
            print(f"\n  WARNING: {len(missing)} sample image paths missing:")
            for m in missing:
                print(f"    {m}")
        else:
            print(f"\n  Image path check: 5 random paths OK")

    print("\n  Verification complete.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── SAVE ──────────────────────────────────────────────────────
    # enriched_df must already be defined (run load_all_layers.py first)
    print("Saving enriched_df...")
    enriched_df = save_enriched_df(enriched_df,
                                    save_parquet=True,
                                    save_csv=True)

    # ── VERIFY RELOAD ─────────────────────────────────────────────
    print("\nVerifying parquet reload...")
    enriched_df_reloaded = load_enriched_parquet()
    verify_enriched_df(enriched_df_reloaded)

    # ── USAGE INSTRUCTIONS ────────────────────────────────────────
    print(f"""
To load in any future notebook, just run:

    from pathlib import Path
    import pandas as pd

    enriched_df = pd.read_parquet(
        r"{PARQUET_PATH}"
    )
    final_df = enriched_df[
        enriched_df["V"] == 2000
    ].reset_index(drop=True)

Or use the helper function:

    from save_enriched_df import load_enriched_parquet, verify_enriched_df
    enriched_df = load_enriched_parquet()
    verify_enriched_df(enriched_df)
""")
