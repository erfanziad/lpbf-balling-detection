"""
Improved VLM Evaluation — Sequence Tiling + Stratified Few-Shot
================================================================
Lessons from initial experiment:
  1. Single diff frames → Claude defaults to balling (all images look active)
  2. Few-shot helps (AUC 0.46→0.60) but examples were random
  3. Confidence is uniform (0.82-0.87) → model not truly discriminating

Two improvements:
  A. SEQUENCE INPUT: tile 5 consecutive diffs into one grid image
     → gives Claude the same temporal context as the CNN-LSTM
     → balling = persistent asymmetric bright spot across frames
     → clean = consistent symmetric pattern that doesn't evolve

  B. STRATIFIED EXAMPLES: for few-shot, pick examples that cover
     the full range of what balling and clean look like:
     Balling examples: one Right-type, one Left-type, one Middle-type
     Clean examples: one from each V group (1800, 2000)
     → better calibration of the decision boundary

Two modes evaluated:
  1. sequence_zero_shot: 5-frame grid, no examples
  2. sequence_few_shot:  5-frame grid + stratified examples

Evaluation mirrors LOLO: examples always from OTHER layers.
Primary metric: AUC-ROC vs CNN-LSTM baseline.
"""

import base64
import json
import time
import random
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import requests
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("extended_analysis/vlm_evaluation_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── API ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = "YOUR_API_KEY_HERE"   # ← paste your key here

# ── Config ───────────────────────────────────────────────────────
ALL_LAYERS      = list(range(226, 232)) + list(range(245, 253))
SEQ_LEN         = 5       # number of consecutive diffs to tile
TILE_SIZE       = (128, 128)   # size of each diff tile
MAX_TEST_WINDOWS = 40     # windows per layer
SLEEP_BETWEEN    = 1.5    # seconds between API calls
N_BALL_EXAMPLES  = 3      # balling examples for few-shot
N_CLEAN_EXAMPLES = 3      # clean examples for few-shot

# CNN-LSTM LOLO AUC for comparison
CNN_LOLO_AUC = {
    226: 0.684, 227: 0.711, 228: 0.773, 229: 0.662,
    230: 0.569, 231: 0.479, 245: 0.654, 246: 0.788,
    247: 0.705, 248: 0.582, 249: 0.842, 250: 0.711,
    251: 0.626, 252: 0.644,
}


# ─────────────────────────────────────────────
# IMAGE UTILITIES
# ─────────────────────────────────────────────

def load_img(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    img_f = img.astype(np.float32)
    img_f = img_f / 65535.0 if img.dtype == np.uint16 else img_f / 255.0
    if img_f.ndim == 3:
        img_f = img_f.mean(axis=2)
    return img_f


def make_single_diff(path_curr, path_prev, size=TILE_SIZE):
    """Single diff image as numpy array (hot colormap)."""
    ic = load_img(path_curr)
    ip = load_img(path_prev)
    if ic is None or ip is None:
        return None
    diff = np.abs(ic - ip)
    dmax = diff.max()
    if dmax < 1e-8:
        return None
    diff_norm    = diff / dmax
    diff_resized = cv2.resize(
        diff_norm, size, interpolation=cv2.INTER_LINEAR
    )
    diff_uint8 = (diff_resized * 255).astype(np.uint8)
    diff_color = cv2.applyColorMap(diff_uint8, cv2.COLORMAP_HOT)
    diff_rgb   = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
    return diff_rgb


def make_sequence_grid(frame_paths, prev_paths, size=TILE_SIZE, cols=5):
    """
    Tile multiple diff images into a single grid image.
    Layout: 1 row of N diffs side by side with frame labels.
    Also adds a thin separator line between frames for clarity.
    """
    diffs = []
    for curr, prev in zip(frame_paths, prev_paths):
        d = make_single_diff(curr, prev, size)
        if d is not None:
            diffs.append(d)

    if not diffs:
        return None

    n      = len(diffs)
    sep    = 3      # separator width in pixels
    h, w   = size
    label_h = 20   # space for frame label

    # Build grid: 1 row
    total_w = n * w + (n-1) * sep
    total_h = h + label_h
    grid    = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    grid[:] = (30, 30, 30)  # dark background

    for i, diff in enumerate(diffs):
        x = i * (w + sep)
        grid[label_h:label_h+h, x:x+w] = diff

    # Add time arrow text using PIL
    pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil)
    draw.text(
        (5, 2),
        f"← time (t-{n} to t) — {n} consecutive frame diffs →",
        fill=(200, 200, 200)
    )
    grid = np.array(pil)

    # Encode to PNG bytes
    pil_final = Image.fromarray(grid)
    buf = io.BytesIO()
    pil_final.save(buf, format="PNG")
    return buf.getvalue()


def img_to_b64(img_bytes):
    return base64.standard_b64encode(img_bytes).decode("utf-8")


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert in Laser Powder Bed Fusion (LPBF) \
additive manufacturing process monitoring using melt pool cameras.

IMAGE FORMAT:
You will see a SEQUENCE of consecutive frame differences from a melt pool camera.
Each panel shows |frame[t] - frame[t-1]| — bright/hot = large pixel change.
The panels are ordered left to right: oldest to most recent.
The laser scans horizontally left to right in each panel.

BALLING DEFECT SIGNATURE:
- A bright asymmetric spot that PERSISTS across multiple panels
- The bright spot is displaced laterally from the main pool axis
- OR a sudden irregular eruption that does not match the smooth pool pattern
- The signal EVOLVES consistently across the time sequence

CLEAN SCANNING SIGNATURE:
- Bright region is smooth, symmetric, and follows the scan direction
- The pattern is CONSISTENT across panels — no sudden changes
- The pool head and tail are the only bright regions

IMPORTANT: The diff images always show SOME activity — even clean scanning
has pixel changes. The key is whether the bright region is:
  - Asymmetric / laterally displaced → balling
  - Symmetric / centered on scan axis → clean

Respond ONLY with valid JSON (no markdown, no explanation outside JSON):
{"prediction": "balling" or "clean", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


def build_zero_shot_msg(grid_b64):
    return [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64",
                           "media_type": "image/png",
                           "data": grid_b64},
            },
            {
                "type": "text",
                "text": (
                    "Analyze this sequence of 5 consecutive melt pool "
                    "frame differences.\n"
                    "Is a balling defect occurring?\n\n"
                    "Respond with JSON only: "
                    "{\"prediction\": \"balling\" or \"clean\", "
                    "\"confidence\": 0.0-1.0, "
                    "\"reasoning\": \"one sentence\"}"
                ),
            },
        ],
    }]


def build_few_shot_msg(grid_b64, examples):
    """
    examples: list of dicts with 'grid_b64', 'label', 'bead_type', 'desc'
    """
    content = []
    content.append({
        "type": "text",
        "text": (
            f"Here are {len(examples)} calibration examples showing "
            f"what balling and clean look like in THIS camera setup. "
            f"Each shows 5 consecutive frame differences."
        )
    })

    for i, ex in enumerate(examples):
        content.append({
            "type": "text",
            "text": f"\nExample {i+1}: {ex['label'].upper()} — {ex['desc']}"
        })
        content.append({
            "type": "image",
            "source": {"type": "base64",
                       "media_type": "image/png",
                       "data": ex["grid_b64"]},
        })

    content.append({
        "type": "text",
        "text": (
            "\n\nNow classify this new sequence:\n"
            "Respond with JSON only: "
            "{\"prediction\": \"balling\" or \"clean\", "
            "\"confidence\": 0.0-1.0, "
            "\"reasoning\": \"one sentence\"}"
        )
    })
    content.append({
        "type": "image",
        "source": {"type": "base64",
                   "media_type": "image/png",
                   "data": grid_b64},
    })

    return [{"role": "user", "content": content}]


# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────

def call_claude(messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 200,
                    "system": SYSTEM_PROMPT,
                    "messages": messages,
                },
                timeout=60,
            )
            if response.status_code != 200:
                print(f"      API {response.status_code}: "
                      f"{response.text[:100]}")
                time.sleep(2**attempt)
                continue

            text = response.json()["content"][0]["text"].strip()

            # Strip markdown if present
            if "```" in text:
                parts = text.split("```")
                for p in parts:
                    p = p.strip()
                    if p.startswith("json"):
                        p = p[4:].strip()
                    if p.startswith("{"):
                        text = p
                        break

            # Find JSON object
            start = text.find("{")
            end   = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]

            return json.loads(text)

        except json.JSONDecodeError:
            print(f"      JSON parse failed: {text[:80]}")
            time.sleep(1)
        except Exception as e:
            print(f"      Error: {e}")
            time.sleep(2**attempt)
    return None


# ─────────────────────────────────────────────
# BUILD SEQUENCE WINDOWS
# ─────────────────────────────────────────────

def build_sequence_windows(layer, enriched_df,
                            max_windows=MAX_TEST_WINDOWS):
    """Build sliding windows of SEQ_LEN consecutive frames."""
    g    = enriched_df[enriched_df["layer"]==layer].sort_values("frame")
    fmap = {int(r["frame"]): str(r["image_path"])
            for _, r in g.iterrows()
            if "image_path" in g.columns}
    all_f = sorted(fmap.keys())
    ball  = set(g[g["balling"]==1]["frame"].astype(int))

    windows = []
    for i in range(SEQ_LEN, len(all_f)):
        seq_frames = all_f[i-SEQ_LEN:i+1]   # SEQ_LEN+1 frames → SEQ_LEN diffs
        if any(f not in fmap for f in seq_frames):
            continue
        center_f = seq_frames[SEQ_LEN // 2]
        label    = int(any(f in ball for f in seq_frames))
        cr = g[g["frame"]==center_f]
        if cr.empty:
            cr = g.iloc[[(g["frame"]-center_f).abs().argmin()]]
        r = cr.iloc[0]
        windows.append({
            "frame_paths": [fmap[f] for f in seq_frames[1:]],   # current
            "prev_paths" : [fmap[f] for f in seq_frames[:-1]],  # previous
            "center_f"   : center_f,
            "label"      : label,
            "bead_type"  : r.get("bead_type"),
            "bead_size"  : r.get("bead_size_vis"),
            "bead_size_mm": float(r["bead_size_mm"])
                            if pd.notna(r.get("bead_size_mm")) else np.nan,
        })

    # Balanced sampling
    pos = [w for w in windows if w["label"]==1]
    neg = [w for w in windows if w["label"]==0]
    n   = min(max_windows//2, len(pos), len(neg))
    if n == 0:
        return []
    random.seed(42)
    sampled = random.sample(pos, n) + random.sample(neg, n)
    random.shuffle(sampled)
    return sampled


# ─────────────────────────────────────────────
# STRATIFIED EXAMPLE BUILDER
# ─────────────────────────────────────────────

def build_stratified_examples(train_layers, enriched_df):
    """
    Build few-shot examples covering the full range of balling types.

    Balling examples (one per type if available):
      - Right-type bead  (most detectable — clear lateral asymmetry)
      - Left-type bead   (lateral but less consistent)
      - Middle-type bead (center track — harder to see)

    Clean examples (one per V group):
      - Clean from V=1800 layer
      - Clean from V=2000 layer
      - One more random clean

    Each example is a 5-frame sequence grid.
    """
    examples = []

    # ── Balling examples — by type ────────────────────────────────
    balling_targets = [
        ("Right", "Right-type bead: lateral asymmetric bright spot displaced from pool center"),
        ("Left",  "Left-type bead: asymmetric bright spot on opposite side of pool"),
        ("Middle","Middle-type bead: subtle central perturbation in pool"),
    ]

    for btype, desc in balling_targets:
        # Find a layer in train_layers that has this bead type
        found = False
        for layer in random.sample(train_layers, len(train_layers)):
            g = enriched_df[
                (enriched_df["layer"]==layer) &
                (enriched_df["balling"]==1) &
                (enriched_df["bead_type"]==btype)
            ].sort_values("frame")
            if len(g) < SEQ_LEN + 1:
                continue

            fmap = {int(r["frame"]): str(r["image_path"])
                    for _, r in enriched_df[
                        enriched_df["layer"]==layer
                    ].iterrows() if "image_path" in enriched_df.columns}
            all_f = sorted(fmap.keys())

            # Find a sequence that contains this bead type
            ball_frames = set(g["frame"].astype(int))
            for i in range(SEQ_LEN, len(all_f)):
                seq = all_f[i-SEQ_LEN:i+1]
                if any(f not in fmap for f in seq): continue
                if any(f in ball_frames for f in seq):
                    grid_bytes = make_sequence_grid(
                        [fmap[f] for f in seq[1:]],
                        [fmap[f] for f in seq[:-1]]
                    )
                    if grid_bytes:
                        examples.append({
                            "grid_b64": img_to_b64(grid_bytes),
                            "label"   : "balling",
                            "desc"    : desc,
                        })
                        found = True
                        break
            if found:
                break

    # ── Clean examples — by V group ───────────────────────────────
    clean_targets = [
        (1800, "Clean scanning at V=1800mm/s: smooth symmetric pool dynamics"),
        (2000, "Clean scanning at V=2000mm/s: smooth symmetric pool dynamics"),
        (None, "Clean scanning: no bead formation, consistent pool pattern"),
    ]

    for v_target, desc in clean_targets:
        cands = [l for l in train_layers
                 if v_target is None or
                 int(enriched_df[enriched_df["layer"]==l]["V"].iloc[0]) == v_target]
        if not cands:
            cands = train_layers

        found = False
        for layer in random.sample(cands, len(cands)):
            g = enriched_df[enriched_df["layer"]==layer].sort_values("frame")
            fmap = {int(r["frame"]): str(r["image_path"])
                    for _, r in g.iterrows()
                    if "image_path" in g.columns}
            all_f = sorted(fmap.keys())
            ball  = set(g[g["balling"]==1]["frame"].astype(int))

            for i in range(SEQ_LEN, len(all_f)):
                seq = all_f[i-SEQ_LEN:i+1]
                if any(f not in fmap for f in seq): continue
                # All frames in sequence must be clean
                if any(f in ball for f in seq): continue
                grid_bytes = make_sequence_grid(
                    [fmap[f] for f in seq[1:]],
                    [fmap[f] for f in seq[:-1]]
                )
                if grid_bytes:
                    examples.append({
                        "grid_b64": img_to_b64(grid_bytes),
                        "label"   : "clean",
                        "desc"    : desc,
                    })
                    found = True
                    break
            if found:
                break

    # Shuffle so balling and clean are interleaved
    random.shuffle(examples)
    print(f"      Built {len(examples)} stratified examples: "
          f"{sum(1 for e in examples if e['label']=='balling')} balling, "
          f"{sum(1 for e in examples if e['label']=='clean')} clean")
    return examples


# ─────────────────────────────────────────────
# EVALUATE ONE LAYER
# ─────────────────────────────────────────────

def evaluate_layer(layer, enriched_df, mode, examples=None):
    windows = build_sequence_windows(layer, enriched_df)
    if not windows:
        print(f"    No windows for L{layer}")
        return []

    results = []
    n_total = len(windows)

    for i, w in enumerate(windows):
        grid_bytes = make_sequence_grid(w["frame_paths"], w["prev_paths"])
        if grid_bytes is None:
            continue

        grid_b64 = img_to_b64(grid_bytes)

        if mode == "sequence_zero_shot" or examples is None:
            messages = build_zero_shot_msg(grid_b64)
        else:
            messages = build_few_shot_msg(grid_b64, examples)

        parsed = call_claude(messages)
        time.sleep(SLEEP_BETWEEN)

        if parsed is None:
            print(f"      [{i+1}/{n_total}] f{w['center_f']}: failed")
            continue

        pred       = parsed.get("prediction", "clean").lower().strip()
        confidence = float(parsed.get("confidence", 0.5))
        reasoning  = parsed.get("reasoning", "")

        # Clamp confidence
        confidence = max(0.01, min(0.99, confidence))

        # Convert to probability score
        prob = confidence if pred == "balling" else 1.0 - confidence

        results.append({
            "layer"     : int(layer),
            "center_f"  : w["center_f"],
            "label"     : w["label"],
            "pred_label": pred,
            "confidence": confidence,
            "pred_prob" : prob,
            "bead_type" : w["bead_type"],
            "bead_size" : w["bead_size"],
            "reasoning" : reasoning,
            "mode"      : mode,
        })

        correct = "✓" if (prob >= 0.5) == (w["label"] == 1) else "✗"
        print(f"      [{i+1}/{n_total}] f{w['center_f']:>4} "
              f"true={w['label']} "
              f"pred={pred}({confidence:.2f}) {correct}")

    return results


# ─────────────────────────────────────────────
# FULL LOLO EVALUATION
# ─────────────────────────────────────────────

def evaluate_all_layers(enriched_df, mode, target_layers):
    print(f"\n{'='*65}")
    print(f"VLM EVALUATION: {mode.upper()}")
    print(f"Layers: {target_layers}")
    print(f"{'='*65}")

    all_results = []
    layer_aucs  = {}

    for layer in target_layers:
        p = int(enriched_df[enriched_df["layer"]==layer]["P"].iloc[0])
        v = int(enriched_df[enriched_df["layer"]==layer]["V"].iloc[0])
        print(f"\n  Layer {layer} P={p}W V={v}:")

        # Build examples from OTHER layers for few-shot
        if mode == "sequence_few_shot":
            train_layers = [l for l in ALL_LAYERS if l != layer]
            random.seed(layer)  # reproducible per fold
            examples = build_stratified_examples(train_layers, enriched_df)
        else:
            examples = None

        results = evaluate_layer(layer, enriched_df, mode, examples)
        all_results.extend(results)

        if results:
            df_r = pd.DataFrame(results)
            try:
                auc = roc_auc_score(df_r["label"], df_r["pred_prob"])
            except Exception:
                auc = float("nan")
            layer_aucs[layer] = auc
            cnn  = CNN_LOLO_AUC.get(layer, np.nan)
            delt = auc - cnn if not np.isnan(cnn) else np.nan
            d_str = f"{delt:+.3f}" if not np.isnan(delt) else "n/a"
            print(f"    → VLM={auc:.3f}  CNN={cnn:.3f}  Δ={d_str}")

        pd.DataFrame(all_results).to_csv(
            OUTPUT_DIR / f"{mode}_predictions.csv", index=False
        )

    return all_results, layer_aucs


# ─────────────────────────────────────────────
# COMPARISON PLOT
# ─────────────────────────────────────────────

def plot_results(results_by_mode, target_layers):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Improved VLM evaluation: sequence input + stratified few-shot\n"
        "Each input = 5 consecutive diff images tiled as a sequence grid\n"
        "vs CNN-LSTM trained on same layers (LOLO evaluation)",
        fontsize=10, fontweight="bold"
    )

    colors = {
        "sequence_zero_shot": "#E53935",
        "sequence_few_shot" : "#2E7D32",
        "cnn_lstm"          : "#1565C0",
    }

    x = np.arange(len(target_layers))
    w = 0.25

    ax = axes[0]
    for i, (mode, layer_aucs) in enumerate(results_by_mode.items()):
        aucs = [layer_aucs.get(l, np.nan) for l in target_layers]
        ax.bar(x + (i-1)*w, aucs, w,
               label=mode.replace("_"," "),
               color=colors.get(mode, f"C{i}"),
               alpha=0.85, edgecolor="black", linewidth=0.5)

    cnn_aucs = [CNN_LOLO_AUC.get(l, np.nan) for l in target_layers]
    ax.bar(x + len(results_by_mode)*w - w, cnn_aucs, w,
           label="CNN-LSTM", color=colors["cnn_lstm"],
           alpha=0.85, edgecolor="black", linewidth=0.5)

    ax.axhline(0.5,  color="black", linewidth=1.2, linestyle="--")
    ax.axhline(0.65, color="green", linewidth=1,   linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in target_layers],
                        fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Per-layer AUC", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels, means = [], []
    for mode, layer_aucs in results_by_mode.items():
        vals = [v for v in layer_aucs.values() if not np.isnan(v)]
        if vals:
            labels.append(mode.replace("_","\n"))
            means.append(np.mean(vals))
    cnn_vals = [CNN_LOLO_AUC.get(l, np.nan) for l in target_layers
                if not np.isnan(CNN_LOLO_AUC.get(l, np.nan))]
    labels.append("CNN-LSTM\n(trained)")
    means.append(np.mean(cnn_vals) if cnn_vals else np.nan)

    bar_colors = [colors.get(l.replace("\n","_"), "gray") for l in labels]
    bars = ax.bar(range(len(labels)), means,
                  color=bar_colors, alpha=0.85,
                  edgecolor="black", linewidth=0.8)
    ax.axhline(0.5,  color="black", linewidth=1.2, linestyle="--")
    ax.axhline(0.65, color="green", linewidth=1,   linestyle=":")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean AUC-ROC"); ax.set_ylim(0.3, 1.0)
    ax.set_title("Mean AUC comparison", fontsize=10)
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center",
                    va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "vlm_v2_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    random.seed(42)

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
    print(f"  {len(enriched_df)} frames  "
          f"balling={int(enriched_df['balling'].sum())}")

    # ── Target layers ─────────────────────────────────────────────
    # Use 5 diverse layers covering both V values and a range of AUCs
    # L247: V=2000, mixed types, CNN=0.705
    # L249: V=2000, Right-heavy, CNN=0.842 (best CNN layer)
    # L250: V=2000, all types, CNN=0.711
    # L227: V=1800, strong signal, CNN=0.711
    # L230: V=1800, CNN=0.569 (hard layer)
    TARGET_LAYERS = [227, 247, 249, 250, 230]

    print(f"\nTarget layers: {TARGET_LAYERS}")
    print(f"Estimated API calls: "
          f"{len(TARGET_LAYERS) * MAX_TEST_WINDOWS * 2} "
          f"(2 modes x {MAX_TEST_WINDOWS} windows x {len(TARGET_LAYERS)} layers)")
    print(f"Estimated cost: "
          f"~${len(TARGET_LAYERS) * MAX_TEST_WINDOWS * 2 * 0.005:.1f} "
          f"(sequence images cost more than single frames)")

    results_by_mode = {}

    # Mode 1: sequence zero-shot
    print(f"\n{'='*65}")
    print("MODE 1: sequence_zero_shot")
    print("  Input: 5-frame diff sequence grid, no examples")
    print(f"{'='*65}")
    r1, auc1 = evaluate_all_layers(
        enriched_df, "sequence_zero_shot", TARGET_LAYERS
    )
    results_by_mode["sequence_zero_shot"] = auc1

    # Mode 2: sequence few-shot with stratified examples
    print(f"\n{'='*65}")
    print("MODE 2: sequence_few_shot (stratified)")
    print("  Input: 5-frame sequence + examples covering all bead types")
    print(f"{'='*65}")
    r2, auc2 = evaluate_all_layers(
        enriched_df, "sequence_few_shot", TARGET_LAYERS
    )
    results_by_mode["sequence_few_shot"] = auc2

    # Plot
    plot_results(results_by_mode, TARGET_LAYERS)

    # Final summary
    print(f"\n{'='*65}")
    print("FINAL RESULTS")
    print(f"{'='*65}")
    print(f"\n  {'Layer':>7} {'CNN':>7} {'Seq-0shot':>11} {'Seq-fshot':>11} "
          f"{'Best VLM':>10}")
    print("  " + "-" * 55)
    for layer in TARGET_LAYERS:
        cnn  = CNN_LOLO_AUC.get(layer, np.nan)
        z    = auc1.get(layer, np.nan)
        f    = auc2.get(layer, np.nan)
        best = max([v for v in [z,f] if not np.isnan(v)], default=np.nan)
        d    = best - cnn if not np.isnan(cnn) else np.nan
        d_str = f"{d:+.3f}" if not np.isnan(d) else "n/a"
        print(f"  {layer:>7} {cnn:>7.3f} {z:>11.3f} {f:>11.3f} "
              f"{best:>10.3f} ({d_str} vs CNN)")

    all_z = [v for v in auc1.values() if not np.isnan(v)]
    all_f = [v for v in auc2.values() if not np.isnan(v)]
    all_c = [CNN_LOLO_AUC.get(l,np.nan) for l in TARGET_LAYERS
             if not np.isnan(CNN_LOLO_AUC.get(l,np.nan))]
    print(f"\n  Mean AUC:")
    print(f"    CNN-LSTM          : {np.mean(all_c):.3f}")
    print(f"    Sequence zero-shot: {np.mean(all_z):.3f}")
    print(f"    Sequence few-shot : {np.mean(all_f):.3f}")

    gap = np.mean(all_c) - np.mean(all_f)
    print(f"\n  Gap (CNN - best VLM): {gap:+.3f}")
    if gap < 0.05:
        print("  → VLM matches CNN with no task-specific training")
        print("  → Foundation models transfer well to LPBF monitoring")
    elif gap < 0.15:
        print("  → VLM is competitive but slightly below CNN")
        print("  → Few examples close most of the gap")
    else:
        print("  → CNN substantially outperforms VLM")
        print("  → Domain-specific training is essential for manufacturing")

    print(f"\nOutputs: {OUTPUT_DIR}/")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}")
