# In-Situ Balling Detection in LPBF Using Multi-Modal Camera and Surface Profilometry

**Erfan Ziad — Arizona State University / NIST**

This repository contains the full codebase for the paper:
> *In-Situ Balling Detection in Laser Powder Bed Fusion Using Multi-Modal Camera and Surface Profilometry*

---

## Overview

Balling is a common defect in Laser Powder Bed Fusion (LPBF) where the molten track breaks into discrete spherical beads. We present a multi-modal CNN-LSTM approach that fuses top-down melt pool camera sequences with spatially aligned surface profilometry height patches.

**Best model:** Mean LOLO AUC = 0.720 (vs 0.673 camera-only baseline), with mean AUC = 0.814 on the five layers with profilometry data (+0.126).

**Key findings:**
- Bead position type is the primary camera detectability determinant, driven by gas flow geometry
- Camera and profilometry are genuinely complementary (Pearson r ≈ 0) — they measure different physical phenomena
- Middle-type and small beads previously undetectable from camera alone become detectable with height fusion
- V conditioning (scan speed) hurts generalization through shortcut learning on a binary variable

---

## Dataset

14 single-track LPBF experiments:
- **L226–L231**: V = 1850 mm/s, P = 360–460 W
- **L245–L252**: V = 2000 mm/s, P = 320–460 W
- 3,078 total frames, 770 balling frames (25.0%)
- Labels from post-build optical microscopy (Left/Middle/Right bead type, Large/Medium/Small size)
- Profilometry available for L248–L252

> **Data availability:** The dataset is NIST-sponsored and not publicly released. Contact the authors for access inquiries.

---

## Repository Structure

```
src/
  01_build_dataset.py              # Build enriched_df.csv with all labels and metadata
  02_load_labels.py                # Load and verify rich bead-level labels
  03_signal_analysis.py            # Cohen's d per layer — raw camera signal strength
  04_label_ablation.py             # Label definition ablation (All / Lateral / Size≥0.10mm)
  05_input_representation_ablation.py  # DIFF vs RAW vs CONCAT input comparison (LOLO)
  06_profilometry_alignment.py     # Cross-correlation alignment of height map to camera coords
  07_profilometry_overlay.py       # Height map visualization with balling label overlays
  08_height_profiles.py            # Along-scan height profiles with label spans
  09_camera_height_correlation.py  # Pearson r between height and camera prediction scores
  10_baseline_camera_only.py       # Baseline LOLO: camera diff images + P+V (mean AUC=0.673)
  11_multimodal_gated_height.py    # Best model: gated height patch + smart val + SEQ=8 (0.720)
  12_multimodal_local_anomaly.py   # Exp B: local anomaly normalization + scalar features
  13_augmented_hyperparam_search.py # Augmentation + hyperparameter grid search
  14_process_param_ablation.py     # P+V / P-only / V-only / none — process param contribution
  15_vlm_evaluation.py             # Zero-shot and few-shot VLM (Claude) evaluation

figures/
  generate_signal_figure.py        # Reproduces a2_signal_cohens_d.png (3-panel figure)

docs/
  balling_paper.tex                # Draft academic paper
  literature_review.tex            # Related work section
  references.bib                   # BibTeX references
```

---

## Running the Code

### Requirements

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pillow tqdm
```

### Execution order

```bash
# 1. Build dataset (requires raw camera images and label CSVs)
python src/01_build_dataset.py

# 2. Signal analysis (no model training needed)
python src/03_signal_analysis.py

# 3. Profilometry alignment (requires qq_exp3_c6.csv from NIST)
python src/06_profilometry_alignment.py
python src/07_profilometry_overlay.py

# 4. Run best model (SEQ=8, gated height patch, smart val)
python src/11_multimodal_gated_height.py

# 5. Ablation studies
python src/05_input_representation_ablation.py
python src/14_process_param_ablation.py

# 6. VLM evaluation (requires Anthropic API key)
python src/15_vlm_evaluation.py
```

### Data paths

All scripts expect data at Windows paths — update these at the top of each file:

```python
# Camera images and label CSVs
TRACK_CSV_DIR = Path(r"C:\Users\erfan\Downloads\Erfan_balling_data_updated 2\...")

# Enriched dataset (output of 01_build_dataset.py)
ENRICHED_DF   = Path(r"C:\Users\erfan\Downloads\balling_dataset\enriched_df.csv")

# Profilometry height map (NIST)
PROFIL_CSV    = Path(r"C:\Users\erfan\Downloads\qq_exp3_c6.csv")
```

---

## Model Architecture

The best model (`src/11_multimodal_gated_height.py`) fuses four input streams:

| Stream | Encoder | Output dim |
|---|---|---|
| 7 diff images `\|I_t - I_{t-1}\|` | DiffFrameCNN (shared) → LSTM (2-layer, h=64) | 64+64 |
| Last raw frame | FrameCNN | 32 |
| Height patch × gate `g` | HeightPatchCNN | 16 |
| [P, V] scalars | Linear branch | 16 |
| Gate scalar `g` | (explicit) | 1 |
| **Fused** | Linear(193→64)→ReLU→Linear(64→1) | **193** |

The gate `g=1` for L248–L252 (real height data), `g=0` for all others (zeros).
Smart validation: validation layer chosen by nearest (P,V) distance to test layer.

---

## Results Summary

| Model | Mean AUC | Profil Δ | No-H Δ |
|---|---|---|---|
| Baseline (camera, SEQ=6) | 0.673 | — | — |
| + height patch, no gate | 0.686 | +0.120 | −0.026 |
| + gate + smart val (SEQ=6) | 0.702 | +0.100 | +0.010 |
| **Best: gated + smart val (SEQ=8)** | **0.720** | **+0.126** | **+0.023** |
| Local anomaly + scalars (SEQ=8) | 0.677 | +0.142 | −0.053 |

Per-stratum AUC (pooled, best model vs baseline):

| Stratum | Baseline | Best | Δ |
|---|---|---|---|
| Overall | 0.611 | 0.673 | +0.062 |
| Type=Middle | 0.624 | 0.750 | +0.126 |
| Type=Right | 0.804 | 0.760 | −0.044 |
| Size=S | 0.616 | 0.713 | +0.097 |

---

## Citation

```bibtex
@article{ziad2026balling,
  title={In-Situ Balling Detection in Laser Powder Bed Fusion
         Using Multi-Modal Camera and Surface Profilometry},
  author={Ziad, Erfan and Yang, Zhuo and Lu, Yan and Ju, Feng},
  journal={},
  year={2026},
  note={Manuscript in preparation}
}
```

---

## Acknowledgments

This work is supported by NIST. Profilometry data provided by Dr. Zhuo Yang and Dr. Yan Lu, NIST. Research conducted at Arizona State University under the supervision of Dr. Feng Ju.
