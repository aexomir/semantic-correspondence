# Semantic Correspondence

A reproducible research pipeline for **dense semantic correspondence**: given a keypoint on a source image, predict its matching location on a target image using frozen or fine-tuned vision backbones. The project compares **DINOv2**, **DINOv3**, and **SAM** on [SPair-71k](https://github.com/ignacio-rocco/SPair-71k), then studies **window soft-argmax** decoding and **cross-dataset generalization** to PF-Pascal and PF-Willow.

The implementation lives in a single Jupyter notebook (`main.ipynb`), optimized for **Google Colab** with Google Drive for datasets, checkpoints, and results. Precomputed evaluation outputs are included under `results/`.

**Repository:** [github.com/aexomir/semantic-correspondence](https://github.com/aexomir/semantic-correspondence)

---

## Table of contents

- [Overview](#overview)
- [Method](#method)
- [Pipeline stages](#pipeline-stages)
- [Models](#models)
- [Datasets](#datasets)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the notebook](#running-the-notebook)
- [Results](#results)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

Semantic correspondence maps semantically related points across image pairs (e.g. the left eye of a cat in image A to the left eye of another cat in image B). This project implements a **feature-matching** baseline:

1. Extract a dense **patch feature map** from each image with a pretrained backbone.
2. For each labeled source keypoint, read its feature vector from the source map.
3. Find the best match in the target map via **cosine similarity** (argmax or window soft-argmax).
4. Score predictions with **PCK** (Percentage of Correct Keypoints) at multiple thresholds.

The notebook runs four stages:

| Stage | Goal |
|-------|------|
| **1** | Frozen-backbone baseline on SPair-71k test |
| **2** | Fine-tune the last *N* transformer blocks with InfoNCE on SPair-71k train |
| **3** | Tune window soft-argmax on validation; report test with best checkpoint |
| **4** | Zero-shot transfer to PF-Pascal and PF-Willow (no training on those datasets) |

---

## Method

### Feature extraction

Each backbone produces a spatial tensor `(1, D, H_feat, W_feat)`:

- **DINOv2 / DINOv3:** Patch tokens from `forward_features`, reshaped to a square grid. DINO models use ImageNet-style normalization and fixed input sizes (518×518 for ViT-L/14, 512×512 for ViT-B/16).
- **SAM ViT-B:** Image-encoder features at native **1024×1024** resolution (no positional-embedding resizing).

Features are **L2-normalized** along the channel dimension before matching.

### Correspondence

For source keypoint `(sx, sy)` in pixels:

1. Map to feature grid indices using image dimensions.
2. Take descriptor `f_src[:, fy, fx]`.
3. Compute similarity `sim[h, w] = ⟨f_src, f_trg[:, h, w]⟩` over the target grid.
4. Predict location with either:
   - **Argmax:** discrete peak cell, converted back to pixel coordinates.
   - **Window soft-argmax:** argmax peak → local `w×w` window → temperature-scaled softmax → expected `(x, y)` in feature space (sub-patch precision).

### Fine-tuning (Stage 2)

Training uses **InfoNCE-style correspondence loss** on SPair-71k training pairs:

- For each keypoint, the source descriptor must classify the correct target patch among all `H×W` locations (cross-entropy on similarity logits).
- Only the last **N ∈ {1, 2, 4}** transformer blocks are unfrozen (plus `norm` / `neck` when present).
- **AdamW**, lr `5e-6`, 3 epochs, batch size 1, gradient accumulation 4, optional AMP.
- Checkpoints are selected by **validation PCK@0.1**; the test split is evaluated only after training for each backbone.

### PCK metrics

**SPair-71k** (Stages 1–3): a prediction is correct at threshold `α` if distance to ground truth ≤ `α × max(bbox_width, bbox_height)` of the target instance.

**PF-Pascal / PF-Willow** (Stage 4): normalization uses **`α × max(H, W)`** of the target image (Proposal Flow convention), not the bounding box.

Default thresholds: **0.05, 0.1, 0.15, 0.2**.

---

## Pipeline stages

### Stage 1 — Frozen baseline (SPair-71k)

- Precompute and cache features for all JPEG images (per model).
- Evaluate on the **test** split with argmax only (`N = 0`, no training).
- Outputs: `summary_per_keypoint_pck.csv`, per-image/per-keypoint/per-category CSVs under `results/step 1/`.

### Stage 2 — Backbone fine-tuning (SPair-71k)

- Grid: 3 backbones × 3 values of **N** (9 runs; skips if checkpoint already exists on Drive).
- Train on `PairAnnotation/trn`, validate each epoch on `PairAnnotation/val` (subset capped at 300 pairs during training for speed).
- Full **test** evaluation after all runs for a backbone complete.
- Checkpoints: `checkpoints/finetuned/{model_key}_last{N}.pth` on Google Drive.
- Summary: `summary_finetune.csv` / `results/step 2/summary_finetune_sorted.csv`.

### Stage 3 — Window soft-argmax (SPair-71k)

- Pick best **N** per backbone from Stage 2 validation **PCK@0.1**.
- Sweep on validation: temperatures `{0.001, 0.005, 0.01, 0.05}`, window sizes `{3, 5, 7, 11}`.
- Report test **argmax** vs **soft-argmax** with the best validation hyperparameters.
- Summary: `results/step 3/stage3_summary.csv`.

### Stage 4 — Cross-dataset generalization

Evaluates each backbone under three conditions **without PF training**:

| Condition | Description |
|-----------|-------------|
| `frozen_argmax` | Stage 1 weights, argmax |
| `ft_N*_argmax` | Best fine-tuned checkpoint from Stage 2/3, argmax |
| `ft_N*_softargmax` | Same checkpoint, Stage 3 soft-argmax settings |

**PF-Pascal:** 20 Pascal VOC categories; pairs built from `parsePascalVOC.mat`.  
**PF-Willow:** Proposal Flow layout (per-category folders with images + `.mat` annotations).

Summaries: `results/step 4/PF-Pascal/pfpascal_extension_summary.csv`, `results/step 4/PF-Willow/pfwillow_extension_summary.csv`.

---

## Models

| Registry key | Architecture | Input size | Patch grid | Pretrained weights (expected on Drive) |
|--------------|--------------|------------|------------|----------------------------------------|
| `dinov2_vitl14` | DINOv2 ViT-L/14 | 518×518 | ~37×37 | `dinov2_vitl14_pretrain_big.pth` |
| `dinov3_vitb16` | DINOv3 ViT-B/16 | 512×512 | 32×32 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` |
| `sam_vit_b_res1024` | SAM ViT-B image encoder | 1024×1024 | encoder output | `sam_vit_b_01ec64.pth` |

DINOv2 and DINOv3 are loaded via `torch.hub` from **local clones** of [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) and [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) (cloned in the setup cells). SAM uses [segment-anything](https://github.com/facebookresearch/segment-anything).

---

## Datasets

### SPair-71k (primary)

- **Source:** [SPair-71k repository](https://github.com/ignacio-rocco/SPair-71k) — place `SPair-71k.tar.gz` or extracted folder on Google Drive.
- **Layout:** `JPEGImages/{category}/*.jpg`, `PairAnnotation/{trn,val,test}/*.json`.
- **Splits used:** `trn` (fine-tune), `val` (checkpoint & soft-argmax sweep), `test` (final metrics).
- **Test scale:** 12,234 image pairs, 88,328 keypoints (as in committed Stage 1 summary).

### PF-Pascal (Stage 4)

- **Download:** [PF-dataset-PASCAL.zip](https://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip) (auto-fetched in notebook if missing on Drive).
- **PCK:** `α × max(H, W)`.

### PF-Willow (Stage 4)

- **Archive:** `pf-willow.zip` on Drive (Proposal Flow distribution).
- Same three evaluation conditions as PF-Pascal; includes macro-averaged per-category PCK in the summary CSV.

---

## Project structure

```
semantic-correspondence/
├── main.ipynb              # Full pipeline (setup → stage 4)
├── requirements.txt        # Python dependencies
├── README.md
└── results/                # Committed experiment outputs
    ├── Final-report.csv      # Combined SPair-71k metrics across stages
    ├── step 1/               # Frozen SPair-71k baselines
    ├── step 2/               # Fine-tuning test results per backbone
    ├── step 3/               # Argmax vs soft-argmax on SPair-71k test
    └── step 4/
        ├── PF-Pascal/        # Cross-dataset PF-Pascal
        └── PF-Willow/        # Cross-dataset PF-Willow
```

**Runtime artifacts (not in git)** — created on Colab / Drive during execution:

| Path | Purpose |
|------|---------|
| `/content/data/` | Local copies of datasets (fast I/O) |
| `/content/features/spair71k/` | Cached feature tensors (`.pt`, float16) |
| `MyDrive/features/spair71k/` | Persistent feature cache |
| `MyDrive/checkpoints/finetuned/` | Fine-tuned weights |
| `MyDrive/results/` | CSV summaries and detailed per-run exports |

---

## Requirements

- **Python 3.10+** recommended
- **CUDA GPU** (Colab T4/A100 or similar); CPU fallback exists but full runs are impractical
- **Google Colab + Google Drive** (notebook assumes Drive mount and paths under `MyDrive`)
- Disk: tens of GB for SPair-71k, PF datasets, features, and checkpoints

Install dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Clone DINO repositories next to the runtime working directory (as the notebook does):

```bash
git clone https://github.com/facebookresearch/dinov2.git
git clone https://github.com/facebookresearch/dinov3.git
pip install omegaconf torchmetrics fvcore iopath submitit ftfy regex scikit-learn termcolor
```

---

## Setup

### 1. Open in Google Colab

Upload or open `main.ipynb` from this repository. Use **Runtime → Change runtime type → GPU**.

### 2. Google Drive layout

Place the following under `MyDrive/` (names match notebook constants):

```
MyDrive/
├── dinov2_vitl14_pretrain_big.pth
├── dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
├── sam_vit_b_01ec64.pth          # SAM ViT-B checkpoint
├── SPair-71k.tar.gz              # or extracted SPair-71k/
├── PF-dataset-PASCAL.zip         # optional; notebook can wget
├── pf-willow.zip                 # Stage 4
├── features/spair71k/            # created by pipeline
├── checkpoints/finetuned/        # created by Stage 2
└── results/                      # CSV outputs
```

Download links for weights:

- **DINOv2 ViT-L/14:** [DINOv2 model zoo](https://github.com/facebookresearch/dinov2#pretrained-models)
- **DINOv3 ViT-B/16:** [DINOv3 releases](https://github.com/facebookresearch/dinov3)
- **SAM ViT-B:** [SAM model checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints) (`sam_vit_b_01ec64.pth`)

### 3. Run setup cells

Execute cells in order from the top through model loading and SPair-71k preparation before any stage-specific cells.

---

## Running the notebook

`main.ipynb` contains **37 cells** (setup + 4 stages). Run sequentially; stage markdown cells describe intent.

| Section | What to run |
|---------|-------------|
| **Setup** | Install deps, clone DINO repos, mount Drive, extract SPair-71k, load models, define helpers & cache |
| **Stage 1** | `precompute_features` + `evaluate_cached` for each backbone; writes frozen baseline |
| **Stage 2** | `FINETUNE_RUNS` grid; `run_finetune_for_backbone` once per backbone (three separate cells) |
| **Stage 3** | Resolve best `N`, validation sweep, test argmax vs soft-argmax |
| **Stage 4a** | PF-Pascal prep + `evaluate_pfpascal` full grid |
| **Stage 4b** | PF-Willow prep + PF-Willow evaluator |

**Quick smoke tests** (before full runs):

```python
precompute_features(['dinov2_vitl14'], limit=50)
results = evaluate_cached('dinov2_vitl14', use_softargmax=False, max_pairs=200)
```

**Resume behavior:** Stage 2 skips `(model_key, N)` if `checkpoints/finetuned/{model_key}_last{N}.pth` already exists. Feature cache skips images that already have `.pt` files.

**Sync:** Call `sync_features_to_drive()` after heavy caching so features persist across Colab sessions. `seed_local_cache_from_drive()` copies Drive cache to local SSD at session start.

---

## Results

Committed numbers summarize the full pipeline. Metrics are **PCK (%)** unless noted.

### SPair-71k test — frozen baseline (Stage 1, argmax)

| Model | PCK@0.05 | PCK@0.1 | PCK@0.15 | PCK@0.2 |
|-------|----------|---------|----------|---------|
| DINOv2 ViT-L/14 | 38.57 | 55.48 | 64.62 | 71.05 |
| DINOv3 ViT-B/16 | 36.66 | 53.57 | 62.69 | 68.37 |
| SAM ViT-B | 15.65 | 24.42 | 31.75 | 38.15 |

### SPair-71k test — after fine-tuning (Stage 2, argmax, best N per row)

| Model | N | PCK@0.1 |
|-------|---|---------|
| DINOv2 ViT-L/14 | 4 | 72.85 |
| DINOv3 ViT-B/16 | 2 | 74.48 |
| SAM ViT-B | 2 | 44.08 |

### SPair-71k test — argmax vs soft-argmax (Stage 3)

| Model | Mode | PCK@0.1 | Best soft-argmax config |
|-------|------|---------|-------------------------|
| DINOv2 | argmax / softargmax | 72.85 / **73.49** | τ=0.05, window=7 |
| DINOv3 | argmax / softargmax | 74.48 / **76.41** | τ=0.05, window=7 |
| SAM | argmax / softargmax | 44.08 / **46.02** | τ=0.05, window=11 |

Soft-argmax consistently improves tight thresholds (e.g. PCK@0.05) without changing the backbone weights.

### PF-Pascal (Stage 4, PCK@0.1, α·max(H,W))

| Model | Frozen | Fine-tuned (argmax) | Fine-tuned (soft-argmax) |
|-------|--------|---------------------|---------------------------|
| DINOv2 (N=4) | 80.69 | 83.75 | **84.29** |
| DINOv3 (N=2) | 86.49 | 85.17 | **86.04** |
| SAM (N=2) | 38.92 | 48.98 | **49.79** |

SPair-71k fine-tuning **transfers positively** to PF-Pascal for DINOv2 and SAM; DINOv3 frozen features are already strong on this benchmark.

### PF-Willow (Stage 4, PCK@0.1)

| Model | Frozen | Fine-tuned (argmax) | Fine-tuned (soft-argmax) |
|-------|--------|---------------------|---------------------------|
| DINOv2 (N=4) | 77.33 | 84.22 | **85.56** |
| DINOv3 (N=2) | 85.67 | 87.78 | **89.22** |
| SAM (N=2) | 46.11 | 55.00 | **55.78** |

Detailed per-image, per-keypoint, and per-category CSVs live under `results/step */`. `results/Final-report.csv` aggregates SPair-71k stages 1–3.

---

## Configuration reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| `PCK_THRESHOLDS` | `[0.05, 0.1, 0.15, 0.2]` | SPair & PF-Pascal/Willow (different normalizers) |
| Fine-tune `lr` | `5e-6` | AdamW on unfrozen blocks only |
| Fine-tune `epochs` | `3` | Best checkpoint by val PCK@0.1 |
| InfoNCE `temperature` | `0.07` | Training loss |
| `grad_accum_steps` | `4` | Effective larger batch |
| `use_amp` | `True` | Mixed precision |
| Stage 3 `TEMP_GRID` | `0.001, 0.005, 0.01, 0.05` | Validation sweep |
| Stage 3 `WINDOW_GRID` | `3, 5, 7, 11` | Validation sweep |
| Feature cache dtype | float16 on CPU | Reloaded as float32 for eval |

**Model registry** (`MODEL_SPECS` in notebook): maps keys to live model objects, `img_size`, and `kind` (`dino` vs `sam`).

---

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| DINOv2 not on GPU | Colab: **Runtime → Change runtime type → GPU**, re-run model load cell |
| `xFormers is not available` | Warning only; DINO falls back to standard attention |
| Missing patch tokens | DINO version mismatch — check `x_norm_patchtokens` vs `x_norm_patch_tokens` keys |
| SAM resolution error | SAM path requires `res=1024` without pos-embed interpolation |
| Drive slow during training | Checkpoints stage to `/content/weights` and write to Drive once per run |
| PF-Willow `__MACOSX` paths | Notebook detects and rejects macOS metadata folders |
| Empty `FINETUNE_RUNS` | All checkpoints exist — delete specific `.pth` to re-run a configuration |

---

## References

- **SPair-71k:** Rocco, Araujo, Gajic, Civera, Torii, Sattler, Pajdla, Pollefeys, Sivic, Poggio, Kümmerle. *Neighbourhood Consensus Networks.* NeurIPS 2018. [Dataset](https://github.com/ignacio-rocco/SPair-71k)
- **DINOv2:** Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- **DINOv3:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- **SAM:** Kirillov et al. *Segment Anything.* ICCV 2023.
- **PF-Pascal / Proposal Flow:** Ham et al. *Proposal Flow.* CVPR 2016. [Project page](https://www.di.ens.fr/willow/research/proposalflow/)
- **PF-Willow:** Same Proposal Flow benchmark family (Willow dataset layout)

---

## Citation

If you use this codebase or the reported numbers, please cite the underlying datasets and model papers above, and link to this repository.

---

## License

Third-party models and datasets carry their own licenses (Meta DINO/SAM terms, SPair-71k / Proposal Flow data agreements). Check each upstream project before redistribution or commercial use.
