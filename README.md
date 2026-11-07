# 🔬 Deep Learning for Kidney Microvasculature Instance Segmentation

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.16%2B-orange.svg)](https://pytorch.org/vision/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-HuBMAP%20Kidney-teal.svg)](https://commonfund.nih.gov/HuBMAP)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning framework and interactive diagnostic dashboard for **instance segmentation of renal microvasculature** (capillaries, arterioles, and venules) in digital pathology images. Built upon a COCO-pretrained **Mask R-CNN (ResNet-50-FPN v2)** architecture, the pipeline features domain-specific **pathology stain augmentations (HED jitter & LAB color transfer)**, **two-stage optimization**, a **WSI-held-out validation protocol**, and a **4-checkpoint ensemble with test-time augmentation (TTA)**, advancing from a 0.104 baseline to **0.5300 CV mAP**.

---

## 📑 Table of Contents
- [Overview & Clinical Motivation](#-overview--clinical-motivation)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Pathology-Specific Augmentations](#-pathology-specific-augmentations)
- [Two-Stage Training & Validation Protocol](#-two-stage-training--validation-protocol)
- [Experimental Benchmarks & Progression](#-experimental-benchmarks--progression)
- [Interactive Dashboard](#-interactive-dashboard)
- [Project Directory Structure](#-project-directory-structure)
- [Quickstart Guide](#-quickstart-guide)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Known Limitations & Disclaimer](#-known-limitations--disclaimer)

---

## 🩺 Overview & Clinical Motivation

Automated instance segmentation of microvasculature in Periodic Acid–Schiff (PAS) and Hematoxylin & Eosin (H&E) stained histology tissue is crucial for biological mapping initiatives (such as the NIH **HuBMAP** program) and renal disease profiling. 

### Why Instance Segmentation over Semantic Segmentation?
* **Capillary Cluttering**: Renal capillaries cluster tightly together. Semantic segmentation (e.g., standard U-Net) merges adjacent boundaries into a single blob mask.
* **Discrete Quantification**: Instance segmentation delivers individual masks, bounding boxes, and calibrated confidence scores per vessel profile, allowing morphological metrics (individual luminal area, diameter, and count distribution) to be derived accurately.

---

## ✨ Key Features

- **Advanced Mask R-CNN Backbone**: TorchVision's `maskrcnn_resnet50_fpn_v2` with deep residual bottlenecks and Feature Pyramid Networks (FPN v2) matching scales from tiny $5\text{ px}$ capillary cross-sections to larger venules.
- **Whole-Slide Tiling & Greedy Stitching**: Seamlessly processes standard $512 \times 512$ tiles up to high-resolution Whole-Slide biopsy sections ($6000 \times 6000\text{ px}$) using boundary-aware IoU cross-tile deduplication.
- **Domain-Specific Stain Augmentation**:
  - **HED Optical Density Jitter**: Deconvolves RGB signals into Hematoxylin, Eosin, and PAS dye concentrations to simulate inter-lab chemical staining variability.
  - **Unlabelled LAB Color Transfer**: Reinhard-style transfer sampling from an unlabelled bank of 400 tiles (expanding $(a^*, b^*)$ spread from $4.98$ to $7.10$).
  - **Tissue Masking**: Excludes white glass slide background ($L^* < 225$) to prevent unnatural artifact generation.
- **Two-Stage Optimization**: Joint pre-training on sparse annotations (Dataset 2) followed by fine-tuning on expert-reviewed clean masks (Dataset 1).
- **Inference Consensus Engine**: 4-checkpoint ensemble (Folds 0 & 1, Seeds 0 & 1) with 3-view Flip TTA (12 forward passes per tile) fused via mask clustering ($\text{IoU} \ge 0.55$).
- **Zero-Latency Streamlit Dashboard**: Instant interactive post-filtering of detection confidence and mask thresholds without re-executing inference passes.
- **Multi-Format Export Suite**:
  - Semi-transparent overlay visualizations with highlighted contours.
  - Pure binary union mask ($0/255$).
  - 16-bit integer label mask (unique ID per vessel).
  - Tabular vessel morphology report (bounding boxes, centroids, and pixel areas in CSV format).

---

## 🏗️ System Architecture

```
                    [ 512x512 PAS / H&E Histology Tile ]
                                      │
            (Model Internal Resolution Rescaled to 1280x1280)
                                      │
                                      ▼
                        [ ResNet-50 Backbone (C2-C5) ]
                         Pretrained on COCO / Deep Residuals
                                      │
                                      ▼
                        [ Feature Pyramid Network (FPN v2) ]
                         Multi-Scale Fusion (P2 to P6, 256-d)
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
[ Region Proposal Network (RPN) ]                    [ High-Resolution Features ]
Proposes Candidate Bounding Boxes                              │
            │                                                  │
            └─────────────────────────┬────────────────────────┘
                                      ▼
                                [ RoIAlign ]
                     Sub-pixel Bilinear Interpolation
                     (7x7 for Box Head, 14x14 for Mask Head)
                                      │
            ┌─────────────────────────┴────────────────────────┐
            ▼                                                  ▼
   [ Fast R-CNN Box Head ]                           [ FCN Mask Head ]
   • Classification (Vessel vs Bg)                   • 4x Conv (256-d) + Deconv (2x)
   • Bounding Box Coordinate Offsets                 • 28x28 Pixel-level Binary Mask
                                      │
                                      ▼
                        [ Multi-Task Loss Optimization ]
          L_total = L_RPN-cls + L_RPN-box + L_RoI-cls + L_RoI-box + 2 * L_mask
```

> **Why the $2 \times \mathcal{L}_{mask}$ multiplier?**
> Standard detectors give equal weighting to classification, box regression, and mask branches. In microvascular segmentation, exact luminal boundary delimitation is the primary diagnostic priority; doubling the mask loss gradient forces the network to prioritize boundary precision.

---

## 🎨 Pathology-Specific Augmentations

| Augmentation Technique | Parameters & Probabilities | Biological Motivation |
|---|---|---|
| **HED Stain Perturbation** | $p = 0.8, \sigma = 0.2$ | Uses Ruifrok & Johnston color deconvolution to decompose RGB into optical density (OD) stain channels, perturbing dye concentrations to simulate real-world staining variations. |
| **LAB Color Transfer** | $p = 0.5, \text{blend } [0.2, 0.8]$ | Samples tissue luminance and chromatic distributions from 400 unlabelled tiles (Dataset 3), adapting the model against scanner and lighting shifts. |
| **Tissue Masking** | $L^* < 225$ | Restricts color perturbations strictly to cellular tissue, keeping slide glass transparent and artifact-free. |
| **HSV & Contrast Jitter** | $p = 0.6, \text{Hue } \pm 4$ | Accommodates minor light dispersion and exposure differences. |
| **Gaussian Blur** | $p = 0.2, \sigma \in [0.5, 1.5]$ | Simulates optical focus inconsistencies across different microscope objectives. |
| **Geometric Transforms** | $p = 0.5 \text{ (flips/90° rot)}, p = 0.7 \text{ (affine)}$ | Neutralizes spatial and rotational bias since microvessels have no intrinsic orientation in tissue sections. |

---

## 🔁 Two-Stage Training & Validation Protocol

```
Stage 1: Pre-training (16 Epochs, LR = 1e-4)
┌──────────────────────────────────────────────────────────────┐
│ Datasets 1 & 2 Combined (~1,150 tiles outside held-out slide) │
│ Learns broad vascular morphology from noisy/sparse labels     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
Stage 2: Clean Fine-Tuning (≤ 6 Epochs, LR = 2e-5)
┌──────────────────────────────────────────────────────────────┐
│ Dataset 1 Exclusively (Expert-reviewed labels only)           │
│ Sanitizes boundaries and eliminates false-negative bias      │
└──────────────────────────────────────────────────────────────┘
```

* **WSI-Held-Out Splitting**: To avoid intra-slide spatial leakage, all tiles from complete Whole-Slide Images are held out exclusively for validation.
* **Exponential Moving Average (EMA)**: Running model parameters are smoothed ($d \approx 0.998$) after each gradient step, preventing validation metric oscillations.

---

## 📊 Experimental Benchmarks & Progression

### Iterative Score Progression Matrix

| Version | Architecture & Strategy | CV mAP | Key Architectural Leap |
|---|---|:---:|---|
| **v1 / v2** | Custom Mask R-CNN (Random init) | `0.1040` | Baseline from scratch on native patches |
| **v3** | Frozen DINOv2 ViT-S/14 + Custom FPN | `0.2810` | Foundation model feature extraction |
| **v5** | ResNet-50-FPN v2 + ViT-Adapter | `0.3780` | COCO pretraining + WSI split protocol |
| **v1-inf** | Hybrid ResNet-50-FPN v2 + Halo Stitching | `0.4280` | Low-confidence recall extraction |
| **Final** | **ResNet-50-FPN v2 (4-Model Ensemble + Stain Aug + 2-Stage + TTA)** | **`0.5300`** | **Complete domain-adapted clinical pipeline** |

### Held-Out Whole-Slide Validation Metrics

| Metric | Fold 0 (Held-Out WSI A) | Fold 1 (Held-Out WSI B) | Arithmetic Mean |
|---|:---:|:---:|:---:|
| **Validation Tiles** | 270 | 152 | 211 |
| **AP@0.50** | `0.5590` | `0.5249` | **`0.5420`** |
| **AP@0.60 (Primary)** | `0.4672` | `0.3906` | **`0.4289`** |
| **AP@0.75 (Strict)** | `0.2777` | `0.0995` | **`0.1886`** |
| **mAP@[0.50:0.95]** | `0.2896` | `0.2008` | **`0.2452`** |
| **Best F1-Score** | `0.5028` (score $\ge 0.69$) | `0.4789` (score $\ge 0.30$) | **`0.4909`** |

---

## 🖥️ Interactive Dashboard

The dashboard provides an intuitive interface for pathologists, researchers, and developers:

```
[ Upload Histology Image / Select Built-in Sample ]
                       │
                       ▼
             [ Model Inference Mode ]
      Fast (1 Model) | Balanced (TTA) | Accurate (Ensemble)
                       │
                       ▼
           [ Real-Time Zero-Latency Tuning ]
      Confidence Slider (0.05 - 0.95) | Mask Threshold Slider
                       │
                       ▼
             [ Synchronized Visualizations ]
  Side-by-Side: Original | Detection Overlay | Binary Mask | Label Map
                       │
                       ▼
             [ Morphology Export ]
      PNG Masks | 16-bit Label TIFF | Tabular Morphology CSV
```

### Dashboard Inference Modes

| Mode | Active Models | TTA Views / Tile | GPU Speed | Best Use Case |
|---|---|---|---|---|
| **Fast** | 1 Checkpoint | 1 (Identity) | $\sim 0.5\text{ s / tile}$ | Rapid interactive screening & threshold exploration |
| **Balanced** | 1 Checkpoint | 3 (`id`, `h-flip`, `v-flip`) | $\sim 1.5\text{ s / tile}$ | Standard quality / latency trade-off |
| **Accurate** | 4 Checkpoints | 12 passes ($4 \times 3\text{ views}$) | $\sim 6.0\text{ s / tile}$ | Maximum recall, high boundary fidelity & consensus |

---

## 📂 Project Directory Structure

```
kidney-vessel-segmentation/
├── app.py                      # Interactive Streamlit dashboard
├── core/
│   ├── __init__.py
│   ├── config.py               # Hyperparameters, paths, tiling strides, thresholds
│   ├── model.py                # TorchVision Mask R-CNN architecture & weights loader
│   ├── tta.py                  # Multi-view Test-Time Augmentation (TTA) engine
│   ├── merge.py                # Cross-view & multi-model clustering (IoU ≥ 0.55)
│   ├── tiling.py               # Large image tiling, padding, and deduplication
│   ├── pipeline.py             # Full inference pipeline with CPU/GPU OOM protection
│   ├── imageio.py              # Robust image I/O (TIFF, PNG, JPG, 16-bit)
│   └── viz.py                  # Blending overlays, contour tracing, and exports
├── weights/                    # Checkpoint repository for fold*_best.pth weights
├── samples/                    # Curated sample HuBMAP kidney tiles (TIFF)
├── tests/
│   └── smoke_test.py           # Unit tests (weights, forward pass, label masks)
├── reference/                  # Training notebooks and reference workflows
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (weights, cache, logs)
└── README.md                   # Project documentation
```

---

##  Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Santosh-kumar-sah/kidney-vessel-segmentation.git
cd kidney-vessel-segmentation
```

### 2. Set Up a Virtual Environment
```bash
# Python 3.10 to 3.13 supported
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# Activate on macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Model Weights
Place trained `.pth` model checkpoints in the `weights/` directory:
```
weights/
├── fold0_best.pth
├── fold1_best.pth
├── fold10_best.pth   (Seed 0 full data)
└── fold11_best.pth   (Seed 1 full data)
```
> *If running locally for testing without pretrained checkpoints, initialize a test model state:*
> ```bash
> python -c "import torch, os; from core.model import build_model; os.makedirs('weights', exist_ok=True); torch.save(build_model().state_dict(), 'weights/fold0_best.pth')"
> ```

### 5. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

---

## 🧪 Testing & Quality Assurance

Run the automated test suite to verify weight loading, forward-pass execution, coordinate transformations, and label map consistency:

```bash
python -m unittest tests/smoke_test.py
```

---

## ⚠️ Known Limitations & Disclaimer

1. **Target Specificity**: The model is tuned exclusively for vascular structures (capillaries, arterioles, venules). Non-vascular compartments (tubules, glomeruli, urinary space) are treated as background.
2. **Staining and Resolution**: Optimized for PAS and H&E stained sections at $\approx 20\times / 40\times$ equivalent resolution. Drastic scale divergence or uncalibrated stains may require threshold tuning.
3. **Research Notice**: **This software is for research, educational, and development purposes only.** It is not certified as a medical device and should not be used for primary clinical diagnostics or therapeutic decisions.

---

## 👥 Authors & Academic Context

Developed as part of the **Department of Computer Science & Engineering (Artificial Intelligence)** at **Krishna Institute of Engineering & Technology (KIET)**, Ghaziabad, Uttar Pradesh, India, in collaboration with HuBMAP research benchmarks.
