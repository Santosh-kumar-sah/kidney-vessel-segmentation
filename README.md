# 🔬 Kidney Blood Vessel Segmentation Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58%2B-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive deep learning web application for **instance segmentation of microvasculature** (blood vessels) in histological kidney tissue sections (H&E stain, HuBMAP style).

---

## ✨ Features

- **Deep Learning Core**: Powered by torchvision's `maskrcnn_resnet50_fpn_v2` with specialized heads for binary vessel detection.
- **Whole-Slide Tiling & Stitching**: Handles any image size from small biopsy sections ($300 \times 300$ px) to large tissue regions ($6000 \times 6000$ px) via overlapping $512 \times 512$ tiles and greedy boundary-aware IoU cross-tile deduplication.
- **Test-Time Augmentation (TTA)**: Multi-view flip transforms (`id`, `horizontal`, `vertical`) and multi-model ensemble merging.
- **Zero-Latency Score Filtering**: Post-filters detection confidence in real time—slide the threshold slider and see results instantly without re-running the neural network.
- **Comprehensive Exports**:
  - 🖼️ **Overlay Image (PNG)**: Semi-transparent masks with 2 px contours and optional ID/score tags.
  - ⬛ **Binary Mask (PNG)**: Pure black & white union mask ($0$ / $255$).
  - 🏷️ **16-Bit Label Mask (PNG)**: Each pixel corresponds to unique instance IDs ($0$ = background).
  - 📊 **Metrics Table (CSV)**: Exact bounding boxes, centroid coordinates, and pixel areas.
- **Built-in Sample Tile**: Click **"Use Sample Image"** to test immediately without needing an external dataset.

---

##  Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Santosh-kumar-sah/kidney-vessel-segmentation.git
cd kidney-vessel-segmentation
```

### 2. Set Up a Virtual Environment
```bash
# Python 3.10, 3.11, 3.12, or 3.13 recommended
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell / Command Prompt):
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Weights Setup

Place your trained PyTorch model checkpoints (`fold*_best.pth`) into the `weights/` folder:

```
kidney-vessel-segmentation/
└── weights/
    ├── fold0_best.pth
    ├── fold1_best.pth
    ├── fold10_best.pth   # optional additional fold
    └── fold11_best.pth   # optional additional fold
```

> **Note**: Model weights are excluded from Git commits because each `.pth` checkpoint is ~183 MB (exceeding GitHub's 100 MB file limit).
> If you need to generate a lightweight testing checkpoint for unit testing, run:
> ```bash
> python -c "import torch, os; from core.model import build_model; os.makedirs('weights', exist_ok=True); torch.save(build_model().state_dict(), 'weights/fold0_best.pth')"
> ```

---

## 💻 Running the Dashboard

Start the application with Streamlit:

```bash
streamlit run app.py
```

Then open your browser at **`http://localhost:8501`**.

---

## ⚙️ Inference Modes & Benchmarks

| Mode | Models Used | TTA Views / Tile | GPU Latency | CPU Latency | Best Use Case |
|---|---|---|---|---|---|
| **Fast** | 1 model | 1 (Identity) | ~0.5s / tile | ~15s / tile | Fast exploration and parameter tuning |
| **Balanced** | 1 model | 3 (`id`, `h-flip`, `v-flip`) | ~1.5s / tile | ~40s / tile | Balanced accuracy / speed tradeoff |
| **Accurate** | All $N$ models | $3 \times N$ views | ~1.5s $\times N$ | ~40s $\times N$ | Maximum recall and precision |

### How to Tune Thresholds for Best Results
- **Score Threshold Slider (Sidebar)**:
  - Default: `0.30` (balanced precision and recall).
  - Lower to `0.05` – `0.15` if vessels are faint or for synthetic demonstration tiles.
  - Raise to `0.45` – `0.60` to eliminate false positives and keep only high-confidence vessels.
- **Mask Threshold Slider (Sidebar)**:
  - Default: `0.40` (validated optimum on held-out slides).

---

## 🧪 Running Automated Tests

Run the included test suite to verify weight loading, pink noise inference, tiling coverage, and label mask synchronization:

```bash
python -m unittest tests/smoke_test.py
```

---

## 📂 Project Architecture

```
kidney-vessel-segmentation/
├── app.py                  # Streamlit web dashboard
├── core/
│   ├── __init__.py
│   ├── config.py           # Configuration constants (sizes, strides, thresholds)
│   ├── model.py            # Model architecture builder and weights loader
│   ├── tta.py              # Test-Time Augmentation (TTA) transforms
│   ├── merge.py            # Cross-view and cross-model prediction clustering
│   ├── tiling.py           # Whole-slide tiling and greedy IoU deduplication
│   ├── pipeline.py         # End-to-end inference engine with OOM protection
│   ├── imageio.py          # Multi-format image loader (TIFF, PNG, JPG, 16-bit)
│   └── viz.py              # Mask blending, contours, label masks, and exports
├── weights/                # Directory for fold*_best.pth model weights
├── samples/                # Sample histology images
├── tests/
│   └── smoke_test.py       # Automated smoke and unit tests
├── reference/
│   └── fold1-kidney_1.ipynb# Training and model reference notebook
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignores large model weights and cache
└── README.md
```

---

## ⚠️ Known Limitations & Clinical Disclaimer

1. **Blood Vessels Only**: The model is trained specifically for vascular segmentation. It does not segment glomeruli, tubuli, or interstitial infiltrates.
2. **Histology & Staining Sensitivity**: Performance is optimized for H&E staining at ~20x/40x equivalent magnification matching the HuBMAP competition dataset. Uncalibrated stains or divergent resolutions may degrade predictions.
3. **Expected Quality**: On held-out slides, expected AP@0.6 is ~0.39 with precision ~0.53 and recall ~0.44 at score 0.3.
4. **Clinical Disclaimer**: **For research and demonstration purposes only.** Not intended for clinical diagnostic use or treatment planning.
