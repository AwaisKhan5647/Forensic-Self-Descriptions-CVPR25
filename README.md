# Forensic Self-Descriptions (FSD): Zero-Shot AI-Generated Image Detection

[![CVPR 2025](https://img.shields.io/badge/CVPR-2025-blue)](https://cvpr.thecvf.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.21003-b31b1b)](https://arxiv.org/abs/2503.21003)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

> **This repository contains an independent evaluation and deployment of the FSD method.**
> Original implementation and pre-trained weights by the paper authors:
> **[ductai199x/Forensic-Self-Descriptions-CVPR25](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25)**

---

**Zero-shot AI-generated image detection — trained only on real images, generalizes to any unseen generator.**

Implementation of **"Forensic Self-Descriptions Are All You Need for Zero-Shot Detection, Open-Set Source Attribution, and Clustering of AI-generated Images"** (CVPR 2025).

> **TL;DR:** FSD detects AI-generated images with **96.0% average AUC** across 24 generators (Stable Diffusion, Midjourney, DALL-E, StyleGAN, etc.) while trained **exclusively on real photographs** — no synthetic training data required.

<p align="center">
  <img src="assets/teaser.jpg" width="100%">
</p>

<p align="center">
  <img src="assets/system_diagram.jpg" width="100%">
</p>

## What this repo adds

- **Multi-GPU evaluation pipeline** (`evaluate_datasets.py`, `evaluate_queue.py`) for large-scale batch inference across multiple datasets in parallel
- **Resume-safe execution** — survives crashes and restarts from the last completed file
- **High-resolution image handling** — pre-resize guard for 50 MP+ images that would otherwise OOM on GPU
- **Watchdog** (`watchdog.sh`) for unattended long-running jobs
- **System inference documentation** (`system_inference.txt`) — step-by-step pipeline breakdown

## Overview

FSD is a self-supervised forensic method that detects AI-generated images without needing to train on any specific generator. It works by:

1. **Forensic Residual Extraction (FRE)**: Constrained prediction-error filters extract pixel-level forensic residuals
2. **Multi-scale FSD computation**: Residuals are analyzed across scales to produce a compact 960-dimensional forensic descriptor
3. **GMM scoring**: A Gaussian Mixture Model scores each descriptor, yielding a z-score where more negative values indicate AI-generated content

## Results

| Method | Training Data | COCO17 | IN-1k | IN-22k | MIDB | Average |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| CNNDet | Real + Synthetic | 0.756 | 0.714 | 0.733 | 0.683 | 0.722 |
| PatchFor | Real + Synthetic | 0.833 | 0.823 | 0.845 | 0.790 | 0.823 |
| UFD | Real + Synthetic | 0.903 | 0.862 | 0.815 | 0.612 | 0.798 |
| LGrad | Real + Synthetic | 0.819 | 0.770 | 0.866 | 0.824 | 0.820 |
| DE-FAKE | Real + Synthetic | 0.765 | 0.749 | 0.617 | 0.791 | 0.731 |
| Aeroblade | Training-Free | 0.728 | 0.741 | 0.582 | 0.646 | 0.674 |
| ZED | Real Only | 0.751 | 0.676 | 0.716 | 0.747 | 0.723 |
| NPR | Real + Synthetic | 0.945 | 0.900 | 0.900 | 0.957 | 0.926 |
| FSD (paper) | Real Only | **0.968** | **0.962** | **0.941** | **0.971** | **0.960** |

See [the paper](https://arxiv.org/abs/2503.21003) for full results on source attribution and clustering.

## Installation

```bash
git clone https://github.com/AwaisKhan5647/Forensic-Self-Descriptions-CVPR25.git
cd Forensic-Self-Descriptions-CVPR25

# Option A — uv (recommended)
uv sync
source .venv/bin/activate

# Option B — pip
pip install -e .
```

Pre-trained weights are downloaded automatically on first use to `~/.cache/fsd/`.

## Quick Start

### Python API

```python
from fsd import FSDDetector

detector = FSDDetector.load()
result = detector.score("photo.jpg")

print(result.z_score)   # negative = likely AI-generated
print(result.is_fake)   # True/False at default threshold -2.0
```

Score multiple images:
```python
results = detector.score_batch(["img1.jpg", "img2.png", "img3.webp"])
for path, result in zip(paths, results):
    print(f"{path}: z={result.z_score:.4f} {'FAKE' if result.is_fake else 'REAL'}")
```

### Source Attribution

```python
detector = FSDDetector.load(attribution=True)
result = detector.attribute("suspicious_image.jpg")

print(result.source)      # e.g., "Stable Diffusion XL"
print(result.confidence)  # e.g., 0.95
```

Supported sources: DALL-E 3, Stable Diffusion 1.5/3/XL, Midjourney v6, Adobe Firefly, StyleGAN2/3, ProGAN, GigaGAN, and more.

### Command Line

```bash
fsd-score photo.jpg
fsd-score --dir path/to/images/ --csv > results.csv
fsd-score photo.jpg --attribute
fsd-score photo.jpg --threshold -3.0 --device cuda
```

### Multi-GPU Batch Evaluation

```bash
# AIGI-TEST, image_eval24, ReWIND — runs on GPU groups 5,6,7 and 2,3,4 in parallel
python evaluate_datasets.py   # GPUs 5,6,7  →  AIGI_TEST
python evaluate_queue.py      # GPUs 2,3,4  →  image_eval24 + ReWIND

# Watchdog (auto-restart on crash)
bash watchdog.sh &
```

Results are written to `Results/<dataset>/predictions.csv` with columns
`file_path`, `probability`, `predicted_label`, `ground_truth`.

### Gradio Demo

```bash
uv run demo.py          # local
uv run demo.py --share  # public link
```

## Interpreting Results

| z-score | Interpretation |
|---------|----------------|
| z > −2  | Likely real |
| z < −2  | Likely AI-generated (default threshold) |
| z < −3  | High-confidence AI-generated |

## Pre-trained Weights

Auto-downloaded from the [original repository's releases](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/releases) on first use.

| File | Size | Purpose |
|------|------|---------|
| `fre.pt` | ~10 KB | Forensic Residual Extractor |
| `gmm.pt` | ~15 MB | Gaussian Mixture Model (K=5) |
| `fsd_transforms.pt` | ~40 MB | Learned whitening transforms |
| `attribution_transforms.pt` | ~26 MB | Source attribution transforms |
| `source_gmms.pt` | ~207 MB | Per-source GMMs (14 generators) |

## Citation

If you use this work, please cite the original paper:

```bibtex
@InProceedings{Nguyen_2025_CVPR,
    author    = {Nguyen, Tai D. and Azizpour, Aref and Stamm, Matthew C.},
    title     = {Forensic Self-Descriptions Are All You Need for Zero-Shot Detection,
                 Open-Set Source Attribution, and Clustering of AI-generated Images},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
                 and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {3040-3050}
}
```

## License

CC BY-NC-SA 4.0 — research use only, no commercial use, share-alike.
See [LICENSE](LICENSE) for details.
