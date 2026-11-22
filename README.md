# Recognition-Aware Low-Light Image Enhancement for Improved Object Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![D-FINE-S](https://img.shields.io/badge/Detector-D--FINE--S%20(10.32M)-green.svg)](https://github.com/Peterande/D-FINE)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **M.Tech Minor Project — System Architecture, Implementation & Technical Verification Suite**  
> **Platform Name:** SENTINEX-CV Research Suite  
> **Repository:** [https://github.com/nayanaastakar/Recognition-Aware-Low-Light-Enhancement-Platform](https://github.com/nayanaastakar/Recognition-Aware-Low-Light-Enhancement-Platform)

---

## 📖 Executive Summary

Low-light surveillance, robotics, and autonomous driving scenarios suffer from severe photon noise, severe contrast degradation, and non-uniform illumination. These optical deficiencies disrupt feature extraction in deep neural networks, causing object detectors to fail catastrophically.

This platform implements and evaluates a **Recognition-Aware Low-Light Enhancement Framework**, coupling physical Retinex decomposition with downstream object detection. Rather than optimizing exclusively for human visual aesthetics (e.g., PSNR/SSIM), this project studies **recognition-aware enhancement** designed to preserve semantic edges and maximize object detection performance using the state-of-the-art **D-FINE-S (Distribution Refinement DETR)** detector.

---

## 🏛️ System Architecture

```
                                      +---------------------------------------------+
                                      |            Input Low-Light Image            |
                                      +---------------------------------------------+
                                                             |
                      +--------------------------------------+--------------------------------------+
                      |                                      |                                      |
                      v                                      v                                      v
       [Condition B: Zero-DCE]                [Condition C: IGR-Net]                 [Condition D: Retinexformer]
       • 7-Layer DCE-Net                      • Proposed Physical Retinex            • Adapted IG-MSA Attention
       • Non-Reference Quadratic Curves       • Illumination-Guided Reflectance      • Illumination Guided Transformer
       • 79,416 Parameters                    • 62,060 Parameters                    • 28,708 Parameters
                      |                                      |                                      |
                      +--------------------------------------+--------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |     Unified Downstream Object Detector      |
                                      |   D-FINE-S (Official Pretrained MS-COCO)    |
                                      |   • HGNetv2 B0 Backbone + HybridEncoder     |
                                      |   • Fine-Grained Distribution Refinement    |
                                      |   • 10,321,877 parameters (Strict Checkpoint|
                                      +---------------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |  Rigorous Ground-Truth Benchmark Evaluation |
                                      |  • Class-Aware IoU Matching Engine          |
                                      |  • mAP@0.5, Precision, Recall, F1-Score     |
                                      |  • ExDark / MS-COCO / Pascal VOC Annotations|
                                      +---------------------------------------------+
```

---

## 🔬 Model Architectures & Parameter Summary

| Condition | Method | Core Mechanism | Enhancer Params | Detector Params | Execution Device |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | **Original (Baseline)** | Unprocessed Raw Low-Light RGB | **0** (Raw) | 10,321,877 (D-FINE-S) | CPU / CUDA |
| **B** | **Zero-DCE (CVPR 2020)** | 7-Layer CNN Estimating Quadratic Curves | **79,416** | 10,321,877 (D-FINE-S) | CPU / CUDA |
| **C** | **IGR-Net (Proposed)** | Physical Retinex Decomposition + Attention | **62,060** | 10,321,877 (D-FINE-S) | CPU / CUDA |
| **D** | **Retinexformer (Adapted)** | Illumination-Guided Multi-Head Self-Attention | **28,708** | 10,321,877 (D-FINE-S) | CPU / CUDA |

---

## 📐 Mathematical Formulation

### 1. Classical Retinex Decomposition
Given a degraded low-light image $S(x, y)$, Retinex theory decomposes the signal into illumination $I(x, y)$ and reflectance $R(x, y)$:
$$S(x, y) = R(x, y) \odot I(x, y)$$
where $R$ represents invariant object surface reflectance and $I$ represents non-uniform environmental illumination.

### 2. Zero-DCE Curve Estimation
Zero-DCE formulates enhancement via recursive light curves:
$$LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n(x) \cdot LE_{n-1}(x) \cdot (1 - LE_{n-1}(x))$$
where $\mathcal{A}_n$ denotes pixel-wise parameter maps learned without reference images.

### 3. Object Detection Evaluation Metrics
Given predicted boxes $B_p$ and ground-truth boxes $B_{gt}$, spatial Intersection over Union is computed:
$$\text{IoU}(B_p, B_{gt}) = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
A detection is classified as a True Positive (TP) if $\text{IoU} \ge \tau_{\text{IoU}}$ (default $0.50$) and predicted class matches ground-truth class. Mean Average Precision ($\text{mAP@50}$) is computed across recall thresholds:
$$\text{AP} = \int_{0}^{1} p(r) \, dr, \quad \text{mAP} = \frac{1}{N_{cls}} \sum_{c=1}^{N_{cls}} \text{AP}_c$$

---

## 📁 Repository Directory Structure

```
Recognition-Aware-Low-Light-Enhancement-Platform/
├── app/
│   └── app.py                      # Main SENTINEX-CV Streamlit Dashboard
├── docs/
│   └── screenshots/                # 11 Verified Application Screenshots
│       ├── 01_dashboard_overview.png
│       ├── 02_pipeline_architecture.png
│       ├── 03_four_way_comparison.png
│       ├── 04_analytics_detection_latency.png
│       ├── 05_analytics_luminance_contrast.png
│       ├── 06_analytics_transparency_matrix.png
│       ├── 07_benchmark_verification_overview.png
│       ├── 08_exdark_benchmark_results.png
│       ├── 09_detected_objects_inspection.png
│       ├── 10_scene_00026_four_way_comparison.png
│       └── 11_scene_00026_benchmark_results.png
├── evaluation/
│   ├── detailed_evaluator.py       # Class-aware IoU matching engine & mAP calculator
│   └── exdark_parser.py            # Multi-format annotation parser (TXT, JSON, XML)
├── inference/
│   ├── detection.py                # D-FINE-S Detection Engine
│   └── enhancement.py              # Zero-DCE, IGR-Net, Retinexformer Enhancement Engine
├── models/
│   ├── dfine/                      # D-FINE-S architecture & postprocessors
│   ├── igrnet/                     # Proposed IGR-Net architecture
│   ├── retinexformer/              # Adapted Retinexformer architecture
│   ├── zerodce/                    # Zero-DCE architecture
│   └── transforms.py               # Preprocessing & normalization pipelines
├── weights/
│   ├── dfine_s_coco.pth            # Pretrained D-FINE-S weights (10.32M parameters)
│   ├── igrnet_best.pth             # IGR-Net trained weights
│   ├── retinexformer_best.pth      # Retinexformer trained weights
│   └── zerodce_best.pth            # Zero-DCE trained weights
├── data/
│   └── samples/low/                # Preloaded low-light test images and ground truth
├── scripts/
│   ├── capture_all_screenshots.py  # Automated browser capture automation suite
│   └── package_project.py          # Project packaging & validation script
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Comprehensive documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/nayanaastakar/Recognition-Aware-Low-Light-Enhancement-Platform.git
cd Recognition-Aware-Low-Light-Enhancement-Platform
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Dashboard
```bash
streamlit run app/app.py
```
Open your browser at **`http://localhost:8501`** to access the research suite.

---

## ⚖️ Research Integrity Standards & Limitations Disclosure

1. **Genuine Model Inferences:** All four image enhancement conditions and the downstream D-FINE-S object detector execute genuine forward passes. No simulated bounding boxes or mock latencies are generated.
2. **Deterministic Evaluation:** Single-image benchmark calculations strictly require matching ground-truth annotation coordinates. Dataset-wide mAP claims require running the complete ExDark validation split (7,363 images).
3. **Paired LOL Benchmark:** PSNR and SSIM calculations require high-exposure paired reference photographs (e.g. LOL dataset `eval15/high`) and remain labeled as "Pending" until valid ground-truth pairs are supplied.

---

## 📜 Citation & License

This project is released under the [MIT License](LICENSE).

```bibtex
@misc{nayana2026recognition,
  title={Recognition-Aware Low-Light Image Enhancement for Improved Object Detection},
  author={Nayana Astakar},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/nayanaastakar/Recognition-Aware-Low-Light-Enhancement-Platform}}
}
```
