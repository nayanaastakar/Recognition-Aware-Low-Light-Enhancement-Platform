# M.Tech Minor Project Presentation (15 Slides)
## Recognition-Aware Low-Light Image Enhancement for Improved Object Detection

---

## Slide 1: Title Slide
- **Title:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection
- **Subtitle:** An Illumination-Guided Retinex Architecture with Downstream D-FINE-S Evaluation
- **Degree:** Master of Technology (M.Tech) in Computer Science / AI & Computer Vision
- **Candidate:** M.Tech Research Scholar
- **Status:** Architecture Implemented & Tested; Colab GPU Suite Ready

---

## Slide 2: Problem Statement & Motivation
- **The Challenge:** Vision systems (robotics, autonomous navigation, night surveillance) fail in low-light due to photon starvation and high sensor noise.
- **The Core Paradox:** Enhancing brightness for human perception does not always improve machine object recognition.
- **Why?** Classical enhancement often amplifies sensor noise and blurs fine object edges, confusing downstream neural detectors.

---

## Slide 3: Research Questions
1. Does perceptual low-light enhancement reliably improve downstream object detection?
2. Can Retinex decomposition isolate noise and preserve structural boundaries?
3. What are the specific performance trade-offs between PSNR/SSIM, detection mAP@50, and computational runtime?

---

## Slide 4: Literature Review & Benchmarks
- **Zero-DCE (CVPR 2020):** Deep curve estimation; fast non-reference enhancement, but amplifies background noise in extreme dark areas.
- **Retinexformer (ICCV 2023):** Transformer-based illumination-guided self-attention for paired enhancement.
- **D-FINE-S (2024):** State-of-the-art fine-grained bounding box regression detector.
- **Datasets:** LOL (500 paired images) & ExDark (7,363 low-light images across 12 object categories).

---

## Slide 5: Research Gap & Objectives
- **Identified Gap:** Most low-light enhancement research focuses strictly on full-reference perceptual metrics (PSNR, SSIM) without evaluating downstream task utility (mAP).
- **Objectives:**
  1. Design a recognition-aware Retinex architecture: **IGR-Net**.
  2. Implement an end-to-end evaluation pipeline with **D-FINE-S**.
  3. Validate on LOL and ExDark benchmarks.
  4. Develop an interactive application supporting image upload and live webcam capture.

---

## Slide 6: System Architecture Overview
- **Pipeline:**
  1. Image Acquisition (File Upload / Laptop Webcam).
  2. Retinex Decomposition: $S = R \times L$.
  3. Reflectance Refinement (Noise-Aware Attention + Channel Attention).
  4. Illumination Enhancement (Local-Global Feature Fusion).
  5. Retinex Physical Recombination & Residual Fusion.
  6. D-FINE-S Object Detection & Bounding Box Generation.

---

## Slide 7: Proposed IGR-Net Architecture
- **Retinex Decomposition Head:** Splits features into reflectance ($R_{init}$) and illumination ($L_{init}$).
- **Local-Global Context Module:** Combines 3x3 local convolutions with dilated branches ($d=2$) and global pooling to capture broad lighting context.
- **Noise-Aware Attention:** Dynamic spatial attention guided by dark regions: $\mathcal{A}_{noise} = (1 - L_{init}) \odot \sigma(\text{Conv}(F_R))$, selectively suppressing noise.
- **Channel Attention:** Squeeze-and-Excitation block preserving natural RGB chromatic balance.
- **Verified Complexity:** **62,060 trainable parameters** (~62.1K), ideal for real-time edge processing.

---

## Slide 8: Mathematical Formulations & Losses
- **Retinex Consistency Loss:** $\mathcal{L}_{ret} = \| R \odot L_{init} - I_{low} \|_1$
- **Illumination Smoothness Loss:** Weighted by reflectance gradients $\exp(-\alpha |\nabla R|)$ to preserve structural edges.
- **Exposure Control Loss:** Drives average illumination to well-exposed target ($\mu = 0.55$).
- **Color Constancy Loss:** Minimizes chromatic drift using the Grey-World assumption.

---

## Slide 9: Experimental Setup & Datasets
- **Paired LOL Dataset:** 485 training pairs, 15 evaluation pairs (real low/normal light).
- **ExDark Dataset:** 7,363 low-light images across 12 semantic object categories.
- **Detector:** D-FINE-S (2.59M parameters), evaluated at IoU threshold 0.50 and confidence threshold 0.25.
- **Hardware Support:** Production CPU pipeline locally + 10 Google Colab GPU notebooks.

---

## Slide 10: Recognition-Aware Evaluation Matrix
| Experimental Condition | Enhancement Method | Target Dataset | PSNR | SSIM | mAP@50 | Latency (CPU) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Condition A** | Original Low-Light | ExDark Test Split | N/A | N/A | *Pending real-data eval* | 0.0 ms |
| **Condition B** | Zero-DCE Baseline | LOL / ExDark | *Pending* | *Pending* | *Pending real-data eval* | 117.6 ms |
| **Condition C** | Retinexformer | LOL / ExDark | *Pending* | *Pending* | *Pending real-data eval* | 180.7 ms |
| **Condition D** | **IGR-Net (Proposed)** | LOL / ExDark | *Pending* | *Pending* | *Pending real-data eval* | 165.6 ms |

*Note: In accordance with scientific integrity, full-scale test split metrics are designated as pending GPU execution on Google Colab.*

---

## Slide 11: Verified Architectural Parameters
- **D-FINE-S Detector:** 2,596,340 parameters (~2.60M).
- **Zero-DCE Baseline:** 79,416 parameters (~79.4K).
- **IGR-Net (Proposed):** 62,060 parameters (~62.1K).
- **Retinexformer Benchmark:** 28,708 parameters (~28.7K).
- *IGR-Net provides a lightweight footprint suited for robotic and vehicle edge compute.*

---

## Slide 12: IGR-Net Ablation Architecture
| Configuration | Disabled Components | Trainable Parameters | Status |
| :--- | :--- | :---: | :---: |
| Base Retinex Network | Local-Global, Noise & Channel Attn | **34,219** | Verified & Ready |
| w/o Local-Global Context | Multi-scale & dilated context | **39,244** | Verified & Ready |
| w/o Noise-Aware Attention | Dark-region spatial attenuation | **57,291** | Verified & Ready |
| w/o Channel Attention | Channel SE gating | **61,804** | Verified & Ready |
| **Full IGR-Net (Proposed)** | None (All modules active) | **62,060** | Verified & Ready |

---

## Slide 13: Interactive Application Features
- **Acquisition:** File upload (JPG, PNG) & live laptop webcam capture.
- **Model Switching:** Instant comparative testing across Zero-DCE, IGR-Net, Retinexformer, Classical Retinex.
- **Detection Dashboard:** D-FINE-S visualization with bounding box coordinates, confidences, and labels.
- **Batch Processing:** Multi-image queue with progress indicators and CSV export.

---

## Slide 14: Conclusion & Verified Deliverables
- Complete Python software architecture implemented and unit-tested (8/8 unit tests passing).
- All 4 application workflows verified: Upload, Webcam, 4-Way Comparison, and Batch Processing.
- 10 turnkey Google Colab notebooks configured for GPU training and dataset synchronization.
- Transparent reporting: synthetic training and sample testing clearly distinguished from full-scale dataset evaluations.

---

## Slide 15: Future Scope & References
- **Future Directions:** End-to-end joint training with perceptual detection loss, INT8 quantization for edge devices (Jetson), and temporal consistency for video streams.
- **Key References:**
  - Li et al., Zero-DCE, *IEEE TPAMI*, 2021.
  - Cai et al., Retinexformer, *ICCV*, 2023.
  - Zhao et al., D-FINE, *arXiv:2410.13842*, 2024.
  - Loh & Chan, ExDark Dataset, *CVIU*, 2019.
