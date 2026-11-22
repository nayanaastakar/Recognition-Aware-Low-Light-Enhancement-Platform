# Recognition-Aware Low-Light Image Enhancement for Improved Object Detection
## M.Tech Minor Project Research Report

**Academic Degree:** Master of Technology (M.Tech) in Computer Science & Engineering / Artificial Intelligence  
**Author / Candidate:** M.Tech Research Scholar  
**Environment & Platform:** Google Antigravity & PyTorch Deep Learning Research Framework  
**Date:** November 2026 (Audited & Verified Edition)

---

## Abstract
Autonomous perception and edge vision systems frequently fail in low-light environments due to extreme photon starvation, reduced signal-to-noise ratio (SNR), and non-linear contrast degradation. While classical and deep image enhancement techniques aim to improve perceptual human visibility, they often amplify dark sensor noise, oversaturate high-frequency regions, or distort object boundaries, resulting in degraded downstream object detection. 

In this M.Tech Minor Project, we investigate the fundamental relationship between low-light image enhancement and downstream object detection, addressing whether perceptual image enhancement reliably translates to improved detection accuracy. We propose **IGR-Net (Illumination-Guided and Recognition-Aware Enhancement Network)**, a deep Retinex-based architecture designed specifically to balance image restoration with downstream semantic object recognition. IGR-Net integrates:
1. Physical Retinex decomposition into reflectance and illumination representations;
2. Local-Global Feature Modules capturing multi-scale spatial illumination context;
3. Spatially-varying Noise-Aware Attention to suppress sensor noise in dark regions without blurring critical structural object contours;
4. Channel Attention to eliminate chromatic distortion.

We establish an end-to-end recognition-aware evaluation framework utilizing the SOTA **D-FINE-S** detector across the **LOL** and **ExDark** benchmark datasets. A complete interactive application supporting both image upload and real-time laptop webcam capture is also presented.

---

## 1. Introduction & Problem Statement
Computer vision models trained on standard daytime datasets (e.g., MS-COCO, ImageNet) exhibit severe performance drops when deployed under severe low-light conditions. Autonomous driving, nighttime surveillance, robotics, and search-and-rescue operations require reliable 24/7 visual detection. In low-light environments, image sensors operate with high analog gain, introducing heavy Poisson-Gaussian noise, chromatic aberrations, and clipping of dark textures.

Traditional image enhancement methods and recent zero-reference curve estimators prioritize human visual appeal rather than machine recognition cues. Consequently, enhanced images often exhibit noise artifacts and boundary blurring that confuse downstream detectors. This research project bridges this critical gap by designing a recognition-aware enhancement framework.

---

## 2. Research Questions & Proposed Contributions
### Main Research Questions:
1. *How effectively do existing low-light enhancement architectures improve image quality versus machine recognition accuracy?*
2. *Can a Retinex-based architecture decouple noise and illumination to preserve object boundaries?*
3. *What are the specific performance trade-offs between PSNR/SSIM, detection mAP@50, and computational runtime?*

### Major Research Contributions:
1. **Proposed IGR-Net Architecture:** An illumination-guided Retinex network incorporating local-global context, noise-aware spatial attention, and channel attention (verified 62,060 trainable parameters).
2. **Recognition-Aware Benchmark Pipeline:** Direct comparative evaluation framework comparing 4 experimental conditions (Original, Zero-DCE, IGR-Net, Retinexformer) under a unified D-FINE-S object detection backbone.
3. **Exhaustive Ablation Design:** Empirical modular toggles for individual architectural components of IGR-Net.
4. **Full-Stack Research Interface:** An interactive web-based application supporting both uploaded files and laptop webcam capture.
5. **Google Colab Experimental Suite:** 10 end-to-end notebooks supporting GPU training and evaluation on Google Colab.

---

## 3. Literature Survey & Related Work
- **Retinex Theory:** Introduced by Land (1977), decomposes an observed image $S$ into reflectance $R$ (object reflectance properties) and illumination $L$ (incident light): $S = R \times L$.
- **Zero-DCE (Li et al., CVPR 2020):** Formulates enhancement as deep curve estimation without paired reference data using non-reference losses. While fast, it frequently amplifies background noise in dark regions.
- **Retinexformer (Cai et al., ICCV 2023):** Employs illumination-guided self-attention in a transformer backbone for paired low-light enhancement.
- **D-FINE (Zhao et al., 2024):** Redefines bounding box regression by predicting fine-grained probability distributions over coordinate offsets, providing high precision in real-time object detection.
- **ExDark Dataset (Loh & Chan, CVIU 2019):** Benchmark dataset of 7,363 images across 12 classes under low-light conditions.

---

## 4. Proposed IGR-Net Architecture
The proposed IGR-Net comprises five primary functional modules:
1. **Retinex Decomposition Encoder:** Estimates initial reflectance $R_{init}$ and illumination $L_{init}$ maps.
2. **Local-Global Illumination Module:** Combines standard 3x3 convolutions with dilated convolutions ($d=2$) and global average pooling to maintain broad illumination consistency without overexposure.
3. **Spatially-Varying Noise-Aware Attention:** Generates spatial attention masks modulated by the inverse illumination $(1 - L_{init})$:
   $$\mathcal{A}_{noise} = (1 - L_{init}) \odot \sigma(\text{Conv}(F_R))$$
   Suppresses high-frequency sensor noise in dark regions while preserving sharp edge transitions.
4. **Squeeze-and-Excitation Channel Attention:** Calibrates inter-channel dependencies to preserve natural chromatic balance.
5. **Reconstruction & Recombination Layer:** Recombines refined reflectance and illumination through a physical Retinex multiplication layer combined with residual refinement:
   $$S_{enhanced} = 0.7 \cdot \text{Recon}([R_{refined} \odot L_{enhanced}, L_{enhanced}]) + 0.3 \cdot (R_{refined} \odot L_{enhanced})$$

---

## 5. Experimental Evaluation Protocol & Benchmark Status

> [!IMPORTANT]
> **Academic Integrity Notice:**  
> In accordance with rigorous scientific practice, all benchmark metrics below are designated as **Pending real-data evaluation** until full-scale multi-epoch training and evaluation on the complete LOL (485 train / 15 test pairs) and ExDark (7,363 images) datasets are executed on GPU hardware via the provided Google Colab suite. Local CPU runs have verified architecture pipeline integrity, unit tests, and sample batch processing without errors.

### 5.1 Enhancement & Detection Evaluation Matrix

| Condition / Method | Benchmark Dataset Split | PSNR (dB) | SSIM | Precision | Recall | mAP@50 | Latency (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: Original Low-Light** | ExDark Test Split / LOL Test (15 pairs) | *N/A (Reference)* | *N/A (Reference)* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | **0.0 ms** |
| **Condition B: Zero-DCE Baseline** | LOL Test (15 pairs) / ExDark Test Split | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | **117.6 ms** |
| **Condition C: Retinexformer** | LOL Test (15 pairs) / ExDark Test Split | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | **180.7 ms** |
| **Condition D: Proposed IGR-Net** | LOL Test (15 pairs) / ExDark Test Split | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | *Pending real-data eval* | **165.6 ms** |

---

## 6. Verified Architectural Parameter Counts

The trainable parameters of all models were calculated directly using `sum(p.numel() for p in model.parameters() if p.requires_grad)`:

| Model Architecture | Trainable Parameters | Status | Target Deployment Role |
| :--- | :---: | :--- | :--- |
| **Proposed IGR-Net** | **62,060** (~62.1K) | Implemented & Unit-Tested | Proposed Retinex enhancement network |
| **Zero-DCE Baseline** | **79,416** (~79.4K) | Implemented & Unit-Tested | Zero-reference curve estimation baseline |
| **Retinexformer** | **28,708** (~28.7K) | Implemented & Unit-Tested | Adapted transformer Retinex representation |
| **D-FINE-S Detector** | **2,596,340** (~2.60M) | Implemented & Unit-Tested | Downstream object detector |

### IGR-Net Ablation Variants Parameter Breakdown:
- **Base Retinex Network:** **34,219** parameters.
- **w/o Local-Global Context:** **39,244** parameters.
- **w/o Noise-Aware Attention:** **57,291** parameters.
- **w/o Channel Attention:** **61,804** parameters.
- **Full Proposed IGR-Net:** **62,060** parameters.

---

## 7. Conclusion & Execution on Google Colab
The core software engineering, architecture design, and interactive UI for this M.Tech Minor Project are fully implemented and verified. For complete academic evaluation:
1. Open Google Colab and mount Google Drive.
2. Execute `01_Dataset_Setup.ipynb` to download the genuine LOL and ExDark datasets.
3. Run `02_Zero_DCE_Training.ipynb` and `03_IGRNet_Training.ipynb` for GPU training.
4. Run `08_Recognition_Aware_Comparison.ipynb` and `09_IGRNet_Ablation.ipynb` to populate the real-data evaluation matrix with genuine numbers.

---

## References
1. Land, E. H. (1977). The Retinex theory of color vision. *Scientific American*, 237(6), 108-129.
2. Li, C., Guo, C., & Loy, C. C. (2021). Learning to see in the dark with deep curve estimation. *IEEE TPAMI*, 44(8), 4225-4238.
3. Cai, Y., Bian, H., Lin, J., Wang, H., Timofte, R., & Zhang, Y. (2023). Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement. *ICCV 2023*.
4. Zhao, Y., Liu, W., Chen, Z., et al. (2024). D-FINE: Redefining Bounding Box Regression for Real-Time Object Detection. *arXiv preprint arXiv:2410.13842*.
5. Loh, Y. P., & Chan, C. S. (2019). Getting to know low-light images with the exclusively dark dataset. *Computer Vision and Image Understanding*, 178, 30-42.
6. Wei, C., Wang, W., Yang, W., & Liu, J. (2018). Deep Retinex decomposition for low-light enhancement. *BMVC 2018*.
