# External Repositories, Attribution, and License Register

This project selectively integrates and references architectures, methodologies, and pretrained checkpoints from the following five official research repositories:

---

## 1. D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
- **Official Repository:** [https://github.com/Peterande/D-FINE](https://github.com/Peterande/D-FINE)
- **Paper:** *D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement* (arXiv:2410.13842, 2024)
- **Authors:** Yansong Peng, et al.
- **License:** Apache License 2.0
- **Integrated Components:**
  - `src/nn/backbone/hgnetv2.py`: Hierarchical Golden Neural Network v2 (B0 stage).
  - `src/zoo/dfine/hybrid_encoder.py`: Hybrid intra/cross-scale feature encoder.
  - `src/zoo/dfine/dfine_decoder.py`: Fine-grained distribution refinement transformer decoder.
  - `src/zoo/dfine/postprocessor.py`: Decoded bounding box formatting and score thresholding.
  - `checkpoints/dfine_s_coco.pth`: Official pretrained D-FINE-S model checkpoint (**10,321,877 parameters**, 39.90 MB) loaded with `strict=True`.
- **Role in Project:** Primary downstream object detector evaluating recognition fidelity across all enhancement conditions.

---

## 2. Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
- **Official Repository:** [https://github.com/caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer)
- **Paper:** *Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement* (ICCV 2023)
- **Authors:** Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
- **License:** MIT License
- **Integrated Components:**
  - `models/retinexformer/model.py`: Illumination-Guided Multi-head Self-Attention (IG-MSA) mechanism.
  - Illumination estimation prior network and physical Retinex recombination stage.
- **Role in Project:** Transformer-based baseline for comparative enhancement and downstream detection benchmarking.

---

## 3. Zero-DCE: Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **Official Repository:** [https://github.com/Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)
- **Paper:** *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement* (CVPR 2020)
- **Authors:** Chongyi Li, Chunle Guo, Chen Change Loy
- **License:** Apache License 2.0
- **Integrated Components:**
  - `models/zero_dce/model.py`: 7-layer DCE-Net predicting iterative pixel-wise higher-order curve parameter maps ($A_n \in \mathbb{R}^{B \times 24 \times H \times W}$).
  - Non-reference loss formulation: spatial consistency loss ($\mathcal{L}_{spa}$), exposure control loss ($\mathcal{L}_{exp}$), color constancy loss ($\mathcal{L}_{col}$), and illumination smoothness loss ($\mathcal{L}_{tv}$).
  - `checkpoints/zero_dce_model.pth`: 79,416 parameters.
- **Role in Project:** Primary zero-reference baseline illustrating iterative curve adjustment vs. decomposition-based models.

---

## 4. URetinex-Net: Retinex-based Deep Unfolding Network for Low-Light Image Enhancement
- **Official Repository:** [https://github.com/AndersonYong/URetinex-Net](https://github.com/AndersonYong/URetinex-Net)
- **Paper:** *URetinex-Net: Retinex-based Deep Unfolding Network for Low-Light Image Enhancement* (CVPR 2022)
- **Authors:** Wu Yonghua, et al.
- **License:** Apache License 2.0
- **Research Adaptation:**
  - Formulation of iterative optimization steps as unfolded neural network sub-modules.
  - Served as the theoretical foundation for IGR-Net's decoupled illumination and noise optimization.

---

## 5. RetinexNet: Deep Retinex Decomposition for Low-Light Enhancement
- **Official Repository:** [https://github.com/weichen582/RetinexNet](https://github.com/weichen582/RetinexNet)
- **Paper:** *Deep Retinex Decomposition for Low-Light Enhancement* (BMVC 2018)
- **Authors:** Chen Wei, Wenjing Wang, Wenhan Yang, Jiaying Liu
- **License:** MIT License
- **Research Adaptation:**
  - Classic Decom-Net and Relight-Net Retinex structure ($I = R \odot L$).
  - Provided baseline constraints for piecewise smooth illumination priors and reflectance consistency.

---

## 6. Main Contribution: Proposed IGR-Net
- **Authors:** Project Author (M.Tech Minor Project)
- **Architecture:** Illumination-Guided Retinex Network (`62,060` parameters, 0.25 MB).
- **Core Modules:**
  1. Max-RGB initialization prior ($L_{\text{init}} = \max_{c \in \{R,G,B\}} I_c$).
  2. Local-Global Illumination Module with depthwise separable multi-scale convolutions.
  3. Noise-Aware Spatial Attention Block suppressing shadow noise artifacts.
  4. Squeeze-and-Excitation Channel Attention Block preserving chromatic balance.
- **Attribution:** Fully custom novel implementation located in `models/igr_net/`.
