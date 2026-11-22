# M.Tech Minor Project Presentation & Academic Defense Guide

**Project Title:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection  
**Author:** Candidate (M.Tech Minor Project)  
**Evaluator Reference:** Scientific Clarifications, Contribution Architecture, and Evaluation Protocol

---

## 1. Grounding Your Research Contribution: What Exactly Is IGR-Net?

When asked by your evaluation panel: *"Is IGR-Net your own proposed architecture or an adaptation?"*

**The Precise Academic Answer:**
> *"**IGR-Net (Illumination-Guided and Recognition-Aware Enhancement Network)** is our **proposed novel lightweight architecture (62,060 parameters)**. It is grounded in classic Retinex physical decomposition principles ($I = R \odot L$), but integrates our custom-designed modules to make enhancement recognition-aware:*
> 1. *A **Max-RGB Illumination Prior** ($L_{\text{init}} = \max_{c \in \{R,G,B\}} I_c$) that avoids iterative curve passes.*
> 2. *A **Local-Global Multi-Scale Module** combining local $3\times3$ details, dilated receptive fields ($d=2$), and global squeeze-and-excitation pooling.*
> 3. *A novel **Noise-Aware Spatial Attention Block** that explicitly computes an inverted illumination guidance map $(1 - L) \odot M_{\text{spatial}}$ to attenuate noise in extreme shadow regions before downstream detection.*
> 4. *In contrast, **Zero-DCE (79,416 params)** is the official CVPR 2020 curve-estimation baseline, and **Retinexformer (28,708 params)** is an adapted lightweight IG-MSA self-attention baseline."*

---

## 2. Objective Analysis of the Test Predictions (No Overclaims)

### A. Object Detection Predictions on the Indoor Bedroom Scene
| Predicted Label | Physical Object in Scene | Scientific Classification | Academic Presentation Guidance |
| :--- | :--- | :--- | :--- |
| **`bicycle: 0.91`** | Mountain bicycle against bed | **True Positive** | High confidence across all conditions. |
| **`bed: 0.26`** | Red fabric mattress & bedding | **True Positive** | Correctly localized. |
| **`backpack: 0.38`** / **`handbag: 0.38`** | Blue/black bag on floor | **True Object, Multi-label Duplicate** | Acknowledge that the detector fires duplicate concurrent labels for one physical bag due to class overlap in MS-COCO. |
| **`suitcase: 0.31`** | Packaged cardboard shipping boxes | **Plausible Misclassification** | Do **not** claim the cardboard boxes are suitcases. State: *"The detector classifies the large rectangular packaged boxes as suitcases due to geometric similarity in training priors."* |
| **`chair: 0.30`** / **`bench`** | Wooden headboard & bed frame | **Structural Prior Ambiguity** | Slat-like headboard geometry activates furniture prior anchors. |

---

## 3. How to Present the Metric Table Scientifically

| Metric | Measured Value (Original $\rightarrow$ Zero-DCE $\rightarrow$ IGR-Net $\rightarrow$ Retinexformer) | Scientifically Defensible Statement | What NOT to Claim |
| :--- | :--- | :--- | :--- |
| **Mean Luminance ($L$)** | $39.9 \rightarrow 140.4 \rightarrow 43.0 \rightarrow 85.4$ | Zero-DCE exhibits the largest global brightness shift; Retinexformer provides moderate elevation; IGR-Net maintains low-exposure restraint. | Do *not* claim higher luminance equals superior enhancement. |
| **Contrast ($\sigma$)** | $37.5 \rightarrow 67.7 \rightarrow 19.3 \rightarrow 42.3$ | Zero-DCE expands standard deviation dynamic range significantly, while IGR-Net produces a smoother, compact histogram. | Do *not* claim low contrast means failure, nor high contrast means noise removal. |
| **Detection Count** | $32 \rightarrow 25 \rightarrow 26 \rightarrow 29$ | All conditions successfully preserve detection of the primary bicycle ($0.91$) and room context. Differences reflect threshold boundary sensitivity. | Do *not* claim lower count means better noise removal, nor higher count means better detection accuracy. |
| **Latency** | $389\text{ ms (Enhancement)} + 1048\text{ ms (Detector)}$ | All models execute on standard CPU in under 1.5 seconds per frame. | Real-time $(>30\text{ fps})$ claims are reserved for GPU execution. |

---

## 4. Benchmark Disclosures Checklist for Tomorrow's Presentation

1. **Evaluation Status:**
   - Single-image and multi-image interactive demonstrations are **fully functional and verified live**.
   - Full quantitative benchmark evaluation over the **LOL dataset (PSNR / SSIM)** and **ExDark dataset (mAP@0.5)** is **cataloged as pending GPU compute**.
2. **Detector Equivalence:**
   - All four comparison branches use the exact same **D-FINE-S checkpoint** (`dfine_s_coco.pth`, 10.32M parameters) at identical confidence threshold ($\tau = 0.25$) and identical NMS IoU threshold ($\text{IoU} = 0.45$).
3. **Reproducibility:**
   - Full safety backup is preserved.
   - Self-contained archive is verified and packaged.
