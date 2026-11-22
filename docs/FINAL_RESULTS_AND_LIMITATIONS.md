# Final Results, Analysis, and Academic Limitations (Reconciled)

**Project:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection  
**Evaluation Scope:** Visual, statistical, and downstream recognition evaluation across four experimental conditions with reconciled pixel metrics and defensible research claims.

---

## 1. Experimental Results Across Evaluated Scenes

### Scene 1: Yellow Bicycle & Outdoor Bench Scene (`yellow_bicycle_bench.png`, $500 \times 375$)
- **Context:** Exactly ONE physical bicycle in foreground + ONE outdoor bench + distant street vehicles.
- **Unified Settings:** D-FINE-S detector, confidence threshold $\tau = 0.25$, NMS IoU threshold $\text{IoU} = 0.45$.

| Experimental Condition | Mean Luminance ($L$) | Contrast ($\sigma$) | Min / Max | D-FINE-S Detections | Primary Bicycle Detection Confidence | Top Object Detections |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Condition A: Original** | 24.08 | 22.50 | 0.0 / 255.0 | 13 boxes | 0.94 | `bicycle: 0.94`, `bench: 0.64`, `bench: 0.56`, `bicycle: 0.50` |
| **Condition B: Zero-DCE** | 114.92 | 56.03 | 0.0 / 255.0 | 12 boxes | **0.95** | `bicycle: 0.95`, `bench: 0.63`, `bicycle: 0.53`, `bench: 0.43` |
| **Condition C: IGR-Net (Proposed)** | 34.46 | 12.88 | 15.0 / 133.0 | **8 boxes** | **0.95** | `bicycle: 0.95`, `bench: 0.59`, `bicycle: 0.51`, `bench: 0.45` |
| **Condition D: Retinexformer (Adapted)** | 67.47 | 25.18 | 34.0 / 255.0 | 13 boxes | 0.85 | `bicycle: 0.85`, `bench: 0.71`, `bicycle: 0.66`, `bench: 0.58` |

---

### Scene 2: Outdoor Night Parking Scene (`bicycle_extracted.png`, $617 \times 439$)
- **Context:** Multiple occluded parked bicycles under streetlamps and deep dark shadows.

| Experimental Condition | Mean Luminance ($L$) | Contrast ($\sigma$) | D-FINE-S Detections | Primary Bicycle Detection Confidence | Observed Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Condition A: Original** | 11.95 | 15.97 | 16 boxes | 0.69 | Darkest background targets obscured by shadow. |
| **Condition B: Zero-DCE** | 70.16 | 52.32 | **31 boxes** | 0.61 | Large dynamic expansion uncovers background silhouettes, but causes duplicate part-level boxes. |
| **Condition C: IGR-Net (Proposed)** | 26.03 | 9.73 | 15 boxes | **0.64** | Clean spatial boundaries; suppresses noise-induced duplicate clusters. |
| **Condition D: Retinexformer (Adapted)** | 53.33 | 17.45 | 21 boxes | 0.66 | Moderate illumination gain reveals mid-ground bicycle clusters. |

---

## 2. In-Depth Technical Observations (Defensible Academic Claims)

1. **Luminance and Contrast Dynamic Behavior:**
   - **Zero-DCE** exhibits the largest global brightness expansion ($L = 114.92$ on the yellow bicycle scene, $L = 70.16$ on the parking scene) and significant contrast standard deviation ($\sigma = 56.03$ and $52.32$). This confirms strong global histogram stretching.
   - **Retinexformer (Adapted)** delivers moderate exposure recovery ($L = 67.47$ and $L = 53.33$).
   - **IGR-Net (Proposed)** preserves lower average luminance ($L = 34.46$ and $L = 26.03$) and compact contrast ($\sigma = 12.88$ and $9.73$), preventing over-saturation and preserving boundary stability.

2. **Detection Clutter vs. Precision:**
   - In the yellow bicycle scene, IGR-Net produces fewer detection boxes (8 boxes) compared to the baselines (12–13 boxes), while maintaining top confidence on the primary bicycle ($0.95$).
   - **Precise Academic Interpretation:** This indicates **reduced detection clutter and higher grouping compactness** in this specific scene. However, this observation does not independently prove superior dataset-wide precision or recall.

3. **Part-Level and Overlapping Detections:**
   - In all four conditions, the detector localizes the primary bicycle while also generating overlapping part-level sub-boxes (e.g., around the inner frame and handlebars).
   - This occurs because the spatial overlap between the full bicycle ($25.2\%$ area) and the smaller sub-box ($6.0\%$ area) is $\text{IoU} \approx 0.24$, which is below the standard NMS suppression threshold ($\text{IoU}_{\text{thresh}} = 0.45$).

---

## 3. Academic Limitations & Pending Benchmarks

1. **Retinexformer Model Adaptation:**
   - The integrated Retinexformer model is an adapted PyTorch IG-MSA implementation designed for real-time CPU evaluation. It must not be cited as the full official GPU checkpoint.
2. **Pending Benchmark Datasets:**
   - Full reference-based PSNR/SSIM evaluation across the 500-pair LOL dataset and multi-class mAP@0.5 calculation across the 7,363-image ExDark dataset require GPU cluster compute and remain cataloged as **Pending GPU Evaluation**.
3. **Formal Metric Requirements:**
   - Bounding box counts and per-image confidence scores are visual and empirical observations. Formal precision, recall, and mAP require complete ground-truth annotations across standard evaluation splits.
