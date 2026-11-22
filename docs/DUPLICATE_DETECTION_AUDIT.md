# Ground-Truth Duplicate & Spatial Detection Validation Audit

**Test Scene:** Yellow Bicycle & Outdoor Bench at Night (`media_1795319402059_9f563fc1.jpg`)  
**Resolution:** $500 \times 375$ pixels  
**Ground-Truth Physical Objects:**
- **Primary Subject:** Exactly **1 physical bicycle** (Yellow frame in right foreground).
- **Secondary Subject:** Exactly **1 physical outdoor bench** (Directly behind the bicycle).
- **Distant Background:** **1 parked vehicle** (Far-left background street).

---

## 1. Ground-Truth Categorization Matrix

| Condition | Total Predicted Boxes | Boxes on Primary Bicycle | Duplicate Bicycle Boxes | Boxes on Main Bench | Duplicate Bench Boxes | Distant Background Objects | Spurious Clutter Detections |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Condition A: Original** | 13 | 3 | 2 (wheel & frame sub-boxes) | 6 | 5 (slat & leg sub-boxes) | 4 (`car`, `bus`, `truck` on same far vehicle) | 0 |
| **Condition B: Zero-DCE** | 12 | 2 | 1 (sub-box) | 6 | 5 (sub-boxes) | 3 (`truck`, `car` on far vehicle) | 1 (`potted plant:0.33` on dark pillar) |
| **Condition C: IGR-Net** | **8** | **2** | **1 (sub-box)** | **5** | **4 (sub-boxes)** | **1 (`car:0.32` on far vehicle)** | **0** |
| **Condition D: Retinexformer** | 14 | 3 | 2 (sub-boxes) | 6 | 5 (sub-boxes) | 4 (`bus`, `car`, `truck` on far vehicle) | 1 (`traffic light:0.32` on distant light reflection) |

---

## 2. In-Depth Spatial Bounding Box Analysis

### A. The Primary Bicycle (Ground Truth = 1 Physical Object)
- **Primary Detection:** Across all 4 conditions, D-FINE-S detects the complete physical bicycle with high confidence:
  - Original: `[298, 108, 497, 345]`, **Confidence: 0.94**, Relative Area: **25.2%**
  - Zero-DCE: `[295, 108, 497, 346]`, **Confidence: 0.95**, Relative Area: **25.6%**
  - IGR-Net: `[298, 108, 497, 345]`, **Confidence: 0.95**, Relative Area: **25.2%**
  - Retinexformer: `[300, 108, 497, 345]`, **Confidence: 0.91**, Relative Area: **24.9%**
- **Sub-Component Duplicate:** All four models also produce a sub-box around `[274, 122, 363, 248]` (Rel. Area: ~6.0%, Conf: 0.49–0.53). This box isolates only the triangular yellow frame and handlebar rather than the full bicycle with wheels.
- **Why Standard NMS Does Not Suppress It:** The spatial Intersection over Union (IoU) between the small frame ($6\%$) and the whole bicycle ($25.2\%$) is only $\text{IoU} \approx 0.24$. Because standard NMS uses $\text{IoU}_{\text{thresh}} = 0.45$, the sub-box is retained.

### B. The Outdoor Bench (Ground Truth = 1 Physical Object)
- The bench is localized by a bounding box spanning `[217, 137, 336, 247]` (Rel. Area: ~7.0%, Conf: 0.56–0.64).
- The remaining bench predictions (4 to 5 boxes) are sub-component boxes localizing individual wooden slats and metal legs (`[218, 148, 308, 211]`, `[157, 158, 182, 186]`).

### C. The Distant Background Vehicle
- In the far-left street corner (`[0, 130, 35, 149]`, Rel. Area: 0.4%), a distant dark vehicle is visible.
- Because the resolution is low ($35 \times 19$ pixels), D-FINE-S yields concurrent predictions across vehicle classes (`car:0.32`, `bus:0.32`, `truck:0.26`).
- In IGR-Net, multi-class vehicle clutter is filtered down to a single clean prediction (`car:0.32`).

---

## 3. Academic Conclusions for the M.Tech Minor Project

1. **Detection Count vs. Ground-Truth Accuracy:**
   - 8 to 14 predicted boxes does **not** indicate 8 to 14 physical objects. It represents **2 distinct foreground physical objects (1 bicycle + 1 bench)**, **1 distant background vehicle**, plus **sub-component part detections**.
2. **IGR-Net Behavior:**
   - IGR-Net produces the fewest total boxes (8 vs. 12–14). It maintains top confidence on the true bicycle (`0.95`) and bench (`0.59`), while eliminating false-positive clutter (`potted plant` or `traffic light`) seen in other branches.
   - Without ground-truth annotations, fewer detections should be described as **higher grouping compactness and noise resistance**, rather than higher mAP.
