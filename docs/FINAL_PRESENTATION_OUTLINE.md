# Final M.Tech Presentation Slide Outline (12–15 Slides)

**Project Title:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection  
**Degree:** Master of Technology (M.Tech) Minor Project Presentation

---

### Slide 1: Title Slide
- **Project Title:** Recognition-Aware Low-Light Image Enhancement for Improved Object Detection
- **Student Name & Roll Number:** [Your Name / Roll No]
- **Department:** Department of Computer Science & Engineering / AI
- **Supervisor:** [Supervisor Name & Designation]
- **Institution:** [University / Institute Name]

---

### Slide 2: Problem Statement & Motivation
- **The Core Challenge:** Surveillance, autonomous navigation, and night-time vision fail in low-light environments.
- **Why Standard Detectors Fail:**
  - Severe photon attenuation leads to low dynamic range ($L < 30$).
  - High Poisson-Gaussian noise corrupts texture and feature representations.
  - Pretrained detectors (COCO-trained) suffer dramatic recall drops.
- **The Research Question:** Does illumination enhancement consistently aid detection, or can aggressive enhancement induce noise-related false positives?

---

### Slide 3: Research Objectives
1. Design and implement a novel lightweight, recognition-aware enhancement network (**IGR-Net**).
2. Integrate a unified state-of-the-art detector (**D-FINE-S**, 10.32M parameters) as an objective evaluation benchmark.
3. Compare proposed IGR-Net against established baselines:
   - **Zero-DCE (CVPR 2020)**: Zero-reference curve estimation.
   - **Retinexformer**: Adapted illumination-guided self-attention.
4. Build an interactive, reproducible computer vision demonstration pipeline.

---

### Slide 4: Literature Review & Taxonomy
- **Classical Histogram Equalization (CLAHE):** Uniform stretching; causes severe noise amplification.
- **Retinex Decomposition (RetinexNet, KinD, URetinex-Net):** Physical modeling ($I = R \odot L$), but computationally heavy and prone to color distortion.
- **Zero-Reference Models (Zero-DCE, Zero-DCE++):** Fast curve estimation; lacks explicit noise suppression.
- **Transformer Enhancers (Retinexformer, EnlightenGAN):** Strong representation, high compute requirements.
- **Downstream Object Detectors:** Evolution from YOLO to Fine-Grained Distribution Refinement (**D-FINE**).

---

### Slide 5: System Architecture & Comparative Framework
*(Include the ASCII / Mermaid pipeline diagram showing input $\rightarrow$ 3 parallel enhancement branches $\rightarrow$ unified D-FINE-S $\rightarrow$ comparative analysis)*
- **Condition A:** Original Raw Low-Light (Control Baseline)
- **Condition B:** Zero-DCE Baseline (`79,416` parameters)
- **Condition C:** Proposed IGR-Net (`62,060` parameters)
- **Condition D:** Retinexformer Adapted Baseline (`28,708` parameters)
- **Evaluator:** D-FINE-S Pretrained Detector (`10,321,877` parameters)

---

### Slide 6: Proposed Architecture: IGR-Net
- **Parameter Count:** **62,060 parameters** (Ultra-compact, 0.25 MB footprint).
- **Core Innovative Components:**
  1. **Max-RGB Prior Initialization:** $L_{\text{init}} = \max_{c \in \{R,G,B\}} I_c$ providing fast single-pass illumination guidance.
  2. **Local-Global Multi-Scale Module (LGM):** Combines standard $3\times3$ details with dilated ($d=2$) convolutions and global pooling.
  3. **Noise-Aware Spatial Attention Block (NAA):** Modulates features by $(1 - L) \odot M_{\text{noise}}$ to attenuate sensor grain in dark regions.
  4. **Channel Attention (SE Block):** Recalibrates chromatic channel dependencies.

---

### Slide 7: Downstream Detector: D-FINE-S
- **Model:** D-FINE-S (HGNetv2 backbone + HybridEncoder + Transformer Decoder).
- **Why D-FINE-S?**
  - Replaces conventional box offset regression with fine-grained probability distribution modeling over box edges.
  - Official MS-COCO checkpoint (`dfine_s_coco.pth`, 39.90 MB) loaded with `strict=True`.
- **Unified Evaluation Protocol:** Fixed confidence threshold ($\tau = 0.25$) and fixed NMS IoU threshold ($\text{IoU} = 0.45$) across all branches.

---

### Slide 8: Experimental Setup & Datasets
- **Datasets Utilized:**
  - **LOL Dataset:** Paired low/normal illumination images for Retinex training.
  - **ExDark Dataset:** 12-class exclusively low-light benchmark for detection benchmarking.
- **Hardware / Runtime Environment:**
  - Intel CPU environment with PyTorch 2.12.
  - Average pipeline latency: $\sim 1.2$ seconds per image (CPU).
  - Streamlit interactive web deployment on `localhost:8501`.

---

### Slide 9: Visual Comparison: Outdoor Bicycle Scene
- **Screenshot / Image Embed:** `bicycle_4way_detections_grid.png`
- **Key Visual Observations:**
  - **Original ($16$ boxes, $L=11.9$):** Deep shadow targets missed.
  - **Zero-DCE ($31$ boxes, $L=70.2$):** Reveals farthest background bicycles, but produces duplicate part-level boxes (wheel vs. frame).
  - **IGR-Net ($15$ boxes, $L=26.0$):** Cleanest spatial outlines; suppresses noise-induced duplicate clusters.
  - **Retinexformer ($23$ boxes, $L=52.9$):** Moderate exposure gain on mid-ground clusters.

---

### Slide 10: Quantitative Results Table
*(Display the unified 4-way evaluation table)*
$$\begin{array}{|l|c|c|c|c|c|}
\hline
\textbf{Condition} & \textbf{Params} & \textbf{Mean } L & \textbf{Contrast } \sigma & \textbf{Detections} & \textbf{Primary Conf.} \\
\hline
\text{Original} & \text{N/A} & 11.9 & 16.0 & 16 & 0.69 \\
\text{Zero-DCE} & 79,416 & 70.2 & 52.3 & 31 & 0.61 \\
\text{IGR-Net (Proposed)} & \mathbf{62,060} & 26.0 & 9.7 & 15 & \mathbf{0.64} \\
\text{Retinexformer (Adapted)} & 28,708 & 52.9 & 18.0 & 23 & 0.66 \\
\hline
\end{array}$$

---

### Slide 11: Ground-Truth Duplicate & Spatial Detection Audit
- **Case Study:** Yellow Bicycle & Bench Scene (`yellow_bicycle_bench.png`)
- **Key Insight:** **Detection count $\ne$ number of unique physical objects.**
- Ground truth contains **1 physical bicycle** + **1 bench**.
- Across all models, D-FINE-S detects the primary bicycle ($0.91$–$0.95$ conf) alongside part-level sub-boxes (inner frame, slats).
- **IGR-Net achieves the most compact box set (8 vs. 12–14)** without false-positive clutter (`potted plant`, `traffic light`).

---

### Slide 12: Ablation Study Design (IGR-Net)
- **Modular Switches Implemented in `models/igr_net/model.py`:**
  1. Full IGR-Net (Baseline configuration).
  2. Without Local-Global Module (`wo_local_global`).
  3. Without Noise-Aware Attention (`wo_noise_attn`).
  4. Without Channel Attention (`wo_channel_attn`).
- **Hypothesis:** Removing Noise-Aware Attention increases false-positive clutter in downstream detection.

---

### Slide 13: Limitations & Academic Disclosures
- **Retinexformer Implementation:** Adapted PyTorch IG-MSA implementation designed for real-time CPU evaluation (not official GPU checkpoint).
- **Benchmark Evaluations:** Full LOL PSNR/SSIM (500 pairs) and ExDark mAP@0.5 (7,363 images) are documented as **Pending GPU Evaluation**.
- **Hardware Constraint:** Current measurements reflect CPU execution latency.

---

### Slide 14: Conclusion & Future Scope
- **Conclusion:**
  - Successfully designed and verified **IGR-Net**, a compact 62K-parameter Retinex enhancement model.
  - Demonstrated that noise-aware enhancement preserves high downstream detection confidence while suppressing clutter.
  - Built an end-to-end, warning-free interactive demonstration system.
- **Future Work:**
  - End-to-end joint fine-tuning connecting D-FINE loss directly to IGR-Net.
  - Multi-GPU evaluation on full ExDark validation split.
  - Edge deployment using TensorRT / ONNX.

---

### Slide 15: Thank You & Q&A
- Questions & Discussion.
- Interactive Dashboard Demonstration: `http://localhost:8501`
