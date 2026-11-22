import os
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Theme colors
    NAVY = RGBColor(15, 23, 42)      # #0F172A
    BLUE = RGBColor(37, 99, 235)     # #2563EB
    DARK_TEXT = RGBColor(30, 41, 59) # #1E293B
    MUTED = RGBColor(100, 116, 139)  # #64748B
    BG_CARD = RGBColor(248, 250, 252)
    BORDER = RGBColor(226, 232, 240)
    HEADER_BG = RGBColor(241, 245, 249)

    def add_header(slide, title_text, category_text="M.TECH MINOR PROJECT"):
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = BLUE

        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.7), Inches(0.75))
        tf = title_box.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(23)
        p.font.bold = True
        p.font.color.rgb = NAVY

    def add_card(slide, left, top, width, height, bg_color=BG_CARD, border_color=BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        return shape

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.3), Inches(11.333), Inches(3.6))
    tf = tb.text_frame
    p0 = tf.paragraphs[0]
    p0.text = "M.TECH MINOR PROJECT PRESENTATION"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(56, 189, 248)

    p1 = tf.add_paragraph()
    p1.text = "Recognition-Aware Low-Light Image Enhancement for Improved Object Detection"
    p1.font.size = Pt(30)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(255, 255, 255)
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Evaluating Retinex Physical Decomposition with Pretrained D-FINE-S Object Detection"
    p2.font.size = Pt(17)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(10)

    # Candidate & Institution Metadata box
    meta_box = s1.shapes.add_textbox(Inches(1.0), Inches(5.1), Inches(11.333), Inches(2.0))
    tf_m = meta_box.text_frame
    
    pm1 = tf_m.paragraphs[0]
    pm1.text = "Candidate: [Student Name / 1RV23SCSXX]  |  Master of Technology in Computer Science & Engineering"
    pm1.font.size = Pt(13)
    pm1.font.bold = True
    pm1.font.color.rgb = RGBColor(255, 255, 255)

    pm2 = tf_m.add_paragraph()
    pm2.text = "Department of Computer Science & Engineering  |  RV College of Engineering® (Autonomous), Bengaluru"
    pm2.font.size = Pt(12)
    pm2.font.color.rgb = RGBColor(203, 213, 225)
    pm2.space_before = Pt(4)

    pm3 = tf_m.add_paragraph()
    pm3.text = "Project Guide: [Faculty Advisor Name, Designation, Dept. of CSE]  |  Academic Year: 2025–2026"
    pm3.font.size = Pt(12)
    pm3.font.color.rgb = RGBColor(148, 163, 184)
    pm3.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Research Motivation
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "1. Problem Statement & Research Motivation", "Background & Challenges")

    add_card(s2, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Low-Light Degradation"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = NAVY
    bullets = [
        "Photon Starvation: Severe luminance attenuation (L < 30) heavily compresses scene dynamic range.",
        "Sensor Noise Amplification: Dark exposures suffer severe Poisson-Gaussian noise. Standard enhancement stretches brightness but simultaneously magnifies noise grain.",
        "Texture Loss: Structural gradients, colors, and fine contours are buried in low signal-to-noise regions."
    ]
    for b in bullets:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(10)

    add_card(s2, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tb = s2.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Detector Domain Gap"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = NAVY
    bullets2 = [
        "Pretrained Domain Gap: SOTA detectors (D-FINE-S) are trained on clean, well-lit benchmarks (MS-COCO).",
        "Spurious Clutter: Amplified high-frequency noise and boundary artifacts can trigger false-positive object proposals.",
        "The Enhancement Paradox: Aggressive illumination boosting improves human visibility, but amplified noise can distort detector feature maps."
    ]
    for b in bullets2:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(10)

    add_card(s2, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3))
    tb = s2.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Research Objectives"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = NAVY
    bullets3 = [
        "Investigate the Trade-Off: Quantify whether illumination boost consistently aids downstream detection, or if over-enhancement causes detection clutter.",
        "Design IGR-Net: A compact, recognition-aware Retinex model (62K params) with explicit illumination guidance and spatial noise-aware attenuation.",
        "Rigorous Benchmark: Compare 4 branches (Raw, Zero-DCE, IGR-Net, Retinexformer) under identical D-FINE-S detection settings."
    ]
    for b in bullets3:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 3: System Architecture & Comparative Framework
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "2. System Architecture & Comparative Framework", "System Design")

    add_card(s3, Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.3))
    tb = s3.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "End-to-End Comparative Evaluation Pipeline"
    p.font.bold = True
    p.font.size = Pt(16)
    p.font.color.rgb = NAVY

    arch_desc = """
Unified 4-Branch Enhancement Topology:
  • Branch 1 (Control / Baseline): Raw unenhanced low-light image.
  • Branch 2 (Zero-DCE - CVPR 2020): 7-layer DCE-Net predicting iterative higher-order curve parameter maps (79,416 parameters).
  • Branch 3 (Proposed IGR-Net): Max-RGB guided Retinex decomposition with Local-Global features and Noise-Aware Spatial Attention (62,060 parameters).
  • Branch 4 (Retinexformer Adapted - ICCV 2023): Lightweight Illumination-Guided Self-Attention (IG-MSA) representation (28,708 parameters).

Unified Downstream Object Detector:
  • Model: D-FINE-S (HGNetv2 B0 backbone + HybridEncoder + Transformer Decoder).
  • Pretrained Weights: Official MS-COCO Checkpoint (dfine_s_coco.pth, 10,321,877 parameters) loaded via strict state-dictionary matching.
  • Controlled Calibration: Identical confidence threshold (tau = 0.25) and identical NMS IoU threshold (IoU = 0.45) across all branches.
  • Grounded Evaluation: Standardized RGB float64 evaluation pipeline measuring mean luminance, contrast dispersion, and detection bounding boxes.
    """
    pb = tf.add_paragraph()
    pb.text = arch_desc.strip()
    pb.font.size = Pt(13)
    pb.font.color.rgb = DARK_TEXT
    pb.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 4: Proposed Architecture: IGR-Net & Attention Verification
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "3. Proposed Architecture: IGR-Net", "Methodology & Code Verification")

    add_card(s4, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Architectural Innovations & Parameters"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY

    bullets_igr = [
        "Lightweight Footprint: Exactly 62,060 parameters (0.25 MB) — highly viable for CPU / embedded real-time execution.",
        "Max-RGB Illumination Prior: Initializes the illumination map directly via L_init = max(R,G,B), eliminating iterative optimization steps.",
        "Local-Global Multi-Scale Module (LGM): Parallel local 3x3 convolutions, dilated convolutions (d=2) for broad receptive field, and Squeeze-and-Excitation channel gating.",
        "Physical Retinex Fusion: Reconstructs enhanced output via I_enhanced = R_refined * L_enhanced + residual_delta.",
        "Built-in Ablation Switches: Model source includes toggles (wo_local_global, wo_noise_attn, wo_channel_attn) for modular validation."
    ]
    for b in bullets_igr:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    add_card(s4, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb = s4.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Noise-Aware Attention Code Verification"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY

    math_desc = """
Exact Implementation (models/igr_net/model.py):
  class NoiseAwareAttention(nn.Module):
    # f = ReLU(Conv1(x))
    # spatial_noise_map = Sigmoid(Conv2(f))      --> M_noise in [0, 1]
    # noise_guidance = (1.0 - L_init) * spatial_noise_map
    # attn_out = x * (1.0 - 0.5 * noise_guidance)

Mathematical Representation:
  W_attn = 1.0 - 0.5 * ((1.0 - L_init) * M_noise)

Physical Meaning of Variables:
  • L_init: Initial illumination prior in [0, 1]. Darker pixels produce higher (1 - L_init) weighting.
  • M_noise: Spatial feature variance map predicted by 2-layer convolution + Sigmoid, identifying high-frequency noise regions.
  • Factor 0.5: Attenuation damping coefficient. It prevents over-smoothing by bounding the maximum feature attenuation to 50%, ensuring structural edges and contours are strictly preserved.
    """
    pb = tf.add_paragraph()
    pb.text = math_desc.strip()
    pb.font.size = Pt(11.5)
    pb.font.color.rgb = DARK_TEXT
    pb.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 5: Baseline Models & Detector Attribution
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "4. Baseline Models & Detector Attribution", "Model Provenance")

    add_card(s5, Inches(0.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tb = s5.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(3.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Zero-DCE (CVPR 2020)"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    b_dce = [
        "Authors: Chongyi Li, Chunle Guo, Chen Change Loy.",
        "Architecture: 7-layer DCE-Net with symmetric skip connections.",
        "Parameters: 79,416 (0.31 MB).",
        "Mechanism: Estimates dynamic higher-order pixel-wise curve parameter maps iteratively without paired ground truth.",
        "Status: Genuine, fully implemented and verified."
    ]
    for b in b_dce:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(8)

    add_card(s5, Inches(4.8), Inches(1.6), Inches(3.6), Inches(5.3))
    tb = s5.shapes.add_textbox(Inches(5.0), Inches(1.8), Inches(3.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Retinexformer (ICCV 2023)"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    b_ret = [
        "Authors: Yuanhao Cai, Hao Bian, Jing Lin, et al.",
        "Core Module: Illumination-Guided Multi-Head Self-Attention (IG-MSA).",
        "Parameters: 28,708.",
        "Implementation: Adapted lightweight PyTorch representation for CPU inference.",
        "Academic Disclosure: Explicitly cataloged as an adapted PyTorch implementation, not the official GPU-trained checkpoint."
    ]
    for b in b_ret:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(8)

    add_card(s5, Inches(8.8), Inches(1.6), Inches(3.7), Inches(5.3))
    tb = s5.shapes.add_textbox(Inches(9.0), Inches(1.8), Inches(3.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "D-FINE-S Detector (2024)"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    b_dfine = [
        "Authors: Yansong Peng, et al.",
        "Architecture: HGNetv2 B0 backbone + HybridEncoder + Transformer Decoder.",
        "Parameters: 10,321,877 (39.90 MB).",
        "Weights: Official MS-COCO checkpoint (dfine_s_coco.pth) loaded via strict state matching.",
        "Mechanism: Fine-Grained Distribution Refinement (FDR) probability modeling."
    ]
    for b in b_dfine:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 6: Visual Comparison: Outdoor Night Scene
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5. Visual Comparison: Outdoor Night Scene", "Experimental Results")

    grid_path = os.path.join(PROJECT_ROOT, "outputs", "demonstration", "bicycle_4way_detections_grid.png")
    if os.path.exists(grid_path):
        s6.shapes.add_picture(grid_path, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s6, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    tb = s6.shapes.add_textbox(Inches(8.0), Inches(1.8), Inches(4.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Observations & Detection Differences"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    obs = [
        "Original (16 boxes, L = 11.95): Low illumination causes heavy occlusion; background bicycles fall below detection thresholds.",
        "Zero-DCE (31 boxes, L = 70.16): Highest illumination boost reveals faint background targets, but also produces multiple overlapping part-level boxes (wheel vs. frame).",
        "IGR-Net (15 boxes, L = 26.03): Produces fewer predicted boxes than other enhancement methods in this scene, while retaining a bicycle detection confidence of 0.64.",
        "Retinexformer (21 boxes, L = 53.33): Balanced illumination recovery detects primary mid-ground clusters.",
        "Cautious Interpretation: Fewer boxes indicate reduced detection clutter in this scene, but formal accuracy comparison requires full ground-truth benchmark mAP."
    ]
    for o in obs:
        pb = tf.add_paragraph()
        pb.text = "• " + o
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    # -------------------------------------------------------------
    # SLIDE 7: Reconciled Quantitative Evaluation Table (PPT Native Table)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "6. Reconciled Quantitative Evaluation Table", "Empirical Evaluation")

    add_card(s7, Inches(0.8), Inches(1.55), Inches(11.7), Inches(5.4))
    
    # Table Header text
    tb_title = s7.shapes.add_textbox(Inches(1.0), Inches(1.65), Inches(11.3), Inches(0.5))
    tf_tt = tb_title.text_frame
    p_tt = tf_tt.paragraphs[0]
    p_tt.text = "Target Scene: yellow_bicycle_bench.png (500 x 375) — Ground Truth: 1 Bicycle + 1 Bench"
    p_tt.font.size = Pt(14)
    p_tt.font.bold = True
    p_tt.font.color.rgb = NAVY

    # Add Native PowerPoint Table
    rows = 5
    cols = 6
    left = Inches(1.0)
    top = Inches(2.2)
    width = Inches(11.3)
    height = Inches(2.1)
    table_shape = s7.shapes.add_table(rows, cols, left, top, width, height)
    tbl = table_shape.table

    # Column widths
    tbl.columns[0].width = Inches(2.7)
    tbl.columns[1].width = Inches(1.5)
    tbl.columns[2].width = Inches(1.6)
    tbl.columns[3].width = Inches(1.7)
    tbl.columns[4].width = Inches(1.6)
    tbl.columns[5].width = Inches(2.2)

    headers = ["Condition", "Parameters", "Mean (L)", "Contrast (σ)", "D-FINE-S Boxes", "Top Detection Conf."]
    data = [
        ["Condition A: Original", "N/A", "24.08", "22.50", "13 boxes", "bicycle: 0.94"],
        ["Condition B: Zero-DCE", "79,416", "114.92", "56.03", "12 boxes", "bicycle: 0.95"],
        ["Condition C: IGR-Net", "62,060", "34.46", "12.88", "8 boxes", "bicycle: 0.95"],
        ["Condition D: Retinexformer", "28,708", "67.47", "25.18", "13 boxes", "bicycle: 0.85"]
    ]

    for col_idx, h in enumerate(headers):
        cell = tbl.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BLUE
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = RGBColor(255, 255, 255)
        p.alignment = PP_ALIGN.CENTER

    for row_idx, row_data in enumerate(data):
        for col_idx, val in enumerate(row_data):
            cell = tbl.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if row_idx % 2 == 0 else HEADER_BG
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.size = Pt(11.5)
            p.font.color.rgb = DARK_TEXT
            if col_idx in [1, 2, 3, 4, 5]:
                p.alignment = PP_ALIGN.CENTER
            if row_idx == 2 and col_idx == 4:  # Highlight IGR-Net 8 boxes
                p.font.bold = True
                p.font.color.rgb = BLUE

    # Outdoor parking scene summary box below table
    tb_bottom = s7.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.3), Inches(2.2))
    tf_b = tb_bottom.text_frame
    tf_b.word_wrap = True
    p_b0 = tf_b.paragraphs[0]
    p_b0.text = "Outdoor Parking Scene (bicycle_extracted.png, 617 x 439) & Scientific Insights:"
    p_b0.font.bold = True
    p_b0.font.size = Pt(13)
    p_b0.font.color.rgb = NAVY

    b_pts = [
        "Comparative Summary: Original: 16 boxes (L=11.95, σ=15.97) | Zero-DCE: 31 boxes (L=70.16, σ=52.32) | IGR-Net: 15 boxes (L=26.03, σ=9.73) | Retinexformer: 21 boxes (L=53.33, σ=17.45).",
        "Consistent Reconciled Verification: In the Yellow Bicycle scene, IGR-Net produces exactly 8 detections (verified from saved inference coordinates), while Original & Retinexformer output 13 and Zero-DCE outputs 12.",
        "Contrast & Latency: IGR-Net exhibits moderate contrast dispersion (σ = 12.88 / 9.73), avoiding noise grain blowup. End-to-end CPU pipeline latency is ~1.0-1.2s (Enhancement ~350ms + D-FINE-S ~650ms)."
    ]
    for pt in b_pts:
        pb = tf_b.add_paragraph()
        pb.text = "• " + pt
        pb.font.size = Pt(11.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(4)

    # -------------------------------------------------------------
    # SLIDE 8: Ground-Truth Detection & Spatial Overlap Audit
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "7. Ground-Truth Detection & Spatial Overlap Audit", "Scientific Rigor")

    grid_yb = os.path.join(PROJECT_ROOT, "outputs", "demonstration", "yellow_bicycle_4way_validation_grid.png")
    if os.path.exists(grid_yb):
        s8.shapes.add_picture(grid_yb, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s8, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    tb = s8.shapes.add_textbox(Inches(8.0), Inches(1.8), Inches(4.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Part-Level Boxes vs. Formal Accuracy"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY

    audit_pts = [
        "Physical Ground Truth: Exactly 1 physical bicycle + 1 outdoor bench in foreground.",
        "Primary Object Localization: All four pipelines cleanly identify the physical bicycle with top confidence (0.85 - 0.95).",
        "IoU & Overlapping Part-Level Boxes: The whole-bicycle box ([298, 108, 497, 345]) covers ~25% of the frame. A secondary box ([274, 122, 363, 248]) frames only the inner triangle. Spatial IoU is ~0.24, well below the standard NMS threshold (0.45), so both legitimately survive.",
        "Academic Clarification: This IoU calculation explains why sub-boxes persist without detector malfunction; it is not a formal claim of detection accuracy.",
        "Defensible Takeaway: IGR-Net yields fewer redundant boxes (8 boxes vs. 12-13) while keeping primary bicycle confidence at 0.95, indicating lower detection clutter in this scene."
    ]
    for pt in audit_pts:
        pb = tf.add_paragraph()
        pb.text = "• " + pt
        pb.font.size = Pt(11.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 9: Indoor Low-Light Analysis: Noise vs. Detection Clutter
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "8. Indoor Low-Light Analysis: Noise vs. Detection Clutter", "Detection Behavior")

    grid_in = os.path.join(PROJECT_ROOT, "outputs", "demonstration", "indoor_sample001_4way_detections_grid.png")
    if os.path.exists(grid_in):
        s9.shapes.add_picture(grid_in, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s9, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    tb = s9.shapes.add_textbox(Inches(8.0), Inches(1.8), Inches(4.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Clutter Reduction vs. Noise Claims"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY

    in_pts = [
        "Indoor Low-Light Scene: Severely dark environment (mean luminance ~ 19) with heavy shadow grain.",
        "Spurious Triggering: In raw and over-enhanced conditions, high-variance shadow artifacts activate spurious object proposals across blank walls (umbrella: 0.45, bed: 0.37).",
        "Empirical Clutter Reduction: IGR-Net produces fewer spurious bounding boxes in low-contrast background regions compared to aggressive brightness boosting.",
        "Rigorous Academic Disclaimer: Lower box count indicates reduced detection clutter, but does NOT independently prove Poisson-Gaussian noise suppression. Physical noise attenuation requires separate full-reference PSNR/SSIM and noise-variance measurements across curated datasets."
    ]
    for pt in in_pts:
        pb = tf.add_paragraph()
        pb.text = "• " + pt
        pb.font.size = Pt(11.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    # -------------------------------------------------------------
    # SLIDE 10: Interactive Demonstration Dashboard (with Actual Screenshot)
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "9. Interactive Demonstration Dashboard", "Software Engineering")

    ss_path = os.path.join(PROJECT_ROOT, "docs", "streamlit_dashboard_screenshot.jpg")
    if os.path.exists(ss_path):
        s10.shapes.add_picture(ss_path, Inches(0.8), Inches(1.6), width=Inches(6.8))

    add_card(s10, Inches(7.8), Inches(1.6), Inches(4.7), Inches(5.3))
    tb = s10.shapes.add_textbox(Inches(8.0), Inches(1.8), Inches(4.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Full-Stack Web Application"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY

    app_desc = [
        "Live Deployment: Hosted locally on Streamlit (http://localhost:8501) with responsive multi-column comparison.",
        "Dynamic Ingestion: Supports sample selection, arbitrary file uploads (PNG/JPG), and real-time webcam frame acquisition.",
        "Unified 4-Way Inference: Concurrently evaluates Raw Baseline, Zero-DCE, IGR-Net, and Retinexformer under D-FINE-S.",
        "Memory Safeguard: Automatic bilinear downsampling (max dimension 800px) prevents CPU allocation memory spikes.",
        "Session State Reset: Cache invalidation ensures uploaded images update immediately across all 4 branches.",
        "Detailed Inspection: Renders side-by-side visual bounding boxes, real-time latencies, and interactive bounding box tables ([x1, y1, x2, y2], confidence)."
    ]
    for pt in app_desc:
        pb = tf.add_paragraph()
        pb.text = "• " + pt
        pb.font.size = Pt(11.5)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(6)

    # -------------------------------------------------------------
    # SLIDE 11: Academic Limitations & Pending Benchmarks
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "10. Academic Limitations & Pending Benchmarks", "Transparency & Integrity")

    add_card(s11, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tb = s11.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Completed vs. Pending Evaluations"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    b_eval = [
        "Completed: Synchronized 4-branch pipeline, official D-FINE-S checkpoint integration, RGB float64 luminance/contrast calculations, spatial IoU duplicate audit, and real-time CPU latency benchmarking.",
        "Pending GPU Benchmark: Full reference-based PSNR/SSIM across the 500 image pairs of the LOL dataset.",
        "Pending GPU Benchmark: Multi-class mAP@0.5 and mAP@0.5:0.95 across all 7,363 images of the ExDark low-light dataset.",
        "Academic Integrity: Benchmark scripts are implemented and validated on sample pairs; full-dataset numbers are cataloged as pending rather than simulated or unverified."
    ]
    for b in b_eval:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    add_card(s11, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb = s11.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Methodological Disclosures"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    b_disc = [
        "Retinexformer Baseline: Evaluated using an adapted lightweight PyTorch IG-MSA architecture, not the official multi-GPU checkpoint.",
        "Detection Metric Grounding: Single-scene detection counts indicate detection density and clutter, and must not be conflated with dataset-level mAP or recall.",
        "Hardware Constraints: CPU execution averages ~1.0-1.2s total pipeline latency. Edge deployment (e.g., Jetson) requires ONNX / TensorRT acceleration."
    ]
    for b in b_disc:
        pb = tf.add_paragraph()
        pb.text = "• " + b
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    # -------------------------------------------------------------
    # SLIDE 12: Conclusion & Future Scope
    # -------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    add_header(s12, "11. Conclusion & Future Scope", "Summary & Roadmap")

    add_card(s12, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.3))
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(5.2), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Project Conclusions"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    concl = [
        "Proposed IGR-Net Architecture: Successfully designed and deployed an ultra-compact (62,060 parameters) Retinex model with Max-RGB illumination guidance.",
        "Clutter Mitigation: Demonstrated that targeted spatial attention reduces spurious background proposals while preserving high detection confidence (0.95) for primary objects.",
        "Rigorous Unified Detector: Integrated the official COCO-pretrained D-FINE-S detector (10.32M parameters) without uncalibrated random layers.",
        "Interactive Dashboard: Delivered a robust, warning-free web application for side-by-side low-light computer vision evaluation."
    ]
    for c in concl:
        pb = tf.add_paragraph()
        pb.text = "• " + c
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    add_card(s12, Inches(6.8), Inches(1.6), Inches(5.7), Inches(5.3))
    tb = s12.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Future Research Roadmap"
    p.font.bold = True
    p.font.size = Pt(15.5)
    p.font.color.rgb = NAVY
    future = [
        "End-to-End Joint Fine-Tuning: Propagate D-FINE-S detection and classification loss gradients back into IGR-Net attention weights for true recognition-driven enhancement.",
        "Full ExDark GPU Evaluation: Execute distributed GPU benchmark across all 7,363 images to calculate class-wise mAP@0.5 and precision-recall curves.",
        "Embedded Acceleration: Quantize IGR-Net and D-FINE-S via TensorRT / ONNX for real-time edge processing on embedded platforms (NVIDIA Jetson / robotic systems)."
    ]
    for f in future:
        pb = tf.add_paragraph()
        pb.text = "• " + f
        pb.font.size = Pt(12)
        pb.font.color.rgb = DARK_TEXT
        pb.space_before = Pt(7)

    # -------------------------------------------------------------
    # SLIDE 13: Viva Q&A Slide
    # -------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    bg = s13.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()

    tb = s13.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.333), Inches(3.2))
    tf = tb.text_frame
    p0 = tf.paragraphs[0]
    p0.text = "Thank You"
    p0.font.size = Pt(40)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(255, 255, 255)

    p1 = tf.add_paragraph()
    p1.text = "Questions & Academic Discussion"
    p1.font.size = Pt(22)
    p1.font.color.rgb = RGBColor(56, 189, 248)
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Interactive Application Live at: http://localhost:8501"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(203, 213, 225)
    p2.space_before = Pt(14)

    p3 = tf.add_paragraph()
    p3.text = "RV College of Engineering®  |  Department of Computer Science & Engineering"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(148, 163, 184)
    p3.space_before = Pt(8)

    # Save presentation
    ppt_path = os.path.join(PROJECT_ROOT, "docs", "M_TECH_FINAL_PRESENTATION.pptx")
    prs.save(ppt_path)
    print("Successfully created M_TECH_FINAL_PRESENTATION.pptx at:", ppt_path)

if __name__ == "__main__":
    create_presentation()
