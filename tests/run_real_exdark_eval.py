import os
import sys
from PIL import Image

PROJECT_ROOT = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "yellow_bicycle_bench.png")
txt_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00003.png.txt")

img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"Loaded image {img_path}: {w}x{h}")

with open(txt_path, "r") as f:
    raw_txt = f.read()

print("Raw ExDark annotation content:\n", raw_txt.strip())

gts = parse_annotation_content(raw_txt, w, h, filename="2015_00003.png.txt")
print("\nParsed Ground Truths:")
for g in gts:
    print(f"  Class: {g['class']}, BBox [x1, y1, x2, y2]: {g['bbox']}")

enhancer = EnhancementEngine(device='cpu')
detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

methods = ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]
for m in methods:
    if m == "Original":
        enh = img
    else:
        enh, _, _ = enhancer.enhance(img, method=m)
    
    ann, dets, lat = detector.detect_objects(enh)
    metrics = compute_detailed_detection_metrics(dets, gts, iou_threshold=0.50)
    print(f"\n==========================================")
    print(f"CONDITION: {m}")
    print(f"==========================================")
    print(f"Total D-FINE-S Predictions: {len(dets)}")
    print(f"mAP@50:    {metrics['mAP@50']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"Matching:  {metrics['tp']} True Positives, {metrics['fp']} False Positives, {metrics['fn']} False Negatives")
    print("Class-wise Breakdown:")
    for c_name, c_data in metrics['class_breakdown'].items():
        print(f"  [{c_name.capitalize()}] GT: {c_data['gt']} | Pred: {c_data['pred']} | TP: {c_data['tp']} | FP: {c_data['fp']} | FN: {c_data['fn']} | AP: {c_data['ap']:.3f}")
