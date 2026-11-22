import os
import sys
from PIL import Image
import numpy as np

PROJECT_ROOT = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00005.jpg")
txt_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00005.jpg.txt")

img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"Loaded image {img_path}: {w}x{h}")

with open(txt_path, "r") as f:
    raw_txt = f.read()

print("Raw annotation content:\n", raw_txt.strip())

gts = parse_annotation_content(raw_txt, w, h, filename="2015_00005.jpg.txt")
print("Parsed Ground Truths:", gts)

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
    print(f"\n--- Condition: {m} ---")
    print(f"Detections count: {len(dets)}")
    print(f"Top dets: {[d['class'] + ':' + str(round(d['score'],2)) + str(d['box']) for d in dets[:4]]}")
    print(f"mAP@50: {metrics['mAP@50']:.3f}, Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print("Class breakdown:", metrics['class_breakdown'])
