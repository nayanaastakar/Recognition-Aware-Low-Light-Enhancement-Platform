import sys
import os
from PIL import Image

PROJECT_ROOT = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_02446.jpg")
txt_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_02446.jpg.txt")

img = Image.open(img_path).convert("RGB")
w, h = img.size
print(f"Uncropped ExDark Image {img_path}: {w}x{h}")

with open(txt_path, "r") as f:
    txt_content = f.read()

print("Annotation content:\n", txt_content.strip())
gts = parse_annotation_content(txt_content, w, h, "2015_02446.jpg.txt")
print("Parsed GTs:", gts)

enhancer = EnhancementEngine(device='cpu')
detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

methods = ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]
for m in methods:
    if m == "Original":
        enh = img
    else:
        enh, _, _ = enhancer.enhance(img, method=m)
    
    ann, dets, _ = detector.detect_objects(enh)
    res = compute_detailed_detection_metrics(dets, gts, iou_threshold=0.5)
    print(f"\nCondition {m}:")
    print(f"  Detections ({len(dets)}):", [(d['class'], round(d['score'], 2), d['box']) for d in dets[:3]])
    print(f"  mAP@50: {res['mAP@50']:.3f}, Precision: {res['precision']:.3f}, Recall: {res['recall']:.3f}, F1: {res['f1_score']:.3f}")
    print(f"  TP: {res['tp']}, FP: {res['fp']}, FN: {res['fn']}")
