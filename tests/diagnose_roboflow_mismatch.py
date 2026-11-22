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

rf_img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00026_rf.jpg")
txt_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00026.jpg.txt")

rf_img = Image.open(rf_img_path).convert("RGB")
print(f"Roboflow image size: {rf_img.size}")

with open(txt_path, "r") as f:
    txt_content = f.read()

print("Annotation content:\n", txt_content.strip())
gts = parse_annotation_content(txt_content, rf_img.size[0], rf_img.size[1], "2015_00026.jpg.txt")
print("Parsed GTs with 640x640:", gts)

enhancer = EnhancementEngine(device='cpu')
detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

enh_igr, _, _ = enhancer.enhance(rf_img, method="IGR-Net")
ann_igr, dets_igr, _ = detector.detect_objects(enh_igr)

print("\nD-FINE-S Detections on Roboflow 640x640:")
for d in dets_igr:
    print(" ", d['class'], d['score'], d['box'])

m = compute_detailed_detection_metrics(dets_igr, gts, iou_threshold=0.5)
print("\nMetrics with Roboflow image:", m)
