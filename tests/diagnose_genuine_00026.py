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

orig_img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00026_original.jpg")
rf_img_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00026_rf.jpg")
txt_path = os.path.join(PROJECT_ROOT, "data", "samples", "low", "2015_00026.jpg.txt")

img_orig = Image.open(orig_img_path).convert("RGB")
img_rf = Image.open(rf_img_path).convert("RGB")

print(f"Original ExDark Image Dimensions: {img_orig.size[0]}x{img_orig.size[1]}")
print(f"Roboflow Resized Image Dimensions: {img_rf.size[0]}x{img_rf.size[1]}")

with open(txt_path, "r") as f:
    raw_ann = f.read()

print("\nRaw ExDark Annotation Content:")
print(raw_ann.strip())

# Parse GT for original image
gt_orig = parse_annotation_content(raw_ann, img_orig.size[0], img_orig.size[1], "2015_00026.jpg.txt")
print(f"\nGround Truth Bounding Box on Original ({img_orig.size[0]}x{img_orig.size[1]}):")
for g in gt_orig:
    print(f"  Class: {g['class']}, Box [x1, y1, x2, y2]: {g['bbox']}")

# Run D-FINE-S on Original Image across all 4 enhancement conditions
enhancer = EnhancementEngine(device='cpu')
detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

def calc_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
    return inter / union if union > 0 else 0.0

gt_box = gt_orig[0]['bbox']

print("\n" + "="*70)
print("EVALUATION ON GENUINE UNCROPPED EXDARK IMAGE (2015_00026.jpg)")
print("="*70)

methods = ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]
for m in methods:
    if m == "Original":
        enh = img_orig
    else:
        enh, _, _ = enhancer.enhance(img_orig, method=m)
    
    ann, dets, lat = detector.detect_objects(enh)
    res = compute_detailed_detection_metrics(dets, gt_orig, iou_threshold=0.50)
    
    # Find IoU with bicycle predictions
    bicy_dets = [d for d in dets if d['class'].lower() == 'bicycle']
    ious = [calc_iou(d['box'], gt_box) for d in bicy_dets]
    
    print(f"\nCondition: {m}")
    print(f"  Total D-FINE-S Detections: {len(dets)}")
    print(f"  Bicycle Predictions: {len(bicy_dets)}")
    for d, iou in zip(bicy_dets, ious):
        print(f"    - Score: {d['score']:.3f}, Predicted Box: {d['box']}, IoU with GT: {iou:.4f}")
    print(f"  mAP@50:    {res['mAP@50']:.4f}")
    print(f"  Precision: {res['precision']:.4f}")
    print(f"  Recall:    {res['recall']:.4f}")
    print(f"  F1-Score:  {res['f1_score']:.4f}")
    print(f"  Matching:  {res['tp']} TP, {res['fp']} FP, {res['fn']} FN")
