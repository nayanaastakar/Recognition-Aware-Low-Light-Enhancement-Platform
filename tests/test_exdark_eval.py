import sys
import os

PROJECT_ROOT = r"C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

# Test case 1: ExDark format
sample_txt = """
Bicycle 100 150 200 150 0 0 0 0 0 0
Car 10 20 50 60 0 0 0 0 0 0
"""
gts = parse_annotation_content(sample_txt, 500, 500)
print("Parsed GTs:", gts)

preds = [
    {'box': [100, 150, 300, 300], 'class': 'bicycle', 'score': 0.95},
    {'box': [15, 25, 60, 80], 'class': 'car', 'score': 0.85},
    {'box': [300, 300, 400, 400], 'class': 'bicycle', 'score': 0.40}
]

metrics = compute_detailed_detection_metrics(preds, gts, iou_threshold=0.5)
print("mAP@50:", metrics['mAP@50'])
print("Precision:", metrics['precision'])
print("Recall:", metrics['recall'])
print("F1-score:", metrics['f1_score'])
print("TP / FP / FN:", metrics['tp'], metrics['fp'], metrics['fn'])
print("Class breakdown:", metrics['class_breakdown'])
