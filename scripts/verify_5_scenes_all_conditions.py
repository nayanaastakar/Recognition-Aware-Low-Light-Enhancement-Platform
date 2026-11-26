import os
import sys
import glob
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics
from evaluation.annotation_indexer import AnnotationIndexer

def run_test():
    indexer = AnnotationIndexer(PROJECT_ROOT)
    enhancer = EnhancementEngine(device='cpu')
    detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

    test_images = [
        "yellow_bicycle_bench.png",
        "2015_00005.jpg",
        "2015_00008.jpg",
        "2015_00026.jpg",
        "2015_02446.jpg"
    ]

    print("=" * 80)
    print("VERIFICATION OF 5 REAL EXDARK IMAGES ACROSS ALL 4 CONDITIONS")
    print("=" * 80)

    for img_name in test_images:
        # Search path
        matches = glob.glob(f"{PROJECT_ROOT}/**/{img_name}", recursive=True)
        if not matches:
            print(f"Skipping {img_name}, file not found.")
            continue
        img_path = matches[0]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Lookup annotation
        ann_path, via, warn = indexer.lookup(img_name)
        if not ann_path or not os.path.exists(ann_path):
            print(f"No annotation for {img_name}")
            continue

        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            gt_text = f.read()

        gts = parse_annotation_content(gt_text, w, h, filename=img_name)
        print(f"\n--- Scene: {img_name} ({w}x{h} px) | GT Targets: {len(gts)} ({os.path.basename(ann_path)}) ---")

        for cond in ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]:
            if cond == "Original":
                enh_img = img
            else:
                enh_img, _, _ = enhancer.enhance(img, method=cond)

            _, dets, _ = detector.detect_objects(enh_img)
            metrics = compute_detailed_detection_metrics(dets, gts, iou_threshold=0.5)
            
            # Print summary and TP IoUs
            tp_matches = [f"{m['class']} (IoU={m['iou']})" for m in metrics['matches'] if 'TP' in m['status']]
            tp_str = ", ".join(tp_matches) if tp_matches else "None"
            print(f"  [{cond:<13}] mAP@50: {metrics['mAP@50']:.3f} | Rec: {metrics['recall']:.3f} | {metrics['tp']} TP, {metrics['fp']} FP | Matches: {tp_str}")

if __name__ == "__main__":
    run_test()
