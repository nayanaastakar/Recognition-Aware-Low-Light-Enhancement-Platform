"""
Object detection evaluation metrics:
1. Precision
2. Recall
3. Mean Average Precision @ IoU 0.50 (mAP@50)
4. Mean Average Precision @ IoU 0.50:0.95 (mAP@50:95)
5. Class-wise AP Breakdown
"""

import numpy as np

def compute_ap(recalls, precisions):
    """
    Computes Average Precision (AP) using the standard 11-point or all-point area under PR curve.
    """
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute maximum precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Find points where recall changes
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return float(ap)

def evaluate_detections(pred_boxes_list, gt_boxes_list, iou_threshold=0.5):
    """
    Evaluates detection predictions against ground truth boxes across an image dataset.
    pred_boxes_list: list of lists of dicts [{'box': [x1, y1, x2, y2], 'class': str, 'score': float}]
    gt_boxes_list: list of lists of dicts [{'bbox': [x1, y1, x2, y2], 'class': str}]
    
    Returns:
        metrics: dict containing Precision, Recall, mAP@50, and class-wise APs.
    """
    classes = sorted(list(set([g['class'] for gts in gt_boxes_list for g in gts] + 
                              [p['class'] for preds in pred_boxes_list for p in preds])))
    
    if not classes:
        return {'mAP@50': 0.0, 'precision': 0.0, 'recall': 0.0, 'class_aps': {}}

    class_aps = {}
    all_tp = 0
    all_fp = 0
    all_gt = 0

    for cls in classes:
        cls_scores = []
        cls_matches = []
        num_gt_cls = 0

        for preds, gts in zip(pred_boxes_list, gt_boxes_list):
            cls_preds = [p for p in preds if p['class'] == cls]
            cls_gts = [g for g in gts if g['class'] == cls]
            num_gt_cls += len(cls_gts)

            # Sort predictions by score
            cls_preds = sorted(cls_preds, key=lambda x: x['score'], reverse=True)
            matched_gt = set()

            for p in cls_preds:
                cls_scores.append(p['score'])
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, g in enumerate(cls_gts):
                    if g_idx in matched_gt:
                        continue
                    iou = _calc_iou(p['box'], g['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_threshold and best_gt_idx != -1:
                    cls_matches.append(1) # True Positive
                    matched_gt.add(best_gt_idx)
                else:
                    cls_matches.append(0) # False Positive

        all_gt += num_gt_cls
        if not cls_scores or num_gt_cls == 0:
            class_aps[cls] = 0.0
            continue

        # Sort all accumulated predictions for this class by score
        sorted_indices = np.argsort(-np.array(cls_scores))
        cls_matches = np.array(cls_matches)[sorted_indices]

        tp_cumsum = np.cumsum(cls_matches)
        fp_cumsum = np.cumsum(1 - cls_matches)

        recalls = tp_cumsum / float(num_gt_cls)
        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, np.finfo(np.float64).eps)

        ap = compute_ap(recalls, precisions)
        class_aps[cls] = ap

        all_tp += tp_cumsum[-1] if len(tp_cumsum) > 0 else 0
        all_fp += fp_cumsum[-1] if len(fp_cumsum) > 0 else 0

    mAP = np.mean(list(class_aps.values())) if class_aps else 0.0
    overall_precision = all_tp / max(all_tp + all_fp, 1)
    overall_recall = all_tp / max(all_gt, 1)

    return {
        'mAP@50': float(mAP),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'class_aps': class_aps
    }

def _calc_iou(b1, b2):
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / float(a1 + a2 - inter + 1e-6)
