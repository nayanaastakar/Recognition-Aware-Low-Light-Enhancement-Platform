import numpy as np

def compute_detailed_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Computes rigorous class-aware detection metrics for single-image or multi-image evaluation.
    
    Args:
        predictions: list of dicts [{'box': [x1, y1, x2, y2], 'class': str, 'score': float}]
        ground_truths: list of dicts [{'bbox': [x1, y1, x2, y2], 'class': str}]
        iou_threshold: float, IoU matching threshold (default 0.5 for mAP@50)
        
    Returns:
        dict:
            'mAP@50': float,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'total_gt': int,
            'total_pred': int,
            'tp': int,
            'fp': int,
            'fn': int,
            'matches': list of match records,
            'class_breakdown': dict {cls: {'gt': int, 'pred': int, 'tp': int, 'fp': int, 'fn': int, 'ap': float, 'precision': float, 'recall': float}}
    """
    def calc_iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        boxA_area = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        boxB_area = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        union_area = float(boxA_area + boxB_area - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    # Synonym mapping between COCO predictions and ExDark annotations
    SYNONYM_MAP = {
        'chair': 'chair',
        'bench': 'chair',  # COCO 'bench' corresponds to ExDark 'chair' category
    }

    def canonical_class(c):
        cl = c.lower().strip()
        return SYNONYM_MAP.get(cl, cl)

    # Collect all unique classes
    all_classes = sorted(list(set(
        [canonical_class(g['class']) for g in ground_truths] + 
        [canonical_class(p['class']) for p in predictions]
    )))

    if not all_classes:
        return {
            'mAP@50': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
            'total_gt': 0, 'total_pred': 0, 'tp': 0, 'fp': 0, 'fn': 0,
            'matches': [],
            'class_breakdown': {}
        }

    total_tp = 0
    total_fp = 0
    total_fn = 0
    class_breakdown = {}
    class_aps = []
    matches = []

    for cls in all_classes:
        cls_preds = sorted([p for p in predictions if canonical_class(p['class']) == cls], key=lambda x: x['score'], reverse=True)
        cls_gts = [g for g in ground_truths if canonical_class(g['class']) == cls]
        
        num_gt = len(cls_gts)
        num_pred = len(cls_preds)

        matched_gts = set()
        cls_tp = 0
        cls_fp = 0
        precisions = []
        recalls = []

        for p_idx, p in enumerate(cls_preds):
            best_iou = 0.0
            best_gt_idx = -1

            for g_idx, g in enumerate(cls_gts):
                if g_idx in matched_gts:
                    continue
                iou = calc_iou(p['box'], g['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                cls_tp += 1
                matched_gts.add(best_gt_idx)
                matches.append({
                    'class': cls.capitalize(),
                    'gt_box': str(cls_gts[best_gt_idx]['bbox']),
                    'pred_box': str(p['box']),
                    'iou': f"{best_iou:.3f}",
                    'confidence': f"{p['score']:.3f}",
                    'status': '✅ TP (Matched)'
                })
            else:
                cls_fp += 1
                matches.append({
                    'class': cls.capitalize(),
                    'gt_box': '-',
                    'pred_box': str(p['box']),
                    'iou': f"{best_iou:.3f}" if best_iou > 0 else "0.000",
                    'confidence': f"{p['score']:.3f}",
                    'status': '⚠️ FP (False Positive)'
                })

            # Cumulative P & R
            cur_p = cls_tp / (p_idx + 1)
            cur_r = cls_tp / num_gt if num_gt > 0 else 0.0
            precisions.append(cur_p)
            recalls.append(cur_r)

        # Unmatched GTs are False Negatives
        for g_idx, g in enumerate(cls_gts):
            if g_idx not in matched_gts:
                matches.append({
                    'class': cls.capitalize(),
                    'gt_box': str(g['bbox']),
                    'pred_box': '-',
                    'iou': '0.000',
                    'confidence': '-',
                    'status': '❌ FN (Missed Target)'
                })

        cls_fn = num_gt - len(matched_gts)
        total_tp += cls_tp
        total_fp += cls_fp
        total_fn += cls_fn

        # Calculate AP via PR curve area
        if num_gt == 0:
            ap = 0.0
        elif len(precisions) == 0:
            ap = 0.0
        else:
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            indices = np.where(mrec[1:] != mrec[:-1])[0]
            ap = float(np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1]))

        if num_gt > 0:
            class_aps.append(ap)

        cls_precision = cls_tp / num_pred if num_pred > 0 else (1.0 if num_gt == 0 else 0.0)
        cls_recall = cls_tp / num_gt if num_gt > 0 else 0.0

        class_breakdown[cls] = {
            'gt': num_gt,
            'pred': num_pred,
            'tp': cls_tp,
            'fp': cls_fp,
            'fn': cls_fn,
            'precision': float(cls_precision),
            'recall': float(cls_recall),
            'ap': float(ap)
        }

    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_tp + total_fn, 1)
    if (overall_precision + overall_recall) > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    mean_ap = float(np.mean(class_aps)) if class_aps else 0.0

    return {
        'mAP@50': mean_ap,
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1_score': float(overall_f1),
        'total_gt': len(ground_truths),
        'total_pred': len(predictions),
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'matches': matches,
        'class_breakdown': class_breakdown
    }
