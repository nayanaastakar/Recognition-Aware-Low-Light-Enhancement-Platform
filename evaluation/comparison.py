"""
Central Recognition-Aware Evaluation Pipeline.
Compares Condition A (Original), Condition B (Zero-DCE), Condition C (IGR-Net), Condition D (Retinexformer)
under the exact same D-FINE-S detector configuration.
"""

import os
import time
import pandas as pd
import numpy as np
from PIL import Image

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.enhancement_metrics import calculate_psnr, calculate_ssim
from evaluation.detection_metrics import evaluate_detections

class RecognitionAwareEvaluator:
    def __init__(self, device='cpu'):
        self.device = device
        self.enhancer = EnhancementEngine(device=device)
        self.detector = DetectionEngine(device=device)

    def evaluate_pipeline(self, test_samples, ground_truth_boxes=None, high_references=None):
        """
        Runs comprehensive comparative benchmark across all 4 conditions:
        - Condition A: Original Low-Light
        - Condition B: Zero-DCE Enhanced
        - Condition C: IGR-Net Enhanced (Proposed)
        - Condition D: Retinexformer Enhanced

        Returns:
            summary_df: DataFrame of aggregate performance metrics
            detailed_results: dictionary containing per-sample visualizations & metrics
        """
        conditions = ['Original', 'Zero-DCE', 'IGR-Net', 'Retinexformer']
        results = {cond: {'psnr': [], 'ssim': [], 'latency_ms': [], 'predictions': []} for cond in conditions}

        for idx, low_img in enumerate(test_samples):
            high_ref = high_references[idx] if high_references else None
            
            for cond in conditions:
                # 1. Enhance
                enh_img, enh_lat, _ = self.enhancer.enhance(low_img, method=cond)
                
                # 2. Image Quality Metrics (if paired reference available)
                if high_ref is not None:
                    p = calculate_psnr(np.array(enh_img)/255.0, np.array(high_ref)/255.0)
                    s = calculate_ssim(np.array(enh_img)/255.0, np.array(high_ref)/255.0)
                    results[cond]['psnr'].append(p)
                    results[cond]['ssim'].append(s)

                # 3. Object Detection with D-FINE-S
                annotated_img, preds, det_lat = self.detector.detect_objects(enh_img)
                results[cond]['predictions'].append(preds)
                results[cond]['latency_ms'].append(enh_lat + det_lat)

        # Compute Detection Metrics across conditions
        summary_rows = []
        for cond in conditions:
            preds_list = results[cond]['predictions']
            
            if ground_truth_boxes is not None:
                det_metrics = evaluate_detections(preds_list, ground_truth_boxes)
            else:
                det_metrics = {'mAP@50': 0.0, 'precision': 0.0, 'recall': 0.0}

            avg_psnr = np.mean(results[cond]['psnr']) if results[cond]['psnr'] else None
            avg_ssim = np.mean(results[cond]['ssim']) if results[cond]['ssim'] else None
            avg_latency = np.mean(results[cond]['latency_ms']) if results[cond]['latency_ms'] else 0.0

            summary_rows.append({
                'Condition': cond,
                'PSNR (dB)': f"{avg_psnr:.2f}" if avg_psnr is not None else "N/A (Unpaired)",
                'SSIM': f"{avg_ssim:.4f}" if avg_ssim is not None else "N/A (Unpaired)",
                'Precision': f"{det_metrics['precision']:.3f}",
                'Recall': f"{det_metrics['recall']:.3f}",
                'mAP@50': f"{det_metrics['mAP@50']:.3f}",
                'Avg Latency (ms)': f"{avg_latency:.1f}"
            })

        summary_df = pd.DataFrame(summary_rows)
        return summary_df, results
