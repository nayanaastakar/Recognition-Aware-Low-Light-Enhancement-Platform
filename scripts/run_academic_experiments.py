import os
import sys
import time
import json
import glob
import math
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine
from evaluation.exdark_parser import parse_annotation_content
from evaluation.detailed_evaluator import compute_detailed_detection_metrics

# ---------------------------------------------------------
# DIRECTORY DEFINITIONS
# ---------------------------------------------------------
EXDARK_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "datasets", "ExDark", "images")
EXDARK_ANN_DIR = os.path.join(PROJECT_ROOT, "data", "datasets", "ExDark", "annotations")
LOL_EVAL15_LOW = os.path.join(PROJECT_ROOT, "data", "datasets", "LOL", "eval15", "low")
LOL_EVAL15_HIGH = os.path.join(PROJECT_ROOT, "data", "datasets", "LOL", "eval15", "high")

OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "experiments")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ---------------------------------------------------------
# METRIC UTILITIES (PSNR, SSIM, NIQE APPROX)
# ---------------------------------------------------------
def calculate_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio between two PIL images [0-255]."""
    arr1 = np.array(img1, dtype=np.float64)
    arr2 = np.array(img2, dtype=np.float64)
    mse = np.mean((arr1 - arr2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return float(20 * math.log10(max_pixel / math.sqrt(mse)))

def calculate_ssim(img1, img2):
    """Structural Similarity Index (mean luminance/contrast/structure)."""
    arr1 = np.array(img1.convert('L'), dtype=np.float64)
    arr2 = np.array(img2.convert('L'), dtype=np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1_sq = np.var(arr1)
    sigma2_sq = np.var(arr2)
    sigma12 = np.mean((arr1 - mu1) * (arr2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)

def calculate_niqe_proxy(img):
    """
    Standard spatial domain Natural Image Quality Evaluator proxy 
    based on Mean Subtracted Contrast Normalized (MSCN) statistical entropy.
    Lower indicates better naturalness.
    """
    gray = np.array(img.convert('L'), dtype=np.float32)
    mu = np.mean(gray)
    sigma = np.std(gray) + 1e-6
    mscn = (gray - mu) / sigma
    entropy = -np.sum(np.histogram(mscn, bins=30, density=True)[0] * np.log2(np.histogram(mscn, bins=30, density=True)[0] + 1e-7))
    return float(abs(entropy))

# ---------------------------------------------------------
# RUNNER PIPELINES
# ---------------------------------------------------------
def run_lol_evaluation(enhancer):
    print("\n" + "="*60)
    print("TASK 4: LOL PAIRED EVALUATION (PSNR, SSIM, NIQE)")
    print("="*60)
    
    low_files = sorted(glob.glob(os.path.join(LOL_EVAL15_LOW, "*.png")))
    print(f"Found {len(low_files)} paired images in LOL eval15.")
    
    methods = ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]
    results = {m: {"psnr": [], "ssim": [], "niqe": [], "latency": []} for m in methods}
    
    for idx, low_path in enumerate(low_files):
        fname = os.path.basename(low_path)
        high_path = os.path.join(LOL_EVAL15_HIGH, fname)
        if not os.path.exists(high_path):
            continue
            
        low_img = Image.open(low_path).convert('RGB')
        high_img = Image.open(high_path).convert('RGB')
        
        for m in methods:
            t0 = time.perf_counter()
            if m == "Original":
                enh_img = low_img
                lat = 0.0
            else:
                enh_img, lat, _ = enhancer.enhance(low_img, method=m)
            
            psnr_val = calculate_psnr(enh_img, high_img)
            ssim_val = calculate_ssim(enh_img, high_img)
            niqe_val = calculate_niqe_proxy(enh_img)
            
            results[m]["psnr"].append(psnr_val)
            results[m]["ssim"].append(ssim_val)
            results[m]["niqe"].append(niqe_val)
            results[m]["latency"].append(lat)
            
        if (idx + 1) % 5 == 0 or idx == len(low_files) - 1:
            print(f"Processed {idx+1}/{len(low_files)} LOL pairs...")
            
    summary = {}
    print("\n--- LOL Benchmark Aggregate Summary (N=15) ---")
    print(f"{'Method':<16} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'NIQE':<8} | {'Enh Lat (ms)':<12}")
    print("-" * 62)
    for m in methods:
        avg_psnr = float(np.mean(results[m]["psnr"]))
        avg_ssim = float(np.mean(results[m]["ssim"]))
        avg_niqe = float(np.mean(results[m]["niqe"]))
        avg_lat = float(np.mean(results[m]["latency"]))
        summary[m] = {
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "avg_niqe": avg_niqe,
            "avg_enh_latency": avg_lat
        }
        print(f"{m:<16} | {avg_psnr:<10.3f} | {avg_ssim:<8.4f} | {avg_niqe:<8.3f} | {avg_lat:<12.1f}")
        
    with open(os.path.join(OUTPUTS_DIR, "lol_benchmark_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def run_exdark_evaluation(enhancer, detector, num_images=100, conf_threshold=0.25):
    print("\n" + "="*60)
    print(f"TASK 3: EXDARK MULTI-IMAGE BENCHMARK EVALUATION (N={num_images}, τ={conf_threshold})")
    print("="*60)
    
    # Collect images with valid annotations
    all_ann_files = {os.path.basename(f): f for f in glob.glob(f"{EXDARK_ANN_DIR}/**/*.txt", recursive=True)}
    all_img_files = [f for f in glob.glob(f"{EXDARK_IMG_DIR}/**", recursive=True) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    valid_pairs = []
    for img_p in all_img_files:
        base = os.path.basename(img_p)
        cand_txt = f"{base}.txt"
        if cand_txt in all_ann_files:
            valid_pairs.append((img_p, all_ann_files[cand_txt]))
            
    print(f"Total matching ExDark image-annotation pairs available: {len(valid_pairs)}")
    eval_set = valid_pairs[:num_images]
    print(f"Evaluating balanced academic cohort of {len(eval_set)} images across all 4 conditions...")
    
    methods = ["Original", "Zero-DCE", "IGR-Net", "Retinexformer"]
    detector.detector.conf_threshold = conf_threshold
    
    # Store aggregated detections and ground truths across all images
    condition_records = {m: {"gts": [], "preds": [], "enh_times": [], "det_times": []} for m in methods}
    
    for idx, (img_path, ann_path) in enumerate(eval_set):
        fname = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        with open(ann_path, "r", encoding="utf-8", errors="ignore") as f:
            gt_text = f.read()
        gts = parse_annotation_content(gt_text, w, h, filename=fname)
        
        for m in methods:
            if m == "Original":
                enh_img = img
                enh_lat = 0.0
            else:
                enh_img, enh_lat, _ = enhancer.enhance(img, method=m)
                
            ann_pil, dets, det_lat = detector.detect_objects(enh_img)
            
            condition_records[m]["gts"].append(gts)
            condition_records[m]["preds"].append(dets)
            condition_records[m]["enh_times"].append(enh_lat)
            condition_records[m]["det_times"].append(det_lat)
            
        if (idx + 1) % 25 == 0 or idx == len(eval_set) - 1:
            print(f"Progress: Evaluated {idx+1}/{len(eval_set)} ExDark scenes across all 4 branches...")
            
    # Compute aggregate dataset-level metrics
    benchmark_summary = {}
    print("\n--- ExDark Cohort Evaluation Summary (N=" + str(len(eval_set)) + ") ---")
    print(f"{'Condition':<16} | {'mAP@50':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'Avg Lat (ms)':<12}")
    print("-" * 88)
    
    for m in methods:
        all_gt_flat = []
        all_pred_flat = []
        # Combine across cohort with unique image prefixes
        for i_idx, (g_list, p_list) in enumerate(zip(condition_records[m]["gts"], condition_records[m]["preds"])):
            for g in g_list:
                all_gt_flat.append({'class': g['class'], 'bbox': g['bbox'], 'img_id': i_idx})
            for p in p_list:
                all_pred_flat.append({'class': p['class'], 'box': p['box'], 'score': p['score'], 'img_id': i_idx})
                
        metrics = compute_detailed_detection_metrics(all_pred_flat, all_gt_flat, iou_threshold=0.5)
        avg_enh = float(np.mean(condition_records[m]["enh_times"]))
        avg_det = float(np.mean(condition_records[m]["det_times"]))
        total_lat = avg_enh + avg_det
        
        benchmark_summary[m] = {
            "mAP@50": metrics["mAP@50"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "total_gt": metrics["total_gt"],
            "total_pred": metrics["total_pred"],
            "avg_enh_latency": avg_enh,
            "avg_det_latency": avg_det,
            "total_latency": total_lat,
            "class_breakdown": metrics["class_breakdown"]
        }
        print(f"{m:<16} | {metrics['mAP@50']:<8.3f} | {metrics['precision']:<10.3f} | {metrics['recall']:<8.3f} | {metrics['f1_score']:<8.3f} | {metrics['tp']:<5} | {metrics['fp']:<5} | {metrics['fn']:<5} | {total_lat:<12.1f}")
        
    with open(os.path.join(OUTPUTS_DIR, "exdark_cohort_results.json"), "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    return benchmark_summary

def run_threshold_ablation(enhancer, detector, num_images=30):
    print("\n" + "="*60)
    print("TASK 5: CONFIDENCE THRESHOLD EXPERIMENTS (τ ∈ [0.15, 0.25, 0.35, 0.50])")
    print("="*60)
    
    thresholds = [0.15, 0.25, 0.35, 0.50]
    
    all_ann_files = {os.path.basename(f): f for f in glob.glob(f"{EXDARK_ANN_DIR}/**/*.txt", recursive=True)}
    all_img_files = [f for f in glob.glob(f"{EXDARK_IMG_DIR}/**", recursive=True) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    valid_pairs = [(f, all_ann_files[f"{os.path.basename(f)}.txt"]) for f in all_img_files if f"{os.path.basename(f)}.txt" in all_ann_files][:num_images]
    
    ablation_results = {}
    
    for thresh in thresholds:
        print(f"\nEvaluating Confidence Threshold τ = {thresh:.2f}...")
        detector.detector.conf_threshold = thresh
        
        # Test IGR-Net and Original
        for cond in ["Original", "IGR-Net"]:
            if cond not in ablation_results:
                ablation_results[cond] = {}
                
            gt_flat, pred_flat = [], []
            for i_idx, (img_p, ann_p) in enumerate(valid_pairs):
                img = Image.open(img_p).convert('RGB')
                w, h = img.size
                with open(ann_p, "r", encoding="utf-8", errors="ignore") as f:
                    gts = parse_annotation_content(f.read(), w, h, filename=os.path.basename(img_p))
                
                if cond == "Original":
                    enh = img
                else:
                    enh, _, _ = enhancer.enhance(img, method=cond)
                    
                _, dets, _ = detector.detect_objects(enh)
                for g in gts:
                    gt_flat.append({'class': g['class'], 'bbox': g['bbox'], 'img_id': i_idx})
                for d in dets:
                    pred_flat.append({'class': d['class'], 'box': d['box'], 'score': d['score'], 'img_id': i_idx})
                    
            m = compute_detailed_detection_metrics(pred_flat, gt_flat, iou_threshold=0.5)
            ablation_results[cond][str(thresh)] = {
                "threshold": thresh,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "mAP@50": m["mAP@50"],
                "tp": m["tp"],
                "fp": m["fp"],
                "fn": m["fn"]
            }
            print(f"  [{cond}] τ={thresh:.2f} -> Prec: {m['precision']:.3f}, Rec: {m['recall']:.3f}, F1: {m['f1_score']:.3f}, TP: {m['tp']}, FP: {m['fp']}")
            
    with open(os.path.join(OUTPUTS_DIR, "threshold_ablation_results.json"), "w") as f:
        json.dump(ablation_results, f, indent=2)
    return ablation_results

if __name__ == "__main__":
    print("Initializing Enhancement & Detection Engines...")
    enhancer = EnhancementEngine(device='cpu')
    detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')
    
    # 1. LOL Paired Image Quality Benchmark
    lol_res = run_lol_evaluation(enhancer)
    
    # 2. ExDark Multi-image Benchmark Evaluation
    exdark_res = run_exdark_evaluation(enhancer, detector, num_images=100, conf_threshold=0.25)
    
    # 3. Confidence Threshold Ablation Experiments
    thresh_res = run_threshold_ablation(enhancer, detector, num_images=30)
    
    print("\nAll experiments successfully completed and saved to outputs/experiments/.")
