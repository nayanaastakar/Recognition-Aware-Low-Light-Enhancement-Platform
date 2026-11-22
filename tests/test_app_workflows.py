"""
Automated End-to-End Application Workflow Tester.
Tests:
A. Upload image -> Enhance -> Detect -> Export.
B. Synthetic Webcam frame -> Enhance -> Detect.
C. 4-Way Model Comparison (Original, Zero-DCE, IGR-Net, Retinexformer).
D. Batch Processing and CSV generation.
"""

import os
import sys
import csv
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine

def test_all_application_workflows():
    print("[*] Initializing Application Engines...")
    enhancer = EnhancementEngine(device='cpu')
    detector = DetectionEngine(device='cpu')

    sample_img_path = os.path.join(root_dir, 'data', 'samples', 'low', 'sample_001.png')
    if not os.path.exists(sample_img_path):
        from datasets.dataset_setup import generate_synthetic_samples
        generate_synthetic_samples(os.path.join(root_dir, 'data', 'samples'), num_samples=3)

    img = Image.open(sample_img_path).convert('RGB')
    print(f"[+] Loaded input test image: {sample_img_path} ({img.width}x{img.height})")

    # Workflow A: Enhance -> Detect
    print("\n--- Testing Workflow A: Enhance -> Detect ---")
    enh_pil, enh_lat, meta = enhancer.enhance(img, method='IGR-Net')
    print(f"  [+] Enhanced with IGR-Net in {enh_lat:.1f}ms. Output Size: {enh_pil.size}")
    ann_pil, dets, det_lat = detector.detect_objects(enh_pil)
    print(f"  [+] D-FINE-S detected {len(dets)} objects in {det_lat:.1f}ms.")
    for d in dets[:3]:
        print(f"      - {d['class']} (conf: {d['score']:.2f}, box: {d['box']})")

    # Workflow B: Simulated Webcam Frame
    print("\n--- Testing Workflow B: Webcam Frame Processing ---")
    webcam_frame = img.copy() # Represents frame from camera
    cam_enh, cam_lat, _ = enhancer.enhance(webcam_frame, method='Zero-DCE')
    cam_ann, cam_dets, _ = detector.detect_objects(cam_enh)
    print(f"  [+] Webcam Frame processed successfully. Latency: {cam_lat:.1f}ms, Detections: {len(cam_dets)}")

    # Workflow C: 4-Way Model Comparison
    print("\n--- Testing Workflow C: 4-Way Comparison ---")
    for method in ['Original', 'Zero-DCE', 'IGR-Net', 'Retinexformer']:
        e_img, e_lat, _ = enhancer.enhance(img, method=method)
        _, d_list, _ = detector.detect_objects(e_img)
        print(f"  [+] Condition {method:15s} | Enhance Latency: {e_lat:5.1f}ms | Detections: {len(d_list)}")

    # Workflow D: Batch Processing & CSV Export
    print("\n--- Testing Workflow D: Batch Processing & CSV Export ---")
    sample_files = [os.path.join(root_dir, 'data', 'samples', 'low', f) 
                    for f in os.listdir(os.path.join(root_dir, 'data', 'samples', 'low')) if f.endswith('.png')][:3]
    records = []
    for fpath in sample_files:
        p_img = Image.open(fpath).convert('RGB')
        e_img, e_lat, _ = enhancer.enhance(p_img, method='IGR-Net')
        _, d_list, d_lat = detector.detect_objects(e_img)
        records.append({
            'Filename': os.path.basename(fpath),
            'Enhance_Latency_ms': round(e_lat, 2),
            'Detect_Latency_ms': round(d_lat, 2),
            'Num_Detections': len(d_list)
        })
    # Export to CSV via standard csv module
    out_csv = os.path.join(root_dir, 'results', 'tables', 'test_batch_export.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Filename', 'Enhance_Latency_ms', 'Detect_Latency_ms', 'Num_Detections'])
        writer.writeheader()
        writer.writerows(records)
    print(f"  [+] Batch processing completed. Exported CSV to: {out_csv}")
    for r in records:
        print(f"      {r}")
    print("\n[SUCCESS] All application workflows executed and verified without errors!")

if __name__ == '__main__':
    test_all_application_workflows()
