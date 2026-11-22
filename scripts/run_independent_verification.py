import sys
import os
import time
import cv2
import numpy as np
from PIL import Image

sys.path.append(r'C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement')
from inference.enhancement import EnhancementEngine
from inference.detection import DetectionEngine

enhancer = EnhancementEngine(device='cpu')
detector = DetectionEngine(conf_threshold=0.25, iou_threshold=0.45, device='cpu')

output_dir = r'C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement\outputs\verification'
os.makedirs(output_dir, exist_ok=True)

samples_dir = r'C:\Users\HP\.gemini\antigravity\scratch\Recognition_Aware_Low_Light_Image_Enhancement\data\samples\low'
sample_files = ['bicycle_extracted.png', 'sample_001.png', 'sample_002.png', 'sample_003.png', 'sample_004.png', 'sample_005.png']

print(f"Starting verification across {len(sample_files)} images...")
summary_records = []

for s_idx, fname in enumerate(sample_files):
    fpath = os.path.join(samples_dir, fname)
    pil_img = Image.open(fpath).convert('RGB')
    base_name = os.path.splitext(fname)[0]
    
    for method in ['Original', 'Zero-DCE', 'IGR-Net', 'Retinexformer']:
        t0 = time.time()
        if method == 'Original':
            enh_img = pil_img
            enh_lat = 0.0
        else:
            enh_img, enh_lat, meta = enhancer.enhance(pil_img, method=method)
        
        # Save enhanced image
        enh_save_name = f"{base_name}_{method.lower()}_enh.png"
        enh_img.save(os.path.join(output_dir, enh_save_name))
        
        # Detection
        ann_img, dets, det_lat = detector.detect_objects(enh_img)
        ann_save_name = f"{base_name}_{method.lower()}_det.png"
        ann_img.save(os.path.join(output_dir, ann_save_name))
        
        enh_np = np.array(enh_img)
        mean_lum = float(np.mean(enh_np))
        std_lum = float(np.std(enh_np))
        
        top_classes = [f"{d['class']}:{d['score']:.2f}" for d in dets[:3]]
        
        summary_records.append({
            'image': fname,
            'method': method,
            'mean': round(mean_lum, 1),
            'std': round(std_lum, 1),
            'num_dets': len(dets),
            'top_dets': ', '.join(top_classes) if top_classes else 'None',
            'det_lat_ms': round(det_lat, 1)
        })

print("\n" + "="*88)
header = f"{'Image':<22} | {'Method':<14} | {'Mean':<5} | {'Std':<5} | {'Dets':<5} | {'Top Detections'}"
print(header)
print("="*88)
for r in summary_records:
    line = f"{r['image']:<22} | {r['method']:<14} | {r['mean']:<5} | {r['std']:<5} | {r['num_dets']:<5} | {r['top_dets']}"
    print(line)
print("="*88)
