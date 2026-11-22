"""
Detection Inference Engine using D-FINE-S.
Handles:
- Loading D-FINE-S
- Normalization & inference
- Category mapping to ExDark and COCO
- Bounding box rendering
"""

import time
import torch
import numpy as np
from PIL import Image
from models.transforms import to_tensor
from models.dfine.detector import DFineDetector, EXDARK_TO_COCO_MAPPING, COCO_CLASSES

class DetectionEngine:
    def __init__(self, conf_threshold=0.25, iou_threshold=0.45, device='cpu'):
        self.device = device
        self.detector = DFineDetector(conf_threshold=conf_threshold, iou_threshold=iou_threshold, device=device)

    def detect_objects(self, pil_image):
        """
        Runs object detection on input PIL Image.
        Returns:
            annotated_image: PIL.Image with bounding boxes drawn
            detections: list of dicts [{'box': [x1, y1, x2, y2], 'class': str, 'score': float}]
            latency_ms: float
        """
        orig_w, orig_h = pil_image.size
        img_tensor = to_tensor(pil_image).unsqueeze(0)
        
        start_time = time.time()
        detections = self.detector.detect(img_tensor, original_shape=(orig_h, orig_w))
        latency_ms = (time.time() - start_time) * 1000.0

        # Draw detections on numpy array
        np_img = np.array(pil_image)
        rendered_np = self.detector.draw_detections(np_img, detections)
        annotated_pil = Image.fromarray(rendered_np)

        return annotated_pil, detections, latency_ms
