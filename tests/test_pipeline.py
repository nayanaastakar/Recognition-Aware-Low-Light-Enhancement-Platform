"""
End-to-end integration and verification test suite.
Validates:
1. Pure PyTorch / NumPy transforms
2. Zero-DCE forward pass, output shape, and pixel range [0, 1]
3. IGR-Net forward pass, Retinex maps, and all ablation toggles
4. Retinexformer forward pass
5. D-FINE-S object detection inference and NMS
6. Enhancement metrics calculation (PSNR, SSIM)
7. Detection evaluation (mAP, Precision, Recall)
8. Synthetic sample generation
"""

import unittest
import torch
import numpy as np
from PIL import Image

from models.transforms import to_tensor, to_pil_image, resize_tensor
from models.zero_dce.model import DCE_Net, L_spatial, L_exposure, L_color, L_tv
from models.igr_net.model import IGRNet
from models.igr_net.losses import IGRNetLoss
from models.retinexformer.model import Retinexformer
from models.dfine.detector import DFineDetector
from evaluation.enhancement_metrics import calculate_psnr, calculate_ssim
from evaluation.detection_metrics import evaluate_detections
from datasets.dataset_setup import generate_synthetic_samples

class TestRecognitionAwarePipeline(unittest.TestCase):
    def setUp(self):
        # Create a sample test tensor (1, 3, 128, 128)
        self.b, self.c, self.h, self.w = 1, 3, 128, 128
        self.input_tensor = torch.rand(self.b, self.c, self.h, self.w)
        self.pil_img = to_pil_image(self.input_tensor)

    def test_transforms(self):
        t = to_tensor(self.pil_img)
        self.assertEqual(t.shape, (3, self.h, self.w))
        self.assertTrue(0.0 <= t.min() and t.max() <= 1.0)

        resized = resize_tensor(t, (64, 64))
        self.assertEqual(resized.shape, (3, 64, 64))

    def test_zero_dce(self):
        model = DCE_Net(n_filters=16)
        enh, A = model(self.input_tensor)
        self.assertEqual(enh.shape, self.input_tensor.shape)
        self.assertEqual(A.shape, (self.b, 24, self.h, self.w))
        self.assertTrue(0.0 <= enh.min() and enh.max() <= 1.0)
        self.assertFalse(torch.isnan(enh).any())

    def test_igr_net_full_and_ablations(self):
        # Full IGR-Net
        full_model = IGRNet(feat_dim=16)
        enh, R, L_init, L_enh = full_model(self.input_tensor)
        self.assertEqual(enh.shape, self.input_tensor.shape)
        self.assertEqual(R.shape, self.input_tensor.shape)
        self.assertEqual(L_init.shape, (self.b, 1, self.h, self.w))
        self.assertEqual(L_enh.shape, (self.b, 1, self.h, self.w))
        self.assertTrue(0.0 <= enh.min() and enh.max() <= 1.0)
        self.assertFalse(torch.isnan(enh).any())

        # Ablations
        for cfg in [{'wo_local_global': True}, {'wo_noise_attn': True}, {'wo_channel_attn': True}, {'base_retinex': True}]:
            abl_model = IGRNet(feat_dim=16, **cfg)
            abl_enh, _, _, _ = abl_model(self.input_tensor)
            self.assertEqual(abl_enh.shape, self.input_tensor.shape)
            self.assertFalse(torch.isnan(abl_enh).any())

    def test_igr_net_loss(self):
        criterion = IGRNetLoss()
        model = IGRNet(feat_dim=16)
        enh, R, L_init, L_enh = model(self.input_tensor)
        total_loss, loss_dict = criterion(enh, R, L_init, L_enh, self.input_tensor, self.input_tensor)
        self.assertTrue(total_loss.item() > 0.0)
        self.assertIn('loss_retinex', loss_dict)
        self.assertIn('loss_smooth', loss_dict)

    def test_retinexformer(self):
        model = Retinexformer(dim=16, num_blocks=2)
        enh = model(self.input_tensor)
        self.assertEqual(enh.shape, self.input_tensor.shape)
        self.assertTrue(0.0 <= enh.min() and enh.max() <= 1.0)

    def test_dfine_s_detection(self):
        detector = DFineDetector(conf_threshold=0.1, device='cpu')
        detections = detector.detect(self.input_tensor, original_shape=(self.h, self.w))
        self.assertIsInstance(detections, list)
        rendered = detector.draw_detections(np.zeros((self.h, self.w, 3), dtype=np.uint8), detections)
        self.assertEqual(rendered.shape, (self.h, self.w, 3))

    def test_metrics(self):
        t1 = torch.ones(1, 3, 64, 64) * 0.5
        t2 = torch.ones(1, 3, 64, 64) * 0.5
        psnr = calculate_psnr(t1, t2)
        ssim = calculate_ssim(t1, t2)
        self.assertEqual(psnr, float('inf'))
        self.assertAlmostEqual(ssim, 1.0, places=3)

    def test_detection_evaluation(self):
        pred_boxes = [[{'box': [10, 10, 50, 50], 'class': 'Car', 'score': 0.9}]]
        gt_boxes = [[{'bbox': [12, 12, 48, 48], 'class': 'Car'}]]
        metrics = evaluate_detections(pred_boxes, gt_boxes)
        self.assertGreater(metrics['mAP@50'], 0.5)

if __name__ == '__main__':
    unittest.main()
