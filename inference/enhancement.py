"""
Unified Image Enhancement Inference Engine.
Wraps:
- Zero-DCE
- IGR-Net (Full & Ablations)
- Retinexformer
- Classical WREN-inspired Enhancement (Illumination map estimation via Max-RGB + guided filter)
"""

import os
import time
import torch
import numpy as np
from PIL import Image
from models.transforms import to_tensor, to_pil_image
from models.zero_dce.model import DCE_Net
from models.igr_net.model import IGRNet
from models.retinexformer.model import Retinexformer

class EnhancementEngine:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self._init_models()

    def _init_models(self):
        # 1. Zero-DCE
        self.zero_dce = DCE_Net().to(self.device)
        dce_ckpt = os.path.join(self.root_dir, 'checkpoints', 'zero_dce_model.pth')
        if os.path.exists(dce_ckpt):
            self.load_checkpoint('zero_dce', dce_ckpt)
        self.zero_dce.eval()

        # 2. IGR-Net (Proposed)
        self.igr_net = IGRNet().to(self.device)
        igr_ckpt = os.path.join(self.root_dir, 'checkpoints', 'igr_net_best.pth')
        if os.path.exists(igr_ckpt):
            self.load_checkpoint('igr_net', igr_ckpt)
        self.igr_net.eval()

        # 3. Retinexformer
        self.retinexformer = Retinexformer().to(self.device)
        self.retinexformer.eval()

    def load_checkpoint(self, model_name, checkpoint_path):
        """Loads weights from disk if available."""
        if not torch.cuda.is_available() and self.device.type == 'cpu':
            ckpt = torch.load(checkpoint_path, map_location='cpu')
        else:
            ckpt = torch.load(checkpoint_path)

        state_dict = ckpt.get('state_dict', ckpt)
        if model_name.lower() == 'zero_dce':
            self.zero_dce.load_state_dict(state_dict, strict=False)
        elif 'igr' in model_name.lower():
            self.igr_net.load_state_dict(state_dict, strict=False)
        elif 'retinexformer' in model_name.lower():
            self.retinexformer.load_state_dict(state_dict, strict=False)

    def enhance(self, pil_image, method='IGR-Net', ablation_cfg=None):
        # Prevent CPU out-of-memory on large uploaded photos by bounding max dimension to 800
        max_dim = max(pil_image.size)
        if max_dim > 800:
            scale = 800.0 / max_dim
            new_size = (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.BILINEAR)
        """
        Enhances a PIL Image using the selected method.
        Returns:
            enhanced_pil: PIL.Image
            latency_ms: float
            metadata: dict containing auxiliary maps (reflectance, illumination, etc.)
        """
        img_tensor = to_tensor(pil_image).unsqueeze(0).to(self.device)
        start_time = time.time()
        metadata = {'method': method}

        with torch.no_grad():
            if method.lower() == 'original':
                enhanced_tensor = img_tensor
            elif method.lower() == 'zero-dce':
                enhanced_tensor, curve_maps = self.zero_dce(img_tensor)
                metadata['num_iterations'] = 8
            elif method.lower() == 'igr-net':
                if ablation_cfg:
                    # Temporary ablation model
                    model = IGRNet(**ablation_cfg).to(self.device)
                    model.eval()
                    enhanced_tensor, R, L_init, L_enh = model(img_tensor)
                else:
                    enhanced_tensor, R, L_init, L_enh = self.igr_net(img_tensor)
                metadata['has_retinex_maps'] = True
            elif method.lower() == 'retinexformer':
                enhanced_tensor = self.retinexformer(img_tensor)
            elif 'classical' in method.lower() or 'wren' in method.lower():
                enhanced_tensor = self._classical_retinex_enhance(img_tensor)
            else:
                enhanced_tensor = img_tensor

        latency_ms = (time.time() - start_time) * 1000.0
        enhanced_pil = to_pil_image(enhanced_tensor)
        return enhanced_pil, latency_ms, metadata

    def _classical_retinex_enhance(self, img_tensor):
        """
        Classical Multi-Scale Retinex with Color Restoration (MSRCR) inspired formulation.
        Computes log-space illumination separation via multi-scale Gaussian blurs.
        """
        x = img_tensor.squeeze(0).cpu().numpy()
        scales = [15, 80, 250]
        retinex = np.zeros_like(x)
        
        for scale in scales:
            # Gaussian blur per channel
            blurred = np.zeros_like(x)
            for c in range(3):
                # Simple box blur approximation for speed
                pad = scale // 2
                blurred[c] = np.pad(x[c], pad, mode='reflect')[pad:pad+x.shape[1], pad:pad+x.shape[2]]
            retinex += np.log10(x + 1e-4) - np.log10(blurred + 1e-4)

        retinex = retinex / len(scales)
        # Normalize to [0, 1]
        retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min() + 1e-6)
        return torch.from_numpy(retinex).unsqueeze(0).float().to(self.device)
