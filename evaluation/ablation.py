"""
Systematic Ablation Study for IGR-Net.
Ablation Configurations:
1. Base Retinex Network (base_retinex)
2. Without Local-Global Context (wo_local_global)
3. Without Spatially-Varying Noise-Aware Attention (wo_noise_attn)
4. Without Channel Attention (wo_channel_attn)
5. Full IGR-Net Architecture
"""

import time
import numpy as np
import pandas as pd
from models.igr_net.model import IGRNet
from evaluation.enhancement_metrics import calculate_psnr, calculate_ssim
from models.transforms import to_tensor, to_pil_image

class AblationStudyRunner:
    def __init__(self, device='cpu'):
        self.device = device
        self.configs = {
            'Base Retinex Network': {'base_retinex': True},
            'w/o Local-Global Context': {'wo_local_global': True},
            'w/o Noise-Aware Attention': {'wo_noise_attn': True},
            'w/o Channel Attention': {'wo_channel_attn': True},
            'Full IGR-Net (Proposed)': {}
        }

    def run_ablation(self, test_low_images, test_high_images):
        """
        Runs ablation evaluation on paired test images.
        """
        records = []
        for name, cfg in self.configs.items():
            model = IGRNet(**cfg).to(self.device)
            model.eval()

            # Parameter count
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            psnrs = []
            ssims = []
            latencies = []

            for low_img, high_img in zip(test_low_images, test_high_images):
                tensor_low = to_tensor(low_img).unsqueeze(0).to(self.device)
                tensor_high = to_tensor(high_img).unsqueeze(0).to(self.device)

                start = time.time()
                enhanced, _, _, _ = model(tensor_low)
                latencies.append((time.time() - start) * 1000.0)

                p = calculate_psnr(enhanced, tensor_high)
                s = calculate_ssim(enhanced, tensor_high)
                psnrs.append(p)
                ssims.append(s)

            records.append({
                'Ablation Variant': name,
                'PSNR (dB)': f"{np.mean(psnrs):.2f}",
                'SSIM': f"{np.mean(ssims):.4f}",
                'Parameters': f"{param_count / 1e3:.1f}K",
                'Inference Latency (ms)': f"{np.mean(latencies):.1f}"
            })

        df = pd.DataFrame(records)
        return df
