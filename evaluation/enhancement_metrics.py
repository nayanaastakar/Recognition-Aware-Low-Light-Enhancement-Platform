"""
Image quality evaluation metrics:
1. PSNR (Peak Signal-to-Noise Ratio)
2. SSIM (Structural Similarity Index Measure)
3. NIQE / LPIPS wrappers

Implemented using pure PyTorch / NumPy to ensure high mathematical fidelity and platform portability.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

def calculate_psnr(img1, img2, max_val=1.0):
    """
    Computes PSNR between two images/tensors in range [0, max_val].
    img1, img2: torch.Tensor of shape (B, C, H, W) or (C, H, W), or np.ndarray
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20.0 * math.log10(max_val / math.sqrt(mse))
    return float(psnr)

def _gaussian_kernel(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return kernel_2d.unsqueeze(0).unsqueeze(0)

def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    Computes SSIM index between two images/tensors.
    img1, img2: torch.Tensor (B, C, H, W) in range [0, 1]
    """
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1, dtype=torch.float32)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2, dtype=torch.float32)

    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    channels = img1.shape[1]
    kernel = _gaussian_kernel(window_size, 1.5).repeat(channels, 1, 1, 1).to(img1.device)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(img1, kernel, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean().item())
