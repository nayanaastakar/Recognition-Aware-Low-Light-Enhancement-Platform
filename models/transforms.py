"""
Self-contained image transforms implemented using PyTorch and NumPy.
Avoids external dependencies while guaranteeing reproducibility across platforms.
"""

import torch
import numpy as np
from PIL import Image

def to_tensor(img):
    """
    Convert a PIL Image or numpy ndarray (H, W, C) in [0, 255] to a PyTorch FloatTensor (C, H, W) in [0.0, 1.0].
    """
    if isinstance(img, Image.Image):
        img_np = np.array(img, dtype=np.float32)
    elif isinstance(img, np.ndarray):
        img_np = img.astype(np.float32)
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(img)}")

    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=-1)
    
    # Normalize to [0, 1]
    img_np = img_np / 255.0
    # Transpose from (H, W, C) to (C, H, W)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    return tensor

def to_pil_image(tensor):
    """
    Convert a PyTorch Tensor (C, H, W) in [0.0, 1.0] to a PIL Image (RGB).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    t = tensor.detach().cpu().clamp(0.0, 1.0)
    if t.dim() == 4:
        t = t.squeeze(0)
    
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
        
    np_img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(np_img)

def resize_tensor(tensor, size):
    """
    Resize tensor of shape (B, C, H, W) or (C, H, W) to given (target_h, target_w) using bilinear interpolation.
    """
    is_3d = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        is_3d = True
    
    out = torch.nn.functional.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    if is_3d:
        out = out.squeeze(0)
    return out
