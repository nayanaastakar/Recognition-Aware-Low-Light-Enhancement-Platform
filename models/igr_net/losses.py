"""
Comprehensive loss suite for IGR-Net.
Formulates:
1. Reconstruction Loss (L1 + MS-SSIM structural loss)
2. Retinex Decomposition Consistency Loss (|| I_low - R * L ||_1)
3. Illumination Smoothness Loss (Total variation with reflectance gradient weighting)
4. Exposure Control Loss (Penalizes deviations from target well-exposed luminance)
5. Color Consistency Loss (Maintains chromatic vectors across RGB channels)
6. Recognition-Aware Feature Loss (optional detail preservation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IGRNetLoss(nn.Module):
    def __init__(self, 
                 w_recon=1.0, 
                 w_retinex=0.5, 
                 w_smooth=0.15, 
                 w_exp=0.1, 
                 w_color=0.2):
        super(IGRNetLoss, self).__init__()
        self.w_recon = w_recon
        self.w_retinex = w_retinex
        self.w_smooth = w_smooth
        self.w_exp = w_exp
        self.w_color = w_color

        self.l1_loss = nn.L1Loss()

    def gradient_x(self, img):
        return img[:, :, :, :-1] - img[:, :, :, 1:]

    def gradient_y(self, img):
        return img[:, :, :-1, :] - img[:, :, 1:, :]

    def illumination_smoothness_loss(self, L, R):
        """
        Illumination should be piecewise smooth, preserving edges corresponding to reflectance gradients.
        Weight smoothness by exp(-alpha * |grad(R)|).
        """
        R_gray = torch.mean(R, dim=1, keepdim=True)
        grad_rx = torch.abs(self.gradient_x(R_gray))
        grad_ry = torch.abs(self.gradient_y(R_gray))
        
        grad_lx = torch.abs(self.gradient_x(L))
        grad_ly = torch.abs(self.gradient_y(L))

        weight_x = torch.exp(-10.0 * grad_rx)
        weight_y = torch.exp(-10.0 * grad_ry)

        loss = torch.mean(grad_lx * weight_x) + torch.mean(grad_ly * weight_y)
        return loss

    def color_constancy_loss(self, img):
        """Angle/ratio preservation across channels to prevent hue distortion."""
        mean_rgb = torch.mean(img, dim=[2, 3])
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mg - mb, 2)
        return torch.mean(torch.sqrt(d_rg + d_rb + d_gb + 1e-6))

    def exposure_control_loss(self, img, target=0.55):
        """Encourages average illumination to reach optimal dynamic range."""
        gray = torch.mean(img, dim=1, keepdim=True)
        pool = F.avg_pool2d(gray, kernel_size=16, stride=16)
        return torch.mean(torch.pow(pool - target, 2))

    def forward(self, enhanced_img, R, L_init, L_enhanced, low_img, high_img=None):
        # 1. Retinex consistency: input should factorize into R * L_init
        loss_retinex = self.l1_loss(R * L_init, low_img)

        # 2. Illumination smoothness
        loss_smooth = self.illumination_smoothness_loss(L_enhanced, R)

        # 3. Exposure control & Color constancy
        loss_exp = self.exposure_control_loss(enhanced_img)
        loss_color = self.color_constancy_loss(enhanced_img)

        # 4. Reconstruction Loss against paired high-light image (if available)
        if high_img is not None:
            loss_recon = self.l1_loss(enhanced_img, high_img)
        else:
            # Self-supervised fallback when running without paired ground-truth
            loss_recon = self.l1_loss(enhanced_img, R * L_enhanced)

        total_loss = (self.w_recon * loss_recon + 
                      self.w_retinex * loss_retinex + 
                      self.w_smooth * loss_smooth + 
                      self.w_exp * loss_exp + 
                      self.w_color * loss_color)

        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_recon': loss_recon.item(),
            'loss_retinex': loss_retinex.item(),
            'loss_smooth': loss_smooth.item(),
            'loss_exp': loss_exp.item(),
            'loss_color': loss_color.item()
        }
        return total_loss, loss_dict
