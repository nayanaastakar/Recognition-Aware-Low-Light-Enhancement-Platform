"""
Proposed Architecture: Illumination-Guided and Recognition-Aware Enhancement Network (IGR-Net)
Academic Project: M.Tech Minor Project

Key Components:
1. Retinex Decomposition Encoder (Separating Reflectance & Illumination)
2. Reflectance Refinement Branch (Preserving structural boundaries & texture)
3. Illumination Enhancement Branch (Encoder-decoder with global illumination priors)
4. Local-Global Feature Module (Captures multi-scale contextual features)
5. Spatially-Varying Noise-Aware Attention (Suppresses dark-region sensor noise)
6. Squeeze-and-Excitation Channel Attention (Refines spectral color channels)
7. Retinex Fusion & Reconstruction Layer: I_enhanced = R_refined * L_enhanced + residual

Includes ablation switches for systematic ablation analysis:
- Full IGR-Net
- Without Local-Global module (wo_local_global)
- Without Noise-Aware Attention (wo_noise_attn)
- Without Channel Attention (wo_channel_attn)
- Base Retinex baseline (base_retinex)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention block."""
    def __init__(self, channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class NoiseAwareAttention(nn.Module):
    """
    Spatially-varying Noise-Aware Attention module.
    Detects high-variance noise in low-exposure regions and applies spatial attenuation.
    """
    def __init__(self, in_channels):
        super(NoiseAwareAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, illumination):
        """
        Takes features x and estimated illumination map.
        Darker illumination regions with high high-frequency variance are guided with noise suppression weights.
        """
        f = self.relu(self.conv1(x))
        spatial_noise_map = self.sigmoid(self.conv2(f))
        
        # Invert illumination: 1 - illu emphasizes darker areas
        noise_guidance = (1.0 - illumination) * spatial_noise_map
        # Weighting: attenuate noisy features while maintaining structural residuals
        attn_out = x * (1.0 - 0.5 * noise_guidance)
        return attn_out, spatial_noise_map

class LocalGlobalFeatureModule(nn.Module):
    """
    Combines local detailed convolutions (3x3) with multi-scale dilated receptive fields
    and global average pooling context.
    """
    def __init__(self, channels):
        super(LocalGlobalFeatureModule, self).__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        self.fuse = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.size()
        local_feat = self.local_conv(x)
        dilated_feat = self.dilated_conv(x)
        combined = self.fuse(torch.cat([local_feat, dilated_feat], dim=1))
        
        # Global context modulation
        glo = self.global_pool(x).view(b, c)
        glo_weight = self.global_fc(glo).view(b, c, 1, 1)
        return combined * glo_weight + x

class IGRNet(nn.Module):
    """
    Illumination-Guided and Recognition-Aware Enhancement Network (IGR-Net)
    """
    def __init__(self, 
                 in_channels=3, 
                 feat_dim=32, 
                 wo_local_global=False, 
                 wo_noise_attn=False, 
                 wo_channel_attn=False, 
                 base_retinex=False):
        super(IGRNet, self).__init__()
        
        self.wo_local_global = wo_local_global or base_retinex
        self.wo_noise_attn = wo_noise_attn or base_retinex
        self.wo_channel_attn = wo_channel_attn or base_retinex
        self.base_retinex = base_retinex
        
        # 1. Retinex Decomposition
        self.decomp_conv = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.reflectance_head = nn.Sequential(
            nn.Conv2d(feat_dim, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.illumination_head = nn.Sequential(
            nn.Conv2d(feat_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 2. Reflectance Refinement Branch (Boundaries & Edge details)
        self.ref_conv1 = nn.Conv2d(3, feat_dim, kernel_size=3, padding=1)
        self.ref_conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.ref_out = nn.Conv2d(feat_dim, 3, kernel_size=3, padding=1)

        # 3. Illumination Enhancement Branch
        self.illu_conv1 = nn.Conv2d(1, feat_dim, kernel_size=3, padding=1)
        self.illu_conv2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1)
        self.illu_out = nn.Conv2d(feat_dim, 1, kernel_size=3, padding=1)

        # 4. Optional / Ablatable Modules
        if not self.wo_local_global:
            self.lg_module = LocalGlobalFeatureModule(feat_dim)
            
        if not self.wo_noise_attn:
            self.noise_module = NoiseAwareAttention(feat_dim)
            
        if not self.wo_channel_attn:
            self.ca_module = ChannelAttention(feat_dim)

        # 5. Fusion & Reconstruction
        self.recon_conv = nn.Sequential(
            nn.Conv2d(4, feat_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward Pass:
        x: (B, 3, H, W) in [0, 1]
        Returns:
            enhanced_img: (B, 3, H, W)
            reflectance: (B, 3, H, W)
            illumination: (B, 1, H, W)
            enhanced_illu: (B, 1, H, W)
        """
        # Step 1: Retinex Decomposition
        # Estimate illumination map using maximum color channel prior + learned refinement
        max_channel, _ = torch.max(x, dim=1, keepdim=True)
        f_decomp = self.decomp_conv(x)
        L_delta = self.illumination_head(f_decomp) * 0.5
        L_init = torch.clamp(max_channel + L_delta, 0.01, 1.0)
        
        # Reflectance initial estimate from physical Retinex S / L
        R_raw = x / L_init
        R_refine = self.reflectance_head(f_decomp)
        R_init = torch.clamp(0.6 * R_raw + 0.4 * R_refine, 0.0, 1.0)

        # Step 2: Illumination Enhancement
        f_L = self.relu(self.illu_conv1(L_init))
        if not self.wo_local_global:
            f_L = self.lg_module(f_L)
            
        f_L = self.relu(self.illu_conv2(f_L))
        L_gain = torch.sigmoid(self.illu_out(f_L)) * 0.7 + 0.3
        # Gamma-guided dynamic illumination expansion
        L_enhanced = torch.clamp(torch.pow(L_init, 0.45) * L_gain, 0.0, 1.0)

        # Step 3: Reflectance Refinement with Noise and Channel Attention
        f_R = self.relu(self.ref_conv1(R_init))
        
        if not self.wo_noise_attn:
            f_R, _ = self.noise_module(f_R, L_init)
            
        if not self.wo_channel_attn:
            f_R = self.ca_module(f_R)
            
        f_R = self.relu(self.ref_conv2(f_R))
        R_delta = torch.tanh(self.ref_out(f_R)) * 0.15
        R_refined = torch.clamp(R_init + R_delta, 0.0, 1.0)

        # Step 4: Retinex Physical Recombination
        S_retinex = torch.clamp(R_refined * L_enhanced, 0.0, 1.0)

        # Step 5: Residual Detail Preservation
        f_fuse = torch.cat([S_retinex, L_enhanced], dim=1)
        recon_delta = (self.recon_conv(f_fuse) - 0.5) * 0.2
        final_enhanced = torch.clamp(S_retinex + recon_delta, 0.0, 1.0)

        return final_enhanced, R_refined, L_init, L_enhanced
