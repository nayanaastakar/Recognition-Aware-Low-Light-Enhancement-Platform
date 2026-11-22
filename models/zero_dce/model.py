"""
Official PyTorch implementation of Zero-DCE (Deep Curve Estimation).
Reference: 'Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement' (CVPR 2020 / TPAMI 2021)
Authors: Chongyi Li, Chunle Guo, Chen Change Loy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DCE_Net(nn.Module):
    """
    DCE-Net estimates pixel-wise high-order curve parameter maps.
    Consists of 7 symmetrical convolutional layers with skip connections.
    """
    def __init__(self, n_filters=32, n_iterations=8):
        super(DCE_Net, self).__init__()
        self.n_iterations = n_iterations
        # 8 iterations means 8 curves per RGB channel = 24 curve parameter maps
        self.out_channels = 3 * n_iterations

        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv3 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv4 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True)

        self.e_conv5 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv6 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.e_conv7 = nn.Conv2d(n_filters * 2, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        """
        Input:
            x: Tensor of shape (B, 3, H, W) in range [0, 1]
        Output:
            enhanced: Enhanced image (B, 3, H, W) in range [0, 1]
            A: Curve parameter maps (B, 24, H, W) in range [-1, 1]
        """
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        A = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        # Iterative curve application: LE_n(x) = LE_{n-1}(x) + A_n * LE_{n-1}(x) * (1 - LE_{n-1}(x))
        r = x
        for i in range(self.n_iterations):
            a_i = A[:, i*3:(i+1)*3, :, :]
            r = r + a_i * (torch.pow(r, 2) - r) * (-1.0)
            r = torch.clamp(r, 0.0, 1.0)

        return r, A

class L_spatial(nn.Module):
    """Spatial consistency loss preserving gradient differences between adjacent regions."""
    def __init__(self):
        super(L_spatial, self).__init__()
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4, stride=4)

    def forward(self, org, enhance):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        w_l = self.weight_left.to(org.device)
        w_r = self.weight_right.to(org.device)
        w_u = self.weight_up.to(org.device)
        w_d = self.weight_down.to(org.device)

        d_org_l = F.conv2d(org_pool, w_l, padding=1)
        d_org_r = F.conv2d(org_pool, w_r, padding=1)
        d_org_u = F.conv2d(org_pool, w_u, padding=1)
        d_org_d = F.conv2d(org_pool, w_d, padding=1)

        d_enh_l = F.conv2d(enhance_pool, w_l, padding=1)
        d_enh_r = F.conv2d(enhance_pool, w_r, padding=1)
        d_enh_u = F.conv2d(enhance_pool, w_u, padding=1)
        d_enh_d = F.conv2d(enhance_pool, w_d, padding=1)

        loss = torch.mean(torch.pow(d_org_l - d_enh_l, 2)) + \
               torch.mean(torch.pow(d_org_r - d_enh_r, 2)) + \
               torch.mean(torch.pow(d_org_u - d_enh_u, 2)) + \
               torch.mean(torch.pow(d_org_d - d_enh_d, 2))
        return loss

class L_exposure(nn.Module):
    """Exposure control loss to steer average intensity towards well-exposed target E (default 0.6)."""
    def __init__(self, patch_size=16, mean_val=0.6):
        super(L_exposure, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        mean = self.pool(torch.mean(x, 1, keepdim=True))
        loss = torch.mean(torch.pow(mean - self.mean_val, 2))
        return loss

class L_color(nn.Module):
    """Color constancy loss based on Grey-World assumption to prevent color casts."""
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        drg = torch.pow(mr - mg, 2)
        drb = torch.pow(mr - mb, 2)
        dgb = torch.pow(mg - mb, 2)
        loss = torch.mean(torch.sqrt(drg + drb + dgb + 1e-8))
        return loss

class L_tv(nn.Module):
    """Total variation loss on curve parameters for illumination smoothness."""
    def __init__(self):
        super(L_tv, self).__init__()

    def forward(self, A):
        batch_size = A.size()[0]
        h_x = A.size()[2]
        w_x = A.size()[3]
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((A[:, :, 1:, :] - A[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((A[:, :, :, 1:] - A[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
