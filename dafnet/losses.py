"""
PDS-MPNet Stage3 损失函数
- 重建损失 (Charbonnier + SSIM)
- 物理分支损失 (J_spb, J_scb 驱动 + 物理量正则)
- 熵正则 (防止 ASC 权重坍塌)
- 预处理辅助损失 (可选，暂不加入主损失)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-3):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=1.0):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        from dafnet.losses import ssim as ssim_fn
        self.ssim_fn = ssim_fn

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        return 1.0 - self.ssim_fn(pred, target, window_size=self.window_size,
                                  size_average=self.size_average)


def edge_map(x):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
    gx = F.conv2d(x, sobel_x, padding=1, groups=x.size(1))
    gy = F.conv2d(x, sobel_y, padding=1, groups=x.size(1))
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def spb_loss(J_spb, t_map, veil_map, gt):
    """
    J_spb:   SPB 恢复图像 [B,3,H,W]
    t_map:   透射率图 [B,1,H,W]
    veil_map:背景光残差 [B,3,H,W]
    gt:      GT 图像 [B,3,H,W]
    """
    
    loss_rec_spb = charbonnier_loss(J_spb, gt)

    
    tv_t = (torch.abs(t_map[:, :, :, 1:] - t_map[:, :, :, :-1]).mean() +
            torch.abs(t_map[:, :, 1:, :] - t_map[:, :, :-1, :]).mean())

    
    residual = J_spb - gt
    if veil_map.shape[-2:] != residual.shape[-2:]:
        veil_map = F.interpolate(veil_map, size=residual.shape[-2:], mode='bilinear', align_corners=False)
    loss_veil = F.l1_loss(torch.tanh(veil_map), torch.tanh(residual))

    
    loss_edge = F.l1_loss(edge_map(J_spb), edge_map(gt))

    return loss_rec_spb + 0.2 * tv_t + 0.1 * loss_veil + 0.1 * loss_edge


def scb_loss(J_scb, alpha, spec_map, gt):
    """
    J_scb:    SCB 恢复图像 [B,3,H,W]
    alpha:    通道衰减系数 [B,3]
    spec_map: 光谱校正图 [B,3,H,W]
    gt:       GT 图像 [B,3,H,W]
    """
    
    loss_rec_scb = charbonnier_loss(J_scb, gt)

    
    gt_mean = gt.mean(dim=[2, 3])  
    loss_alpha = F.l1_loss(alpha, torch.sigmoid(gt_mean))

    
    color_res = J_scb - gt
    if spec_map.shape[-2:] != color_res.shape[-2:]:
        spec_map = F.interpolate(spec_map, size=color_res.shape[-2:], mode='bilinear', align_corners=False)
    loss_spec = F.l1_loss(torch.tanh(spec_map), torch.tanh(color_res))

    return loss_rec_scb + 0.2 * loss_alpha + 0.1 * loss_spec


def entropy_regularization(ws, wc, eps=1e-8):
    ent = - (ws * torch.log(ws + eps) + wc * torch.log(wc + eps))
    return ent.mean()

import torch
from math import exp
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np


class Grayworld_Loss(nn.Module):
    def __init__(self):
        super(Grayworld_Loss, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)
        return k


class Color_Loss(nn.Module):
    def __init__(self):
        super(Color_Loss, self).__init__()

    def forward(self, pred, target):
        pred_mean = torch.mean(pred, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target, dim=[2, 3], keepdim=True)
        pred_var = torch.var(pred, dim=[2, 3], keepdim=True)
        target_var = torch.var(target, dim=[2, 3], keepdim=True)

        mean_loss = torch.mean((pred_mean - target_mean) ** 2)
        var_loss = torch.mean((pred_var - target_var) ** 2)
        
        pred_loss = 0.1 * torch.mean(torch.abs(pred - 0.5))

        total_loss = mean_loss + var_loss + pred_loss
        return total_loss


class ColorDiffLoss(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(ColorDiffLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        
        kernel = cv2.getGaussianKernel(kernel_size, sigma) @ cv2.getGaussianKernel(kernel_size, sigma).T
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel = kernel / kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def low_pass_filter(self, input, kernel):
        kernel = kernel.expand(input.size(1), 1, self.kernel_size, self.kernel_size).to(input.device)
        return F.conv2d(input, kernel, padding=self.kernel_size // 2, groups=input.size(1))

    def forward(self, output, label):
        smoothed_output = self.low_pass_filter(output, self.kernel.to(output.device))
        smoothed_label = self.low_pass_filter(label, self.kernel.to(label.device))
        loss = F.mse_loss(smoothed_output, smoothed_label)
        return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / (v2 + 1e-8))

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2 + 1e-8)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_Loss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)


def dark_channel(image, size=3):
    """
    计算暗通道：先取 RGB 最小值，再做平均池化（对应论文 A_s 算子）
    image: (N, C, H, W)
    """
    r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    dark = torch.min(torch.min(r, g), b).unsqueeze(1)  
    
    
    kernel = torch.ones((1, 1, size, size), device=image.device) / (size * size)
    dark_avg = F.conv2d(dark, kernel, padding=size // 2)  
    return dark_avg.squeeze(1)  


def estimate_transmission(image, dark_channel, omega=0.95):
    """
    估计透射率（论文公式 13）
    dark_channel: (N, H, W) 已平均池化后的暗通道
    """
    
    A = dark_channel.view(dark_channel.size(0), -1).max(dim=1, keepdim=True)[0] + 1e-8
    A = A.unsqueeze(-1).unsqueeze(-1)  
    transmission = 1.0 - omega * dark_channel.unsqueeze(1) / A
    return transmission.squeeze(1)  


class DCP_Loss(nn.Module):
    def __init__(self, omega=0.95):
        super(DCP_Loss, self).__init__()
        self.omega = omega

    def forward(self, image1, image2):
        dark1 = dark_channel(image1)
        dark2 = dark_channel(image2)
        trans1 = estimate_transmission(image1, dark1, self.omega)
        trans2 = estimate_transmission(image2, dark2, self.omega)

        dark_loss = torch.mean((dark1 - dark2) ** 2)
        trans_loss = torch.mean((trans1 - trans2) ** 2)

        total_loss = 0.5 * dark_loss + 0.5 * trans_loss
        
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return total_loss