
"""
PDS-MPNet 64通道轻量量化版本
- 输入：9通道（原始+CLAHE+白平衡）
- 结构：SharedStem 3层（int4），LightEncoder int8（1x1），SFM fp（1x1），主干binary
- 其他与之前相同
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize_weight(w, quant_mode):
    if quant_mode == 'fp':
        return w
    if quant_mode == 'int8':
        w_max = w.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        scale = w_max / 127.0
        scale = torch.clamp(scale, min=1e-8)
        w_q = torch.round(w / scale).clamp(-128, 127)
        w_deq = w_q * scale
        return w + (w_deq - w).detach()
    if quant_mode == 'int4':
        w_max = w.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        scale = w_max / 7.0
        scale = torch.clamp(scale, min=1e-8)
        w_q = torch.round(w / scale).clamp(-8, 7)
        w_deq = w_q * scale
        return w + (w_deq - w).detach()
    if quant_mode == 'binary':
        alpha = w.abs().mean()
        w_bin = torch.sign(w) * alpha
        return w + (w_bin - w).detach()
    raise ValueError(f"Unknown quant_mode: {quant_mode}")


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, quant_mode='binary'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.quant_mode = quant_mode

    def forward(self, x):
        w_q = quantize_weight(self.weight, self.quant_mode)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class QuantConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None,
                 act='relu', quant_mode='int8'):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.quant_mode = quant_mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.Identity()

    def forward(self, x):
        w_q = quantize_weight(self.conv.weight, self.quant_mode)
        out = F.conv2d(x, w_q, None, self.conv.stride, self.conv.padding,
                       self.conv.dilation, self.conv.groups)
        out = self.bn(out)
        out = self.act(out)
        return out


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, act='relu'):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act == 'relu' else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        return x * spatial_out


class GammaBetaNetQuant(nn.Module):
    def __init__(self, layers=3, channels=16, quant_mode='binary'):
        super().__init__()
        self.in_conv = QuantConvBNReLU(3, channels, 3, act='relu', quant_mode=quant_mode)
        self.blocks = nn.ModuleList([
            QuantConvBNReLU(channels, channels, 3, act='relu', quant_mode=quant_mode)
            for _ in range(layers)
        ])
        self.out_conv = nn.Conv2d(channels, 3, 3, padding=1)
        self.cbam = CBAMLayer(channels)

    def forward(self, x):
        fea = self.in_conv(x)
        for blk in self.blocks:
            fea = fea + blk(fea)
        fea = self.cbam(fea)
        out = self.out_conv(fea)
        return out, fea


class SPBQuant(nn.Module):
    def __init__(self, ch=64, backbone_quant='binary'):
        super().__init__()
        self.backbone = GammaBetaNetQuant(layers=3, channels=16, quant_mode=backbone_quant)
        self.feat_conv = nn.Sequential(
            nn.Conv2d(16, ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.t_head = nn.Sequential(
            ConvBNReLU(ch, ch//2, 3, act='relu'),
            nn.Conv2d(ch//2, 1, 1),
            nn.Sigmoid()
        )
        self.veil_head = nn.Sequential(
            ConvBNReLU(ch, ch//2, 3, act='relu'),
            nn.Conv2d(ch//2, 3, 1)
        )

    def forward(self, x):
        J, mid = self.backbone(x)
        feat = self.feat_conv(mid)
        t = self.t_head(feat)
        v = self.veil_head(feat)
        return J, t, v


class SCBQuant(nn.Module):
    def __init__(self, ch=64, backbone_quant='binary'):
        super().__init__()
        self.backbone = GammaBetaNetQuant(layers=3, channels=16, quant_mode=backbone_quant)
        self.feat_conv = nn.Sequential(
            nn.Conv2d(16, ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.alpha_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//2), nn.ReLU(),
            nn.Linear(ch//2, 3), nn.Sigmoid()
        )
        self.spec_head = nn.Sequential(
            ConvBNReLU(ch, ch//2, 3, act='relu'),
            nn.Conv2d(ch//2, 3, 1)
        )

    def forward(self, x):
        J, mid = self.backbone(x)
        feat = self.feat_conv(mid)
        alpha = self.alpha_head(feat)
        spec = self.spec_head(feat)
        return J, alpha, spec


class LightEncoderQuant(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, quant_mode='int8'):
        super().__init__()
        self.net = nn.Sequential(
            QuantConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0, act='relu', quant_mode=quant_mode),
            QuantConvBNReLU(out_ch, out_ch, kernel_size=1, padding=0, act='relu', quant_mode=quant_mode)
        )

    def forward(self, x):
        return self.net(x)


class SharedStemQuant(nn.Module):
    def __init__(self, in_ch=9, out_ch=64, quant_mode='int4'):
        super().__init__()
        self.net = nn.Sequential(
            QuantConvBNReLU(in_ch, out_ch, 3, act='relu', quant_mode=quant_mode),
            QuantConvBNReLU(out_ch, out_ch, 3, act='relu', quant_mode=quant_mode),
            QuantConvBNReLU(out_ch, out_ch, 3, act='relu', quant_mode=quant_mode)  
        )

    def forward(self, x):
        return self.net(x)


class ASC(nn.Module):
    def __init__(self, feat_ch=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(feat_ch + 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def compute_stats(self, x):
        mean = x.mean(dim=[2,3])
        std = x.std(dim=[2,3])
        return torch.cat([mean, std], dim=1)

    def forward(self, f0, x):
        pooled = self.pool(f0).view(f0.size(0), -1)
        stats = self.compute_stats(x)
        logits = self.mlp(torch.cat([pooled, stats], dim=1))
        w = torch.softmax(logits, dim=1)
        return w[:, 0:1], w[:, 1:2]


class SFMQuant(nn.Module):
    def __init__(self, ch=64, quant_mode='fp'):
        super().__init__()
        self.fuse_conv1 = QuantConvBNReLU(ch, ch, kernel_size=1, padding=0, act='relu', quant_mode=quant_mode)
        self.se = SEBlock(ch)
        self.fuse_conv2 = QuantConvBNReLU(ch, ch, kernel_size=1, padding=0, act='relu', quant_mode=quant_mode)

    def forward(self, f0, Fs, Fc, ws, wc):
        ws = ws.view(-1,1,1,1)
        wc = wc.view(-1,1,1,1)
        fused = ws * Fs + wc * Fc + f0
        fused = self.fuse_conv1(fused)
        fused = self.se(fused)
        fused = self.fuse_conv2(fused)
        return fused


class ReconstructionHeadQuant(nn.Module):
    def __init__(self, ch=64, quant_mode='int8'):
        super().__init__()
        self.head = nn.Sequential(
            QuantConvBNReLU(ch, ch, kernel_size=1, padding=0, act='relu', quant_mode=quant_mode),
            nn.Conv2d(ch, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        return self.head(feat)


class PDSMPNetQuantLight64Enhanced(nn.Module):
    def __init__(self, feat_ch=64, precision_cfg=None):
            super().__init__()


            default_cfg = {
                "shared_stem": "int4",
                "expert_backbone": "binary",   
                "encoder": "int8",             
                "sfm": "fp",
                "recon": "int8",
            }

            if precision_cfg is None:
                print("ERROR!")
                exit(0)
            else:
                tmp = default_cfg.copy()
                tmp.update(precision_cfg)
                precision_cfg = tmp

            self.precision_cfg = precision_cfg

            self.shared_stem = SharedStemQuant(
                in_ch=9,
                out_ch=feat_ch,
                quant_mode=precision_cfg["shared_stem"]
            )

            self.spb = SPBQuant(
                ch=feat_ch,
                backbone_quant=precision_cfg["expert_backbone"]
            )

            self.scb = SCBQuant(
                ch=feat_ch,
                backbone_quant=precision_cfg["expert_backbone"]
            )

            self.enc_spb = LightEncoderQuant(
                3,
                feat_ch,
                quant_mode=precision_cfg["encoder"]
            )

            self.enc_scb = LightEncoderQuant(
                3,
                feat_ch,
                quant_mode=precision_cfg["encoder"]
            )

            self.asc = ASC(feat_ch)

            self.sfm = SFMQuant(
                feat_ch,
                quant_mode=precision_cfg["sfm"]
            )

            self.recon = ReconstructionHeadQuant(
                feat_ch,
                quant_mode=precision_cfg["recon"]
            )

    def forward(self, x):
        f0 = self.shared_stem(x)
        x_raw = x[:, :3, :, :]
        J_spb, t_map, veil = self.spb(x_raw)
        J_scb, alpha, spec = self.scb(x_raw)
        Fs = self.enc_spb(J_spb)
        Fc = self.enc_scb(J_scb)
        ws, wc = self.asc(f0, x_raw)
        fused = self.sfm(f0, Fs, Fc, ws, wc)
        out = self.recon(fused)
        return {
            'enhanced': out, 'J_spb': J_spb, 'J_scb': J_scb,
            't_map': t_map, 'veil_map': veil,
            'alpha': alpha, 'spec_map': spec,
            'ws': ws, 'wc': wc, 'shared_feat': f0
        }