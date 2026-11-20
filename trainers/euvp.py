
"""
PDS-MPNet 64通道轻量量化训练（增强版，从零开始）
- 输入：9通道（预处理）
- 模型：PDSMPNetQuantLight64Enhanced
- 训练策略：无预训练，从头训练1000轮
- DSOS 交替优化，余弦退火重启（T_0=100, T_mult=2）
- 弱监督阶段延长至200轮，指导损失初始系数0.3，熵正则系数0.05
"""

import io, os, sys, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dafnet.model import PDSMPNetQuantLight64Enhanced
from dafnet.losses import charbonnier_loss, SSIMLoss, spb_loss, scb_loss, entropy_regularization
from dafnet.utils import Logger, save_image, torch_to_np, calculate_psnr
from dafnet.dataset import PDSImageDataset



class Config:
    pass


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.clone().detach() + self.decay * self.shadow[n]

    def apply_shadow(self):
        self.backup = {}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.clone().detach()
                p.data = self.shadow[n].clone()

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data = self.backup[n].clone()


def get_cosine_schedule_with_warmup(optimizer, epoch, warmup_epochs, T_0, T_mult, lr_base, lr_min):
    if epoch <= warmup_epochs:
        lr = lr_base * epoch / warmup_epochs
    else:
        epoch_after = epoch - warmup_epochs
        cycle = 0
        current_T = T_0
        while epoch_after >= current_T:
            epoch_after -= current_T
            cycle += 1
            current_T = int(T_0 * (T_mult ** cycle))
        if current_T == 0: current_T = 1
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * epoch_after / current_T))
    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def get_guide_coeff(epoch, warmup_epochs, guide_epochs, guide_start, guide_end):
    if epoch <= warmup_epochs: return guide_start
    if epoch >= guide_epochs: return guide_end
    progress = (epoch - warmup_epochs) / (guide_epochs - warmup_epochs)
    return guide_start + (guide_end - guide_start) * progress


def get_entropy_coeff(epoch, warmup_epochs, guide_epochs, entropy_start, entropy_end):
    return get_guide_coeff(epoch, warmup_epochs, guide_epochs, entropy_start, entropy_end)


def set_dsos_requires_grad(model, mode, cfg):
    if mode == 'high':
        for n, p in model.named_parameters():
            if any(m in n for m in cfg.low_precision_modules):
                p.requires_grad = False
            else:
                p.requires_grad = True
        print("DSOS: Training HIGH-precision modules")
    else:
        for n, p in model.named_parameters():
            if any(m in n for m in cfg.high_precision_modules):
                p.requires_grad = False
            else:
                p.requires_grad = True
        print("DSOS: Training LOW-precision modules")


def main():
    cfg = Config()
    set_seed(cfg.seed)
    sys.stdout = Logger(cfg.log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PDS-MPNet 64通道增强训练 (9ch, DSOS, 1000 epochs, 从头训练)")

    transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

    train_dataset = PDSImageDataset(
        cfg.input_dir, cfg.label_dir, csv_path=cfg.csv_path,
        transform=transform, use_preprocess=True, augment=True, use_degradation_labels=True
    )
    test_dataset = PDSImageDataset(
        cfg.test_input_dir, cfg.test_label_dir, transform=transform,
        use_preprocess=True, augment=False, use_degradation_labels=False
    )
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"训练集: {len(train_dataset)} 张  测试集: {len(test_dataset)} 张")

    precision_cfg = {
    "shared_stem": cfg.precision_shared_stem,
    "expert_backbone": cfg.precision_expert_backbone,
    "encoder": cfg.precision_encoder,
    "sfm": cfg.precision_sfm,
    "recon": cfg.precision_recon,
    }

    model = PDSMPNetQuantLight64Enhanced(feat_ch=64,precision_cfg=precision_cfg).to(device)
    
    print("未使用预训练权重，所有参数随机初始化")

    ssim_loss_fn = SSIMLoss().to(device)

    asc_params = list(model.asc.parameters())
    base_params = [p for n, p in model.named_parameters() if 'asc' not in n]
    optimizer = optim.AdamW([
        {'params': base_params, 'lr': cfg.lr_base},
        {'params': asc_params,  'lr': cfg.lr_base}
    ], betas=(0.9, 0.999), weight_decay=cfg.weight_decay)

    ema = EMA(model, cfg.ema_decay)
    best_psnr = 0.0
    dsos_start_epoch = cfg.warmup_epochs + 1 if cfg.use_dsos_alternating else cfg.epochs + 1

    for epoch in range(1, cfg.epochs + 1):
        lr = get_cosine_schedule_with_warmup(
            optimizer, epoch,
            warmup_epochs=cfg.warmup_epochs,
            T_0=cfg.T_0, T_mult=cfg.T_mult,
            lr_base=cfg.lr_base, lr_min=cfg.lr_min
        )

        guide_coeff = get_guide_coeff(epoch, cfg.warmup_epochs, cfg.guide_epochs,
                                      cfg.guide_start, cfg.guide_end)
        entropy_coeff = get_entropy_coeff(epoch, cfg.warmup_epochs, cfg.guide_epochs,
                                          cfg.entropy_start, cfg.entropy_end)

        freeze_asc = (epoch <= cfg.warmup_epochs)
        for p in asc_params:
            p.requires_grad = not freeze_asc

        
        if cfg.use_dsos_alternating and epoch >= dsos_start_epoch:
            off = epoch - dsos_start_epoch
            cyc = cfg.dsos_cycle_epochs
            phase = 'low' if (off // cyc) % 2 == 0 else 'high'
            set_dsos_requires_grad(model, phase, cfg)
            if not freeze_asc:
                for p in asc_params:
                    p.requires_grad = True
        else:
            for n, p in model.named_parameters():
                if 'asc' not in n:
                    p.requires_grad = True
            if not freeze_asc:
                for p in asc_params:
                    p.requires_grad = True

        model.train()
        losses = {'total':0, 'rec':0, 'ssim':0, 'spb':0, 'scb':0, 'guide':0, 'ent':0}
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}")

        for img, gt, tws, twc, _ in pbar:
            img, gt = img.to(device), gt.to(device)
            tws, twc = tws.to(device), twc.to(device)

            out = model(img)
            pred = out['enhanced']

            rec = charbonnier_loss(pred, gt)
            ssim = ssim_loss_fn(pred, gt)

            ws, wc = out['ws'], out['wc']
            if freeze_asc:
                ws = torch.full_like(ws, 0.5)
                wc = torch.full_like(wc, 0.5)
            else:
                ws = ws.detach()
                wc = wc.detach()

            spb_val = spb_loss(out['J_spb'], out['t_map'], out['veil_map'], gt)
            scb_val = scb_loss(out['J_scb'], out['alpha'], out['spec_map'], gt)
            spb_loss_val = ws.mean() * spb_val
            scb_loss_val = wc.mean() * scb_val

            guide = F.mse_loss(out['ws'], tws) + F.mse_loss(out['wc'], twc) if (guide_coeff>0 and not freeze_asc) else 0.0
            ent = entropy_regularization(out['ws'], out['wc']) if (entropy_coeff>0 and not freeze_asc) else 0.0

            total = rec + cfg.lambda_ssim * ssim + cfg.lambda_spb * spb_loss_val + cfg.lambda_scb * scb_loss_val \
                    + guide_coeff * guide + entropy_coeff * ent

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            ema.update()

            losses['total'] += total.item()
            losses['rec'] += rec.item()
            losses['ssim'] += ssim.item()
            losses['spb'] += spb_val.item()
            losses['scb'] += scb_val.item()
            losses['guide'] += guide.item() if isinstance(guide, torch.Tensor) else guide
            losses['ent'] += ent.item() if isinstance(ent, torch.Tensor) else ent

            pbar.set_postfix({'Loss': f"{total.item():.4f}"})

        n = num_batches
        print(f"Epoch {epoch:4d}/{cfg.epochs} | Loss={losses['total']/n:.4f} Rec={losses['rec']/n:.4f} "
              f"SSIM={losses['ssim']/n:.4f} SPB={losses['spb']/n:.4f} SCB={losses['scb']/n:.4f} "
              f"Guide={losses['guide']/n:.4f} Ent={losses['ent']/n:.4f} | lr={lr:.2e}")

        if epoch % cfg.model_save_every == 0 or epoch == cfg.epochs:
            ema.apply_shadow()
            test_psnr = evaluate_psnr(model, test_loader, device)
            if test_psnr > best_psnr:
                best_psnr = test_psnr
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'best_psnr': best_psnr},
                           os.path.join(cfg.save_dir, "best_model_ema.pth"))
                print(f"  新最佳 EMA 模型已保存 (PSNR={best_psnr:.4f})")
            else:
                print(f"  测试集 PSNR (EMA): {test_psnr:.4f}")
            ema.restore()

        if epoch % cfg.image_save_every == 0:
            save_test_images(model, test_loader, epoch, cfg.result_dir, device)

    print("训练完成。")


@torch.no_grad()
def evaluate_psnr(model, loader, device):
    model.eval()
    total = 0.0
    for img, gt, _, _, _ in loader:
        img, gt = img.to(device), gt.to(device)
        out = model(img)['enhanced'].clamp(0,1)
        total += calculate_psnr(out, gt).item()
    return total / len(loader)


@torch.no_grad()
def save_test_images(model, loader, epoch, result_dir, device):
    model.eval()
    epoch_dir = os.path.join(result_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    for img, gt, _, _, names in loader:
        img = img.to(device)
        out = model(img)['enhanced'].clamp(0,1)
        for i in range(out.size(0)):
            save_image(names[i] + '.png', torch_to_np(out[i].unsqueeze(0)), epoch_dir + '/')


if __name__ == "__main__":
    main()