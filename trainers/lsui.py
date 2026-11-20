
"""
PDS-MPNet 64通道轻量量化训练（增强版，加入检查点恢复）
- 输入：9通道（预处理）
- 模型：PDSMPNetQuantLight64Enhanced
- 训练策略：无预训练，从头训练1000轮；若检测到 checkpoint，则从 checkpoint 恢复
- DSOS 交替优化，余弦退火重启（T_0=100, T_mult=2）
- 弱监督阶段延长至200轮，指导损失初始系数0.3，熵正则系数0.05
- 检查点恢复：恢复 model / optimizer / EMA / epoch / best_psnr / 随机数状态
"""

import io
import os
import sys
import random
import math
import numpy as np
import torch
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow:
                self.shadow[n] = p.detach().clone()
            else:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)

    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n].data)
        self.backup = {}

    def state_dict(self):
        return {n: t.detach().cpu() for n, t in self.shadow.items()}

    def load_state_dict(self, state_dict, device):
        self.shadow = {
            n: t.detach().to(device).clone()
            for n, t in state_dict.items()
        }
        self.backup = {}


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
        if current_T == 0:
            current_T = 1
        lr = lr_min + 0.5 * (lr_base - lr_min) * (1 + math.cos(math.pi * epoch_after / current_T))

    for g in optimizer.param_groups:
        g['lr'] = lr
    return lr


def get_guide_coeff(epoch, warmup_epochs, guide_epochs, guide_start, guide_end):
    if epoch <= warmup_epochs:
        return guide_start
    if epoch >= guide_epochs:
        return guide_end
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


def get_config_snapshot(cfg):
    snapshot = {}
    for k in dir(cfg):
        if k.startswith('_'):
            continue
        v = getattr(cfg, k)
        if callable(v):
            continue
        if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict)):
            snapshot[k] = v
    return snapshot


def get_rng_state():
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'torch_cuda_random_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(rng_state):
    if not rng_state:
        return

    if 'python_random_state' in rng_state and rng_state['python_random_state'] is not None:
        random.setstate(rng_state['python_random_state'])

    if 'numpy_random_state' in rng_state and rng_state['numpy_random_state'] is not None:
        np.random.set_state(rng_state['numpy_random_state'])

    if 'torch_random_state' in rng_state and rng_state['torch_random_state'] is not None:
        torch.set_rng_state(rng_state['torch_random_state'].cpu())

    cuda_state = rng_state.get('torch_cuda_random_state_all', None)
    if torch.cuda.is_available() and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


def torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(epoch, model, optimizer, ema, best_psnr, cfg, precision_cfg, path, tag="latest"):
    checkpoint = {
        'tag': tag,
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'ema_shadow': ema.state_dict(),
        'best_psnr': best_psnr,
        'precision_cfg': precision_cfg,
        'rng_state': get_rng_state(),
        'config': get_config_snapshot(cfg),
    }
    atomic_torch_save(checkpoint, path)
    print(f"  Checkpoint 已保存: {path} (epoch={epoch}, best_psnr={best_psnr:.4f})")


def save_ema_model(path, epoch, model, ema, best_psnr, cfg, precision_cfg, model_name):
    ema.apply_shadow()
    state = {
        'model_state': model.state_dict(),
        'ema_shadow': ema.state_dict(),
        'epoch': epoch,
        'best_psnr': best_psnr,
        'precision_cfg': precision_cfg,
        'config': get_config_snapshot(cfg),
        'model_name': model_name,
    }
    atomic_torch_save(state, path)
    ema.restore()


def resolve_resume_path(cfg):
    if cfg.resume_path is not None:
        return cfg.resume_path
    return cfg.latest_checkpoint_path


def load_checkpoint_if_available(model, optimizer, ema, cfg, device):
    start_epoch = 1
    best_psnr = 0.0

    if not cfg.auto_resume and cfg.resume_path is None:
        print("Checkpoint 自动恢复已关闭，从头训练。")
        return start_epoch, best_psnr

    resume_path = resolve_resume_path(cfg)
    if not os.path.isfile(resume_path):
        print(f"未检测到可恢复 checkpoint: {resume_path}")
        print("从头训练。")
        return start_epoch, best_psnr

    checkpoint = torch_load_compat(resume_path, map_location=device)

    if 'model_state' not in checkpoint:
        raise KeyError(f"checkpoint 中缺少 model_state: {resume_path}")

    model.load_state_dict(checkpoint['model_state'], strict=cfg.resume_strict)

    if 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        optimizer_to_device(optimizer, device)
    else:
        print("警告：checkpoint 中没有 optimizer_state，优化器将重新初始化。")

    if 'ema_shadow' in checkpoint:
        ema.load_state_dict(checkpoint['ema_shadow'], device)
    else:
        print("警告：checkpoint 中没有 ema_shadow，EMA 将从当前模型重新初始化。")

    best_psnr = float(checkpoint.get('best_psnr', 0.0))
    loaded_epoch = int(checkpoint.get('epoch', 0))
    start_epoch = loaded_epoch + 1

    if cfg.resume_rng_state and 'rng_state' in checkpoint:
        set_rng_state(checkpoint['rng_state'])

    print(f"已从 checkpoint 恢复: {resume_path}")
    print(f"恢复位置: 已完成 epoch={loaded_epoch}，将从 epoch={start_epoch} 开始继续训练")
    print(f"恢复 best_psnr={best_psnr:.4f}")

    return start_epoch, best_psnr


def build_precision_cfg(cfg):
    return {
        "shared_stem": cfg.precision_shared_stem,
        "expert_backbone": cfg.precision_expert_backbone,
        "encoder": cfg.precision_encoder,
        "sfm": cfg.precision_sfm,
        "recon": cfg.precision_recon,
    }


def train_one_epoch(model, train_loader, optimizer, ema, ssim_loss_fn, epoch, cfg, device, asc_params):
    guide_coeff = get_guide_coeff(
        epoch, cfg.warmup_epochs, cfg.guide_epochs,
        cfg.guide_start, cfg.guide_end
    )
    entropy_coeff = get_entropy_coeff(
        epoch, cfg.warmup_epochs, cfg.guide_epochs,
        cfg.entropy_start, cfg.entropy_end
    )

    freeze_asc = (epoch <= cfg.warmup_epochs)
    for p in asc_params:
        p.requires_grad = not freeze_asc

    dsos_start_epoch = cfg.warmup_epochs + 1 if cfg.use_dsos_alternating else cfg.epochs + 1

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
    losses = {'total': 0.0, 'rec': 0.0, 'ssim': 0.0, 'spb': 0.0, 'scb': 0.0, 'guide': 0.0, 'ent': 0.0}
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
            ws_for_aux = torch.full_like(ws, 0.5)
            wc_for_aux = torch.full_like(wc, 0.5)
        else:
            ws_for_aux = ws.detach()
            wc_for_aux = wc.detach()

        spb_val = spb_loss(out['J_spb'], out['t_map'], out['veil_map'], gt)
        scb_val = scb_loss(out['J_scb'], out['alpha'], out['spec_map'], gt)
        spb_loss_val = ws_for_aux.mean() * spb_val
        scb_loss_val = wc_for_aux.mean() * scb_val

        if guide_coeff > 0 and not freeze_asc:
            guide = F.mse_loss(out['ws'], tws) + F.mse_loss(out['wc'], twc)
        else:
            guide = torch.zeros((), device=device)

        if entropy_coeff > 0 and not freeze_asc:
            ent = entropy_regularization(out['ws'], out['wc'])
        else:
            ent = torch.zeros((), device=device)

        total = (
            rec
            + cfg.lambda_ssim * ssim
            + cfg.lambda_spb * spb_loss_val
            + cfg.lambda_scb * scb_loss_val
            + guide_coeff * guide
            + entropy_coeff * ent
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        ema.update()

        losses['total'] += total.item()
        losses['rec'] += rec.item()
        losses['ssim'] += ssim.item()
        losses['spb'] += spb_val.item()
        losses['scb'] += scb_val.item()
        losses['guide'] += guide.item()
        losses['ent'] += ent.item()

        pbar.set_postfix({'Loss': f"{total.item():.4f}"})

    for k in losses:
        losses[k] /= max(num_batches, 1)

    return losses, guide_coeff, entropy_coeff


def main():
    cfg = Config()
    set_seed(cfg.seed)
    sys.stdout = Logger(cfg.log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PDS-MPNet 64通道增强训练 (9ch, DSOS, 1000 epochs, checkpoint resume)")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    train_dataset = PDSImageDataset(
        cfg.input_dir, cfg.label_dir, csv_path=cfg.csv_path,
        transform=transform, use_preprocess=True, augment=True, use_degradation_labels=True
    )
    test_dataset = PDSImageDataset(
        cfg.test_input_dir, cfg.test_label_dir,
        transform=transform, use_preprocess=True, augment=False, use_degradation_labels=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    print(f"训练集: {len(train_dataset)} 张  测试集: {len(test_dataset)} 张")

    precision_cfg = build_precision_cfg(cfg)
    print(f"Precision config: {precision_cfg}")

    model = PDSMPNetQuantLight64Enhanced(feat_ch=64, precision_cfg=precision_cfg).to(device)
    print("模型已初始化。")

    ssim_loss_fn = SSIMLoss().to(device)

    asc_params = list(model.asc.parameters())
    base_params = [p for n, p in model.named_parameters() if 'asc' not in n]
    optimizer = optim.AdamW(
        [
            {'params': base_params, 'lr': cfg.lr_base},
            {'params': asc_params, 'lr': cfg.lr_base}
        ],
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay
    )

    ema = EMA(model, cfg.ema_decay)

    
    start_epoch, best_psnr = load_checkpoint_if_available(model, optimizer, ema, cfg, device)

    if start_epoch == 1:
        print("未使用预训练权重，所有参数随机初始化。")
    elif start_epoch > cfg.epochs:
        print(f"checkpoint 已完成到 epoch={start_epoch - 1}，不需要继续训练。")
        return

    last_finished_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            lr = get_cosine_schedule_with_warmup(
                optimizer, epoch,
                warmup_epochs=cfg.warmup_epochs,
                T_0=cfg.T_0,
                T_mult=cfg.T_mult,
                lr_base=cfg.lr_base,
                lr_min=cfg.lr_min
            )

            losses, guide_coeff, entropy_coeff = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                ema=ema,
                ssim_loss_fn=ssim_loss_fn,
                epoch=epoch,
                cfg=cfg,
                device=device,
                asc_params=asc_params
            )

            print(
                f"Epoch {epoch:4d}/{cfg.epochs} | "
                f"Loss={losses['total']:.4f} Rec={losses['rec']:.4f} "
                f"SSIM={losses['ssim']:.4f} SPB={losses['spb']:.4f} SCB={losses['scb']:.4f} "
                f"Guide={losses['guide']:.4f} Ent={losses['ent']:.4f} | "
                f"guide_coeff={guide_coeff:.4f} entropy_coeff={entropy_coeff:.4f} | lr={lr:.2e}"
            )

            if epoch % cfg.model_save_every == 0 or epoch == cfg.epochs:
                ema.apply_shadow()
                test_psnr = evaluate_psnr(model, test_loader, device)
                if test_psnr > best_psnr:
                    best_psnr = test_psnr
                    state = {
                        'model_state': model.state_dict(),
                        'ema_shadow': ema.state_dict(),
                        'epoch': epoch,
                        'best_psnr': best_psnr,
                        'precision_cfg': precision_cfg,
                        'config': get_config_snapshot(cfg),
                        'model_name': 'PDSMPNetQuantLight64Enhanced_EMA_BEST',
                    }
                    atomic_torch_save(state, cfg.best_ema_path)
                    print(f"  新最佳 EMA 模型已保存: {cfg.best_ema_path} (PSNR={best_psnr:.4f})")
                else:
                    print(f"  测试集 PSNR (EMA): {test_psnr:.4f} | 当前 best_psnr={best_psnr:.4f}")
                ema.restore()

            
            if epoch % cfg.checkpoint_save_every == 0 or epoch == cfg.epochs:
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    best_psnr=best_psnr,
                    cfg=cfg,
                    precision_cfg=precision_cfg,
                    path=cfg.latest_checkpoint_path,
                    tag="latest"
                )

            if cfg.checkpoint_backup_every > 0 and (epoch % cfg.checkpoint_backup_every == 0 or epoch == cfg.epochs):
                backup_path = os.path.join(cfg.save_dir, f"checkpoint_epoch_{epoch:04d}.pth")
                save_checkpoint(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    ema=ema,
                    best_psnr=best_psnr,
                    cfg=cfg,
                    precision_cfg=precision_cfg,
                    path=backup_path,
                    tag=f"epoch_{epoch:04d}"
                )

            if epoch % cfg.image_save_every == 0:
                save_test_images(model, test_loader, epoch, cfg.result_dir, device)

            last_finished_epoch = epoch

        
        save_ema_model(
            path=cfg.final_ema_path,
            epoch=cfg.epochs,
            model=model,
            ema=ema,
            best_psnr=best_psnr,
            cfg=cfg,
            precision_cfg=precision_cfg,
            model_name='PDSMPNetQuantLight64Enhanced_EMA_FINAL'
        )
        print(f"训练完成。Final EMA 已保存: {cfg.final_ema_path}")
        print(f"Best PSNR: {best_psnr:.4f}")

    except KeyboardInterrupt:
        print("训练被手动中断，正在保存 interrupt checkpoint。")
        
        # 恢复时会从 last_finished_epoch + 1 开始，属于 epoch 级恢复，不是 batch 级完全恢复。
        save_checkpoint(
            epoch=last_finished_epoch,
            model=model,
            optimizer=optimizer,
            ema=ema,
            best_psnr=best_psnr,
            cfg=cfg,
            precision_cfg=precision_cfg,
            path=cfg.interrupt_checkpoint_path,
            tag="interrupt"
        )
        print(f"Interrupt checkpoint 已保存: {cfg.interrupt_checkpoint_path}")
        raise


@torch.no_grad()
def evaluate_psnr(model, loader, device):
    model.eval()
    total = 0.0
    for img, gt, _, _, _ in loader:
        img, gt = img.to(device), gt.to(device)
        out = model(img)['enhanced'].clamp(0, 1)
        total += calculate_psnr(out, gt).item()
    return total / max(len(loader), 1)


@torch.no_grad()
def save_test_images(model, loader, epoch, result_dir, device):
    model.eval()
    epoch_dir = os.path.join(result_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    for img, gt, _, _, names in loader:
        img = img.to(device)
        out = model(img)['enhanced'].clamp(0, 1)
        for i in range(out.size(0)):
            save_image(names[i] + '.png', torch_to_np(out[i].unsqueeze(0)), epoch_dir + '/')


if __name__ == "__main__":
    main()
