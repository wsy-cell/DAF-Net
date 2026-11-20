SAVE_PATH = 'outputs/euvp/evaluation/results.txt'

"""
完整评估脚本（适配 64通道9输入增强模型）
- 模型：PDSMPNetQuantLight64Enhanced
- 输入：9通道（原始+CLAHE+白平衡）
- 按照 DSOS-UIE 论文标准测量所有指标
"""

import os
import time
import math
import copy
import numpy as np
import cv2
import torch
import lpips
import pyiqa
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from thop import profile


from dafnet.model import PDSMPNetQuantLight64Enhanced
from dafnet.utils import calculate_psnr, calculate_ssim


TEST_INPUT_DIR = 'datasets/euvp/test/input-200_resize256'
TEST_LABEL_DIR = 'datasets/euvp/test/gt-200_resize256'

WEIGHT_PATH = 'checkpoints/euvp/best_model_ema.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FPS_WARMUP = 50
FPS_ITERS = 1000
INPUT_SIZE = (1, 9, 256, 256)   


def clahe_enhance(img_pil):
    img_np = np.array(img_pil)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return Image.fromarray(cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB))

def gray_world_balance(img_pil):
    img_np = np.array(img_pil).astype(np.float32)
    rm, gm, bm = img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()
    gray_mean = (rm + gm + bm) / 3.0
    img_np[:,:,0] = np.clip(img_np[:,:,0] * gray_mean / (rm + 1e-6), 0, 255)
    img_np[:,:,1] = np.clip(img_np[:,:,1] * gray_mean / (gm + 1e-6), 0, 255)
    img_np[:,:,2] = np.clip(img_np[:,:,2] * gray_mean / (bm + 1e-6), 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))


def uicm(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0
    R, G, B = rgb_f[:,:,0], rgb_f[:,:,1], rgb_f[:,:,2]
    RG = R - G
    YB = (R + G) / 2.0 - B
    def trimmed_mean_std(x, alpha=0.01, beta=0.01):
        n = len(x)
        a = int(np.floor(alpha * n))
        b = int(np.floor(beta * n))
        if a + b >= n:
            return np.mean(x), np.std(x)
        sx = np.sort(x)
        trimmed = sx[a : n-b]
        return np.mean(trimmed), np.std(trimmed)
    mu_RG, sigma_RG = trimmed_mean_std(RG.flatten())
    mu_YB, sigma_YB = trimmed_mean_std(YB.flatten())
    return -0.0268 * np.sqrt(mu_RG**2 + mu_YB**2) + 0.1586 * np.sqrt(sigma_RG**2 + sigma_YB**2)

def uism(rgb):
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    h, w = rgb.shape[:2]
    k1, k2 = 8, 8
    block_h = h // k1
    block_w = w // k2
    weights = [0.299, 0.587, 0.114]
    total_eme = 0.0
    for c in range(3):
        channel = rgb[:,:,c].astype(np.float32)
        grad_x = cv2.filter2D(channel, -1, sobel_x)
        grad_y = cv2.filter2D(channel, -1, sobel_y)
        edge = np.sqrt(grad_x**2 + grad_y**2) + 1e-6
        eme_sum = 0.0
        for i in range(k1):
            for j in range(k2):
                block = edge[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if block.size == 0:
                    continue
                max_val = block.max()
                min_val = block.min()
                if min_val <= 0:
                    min_val = 1e-6
                eme_sum += 20 * np.log10(max_val / min_val)
        total_eme += weights[c] * (eme_sum / (k1 * k2))
    return total_eme

def uiconm(rgb):
    gray = 0.299 * rgb[:,:,0] + 0.587 * rgb[:,:,1] + 0.114 * rgb[:,:,2]
    gray = gray.astype(np.float32)
    h, w = gray.shape
    k1, k2 = 8, 8
    block_h = h // k1
    block_w = w // k2
    total = 0.0
    count = 0
    for i in range(k1):
        for j in range(k2):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size == 0:
                continue
            max_val = block.max()
            min_val = block.min()
            if max_val == min_val:
                continue
            contrast = (max_val - min_val) / (max_val + min_val + 1e-6)
            total += contrast * np.log(contrast + 1e-6)
            count += 1
    if count == 0:
        return 0.0
    return total / count

def uiqm_metric(img_tensor):
    img_np = (img_tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    uicm_val = uicm(img_np)
    uism_val = uism(img_np)
    uiconm_val = uiconm(img_np)
    raw = 0.0282 * uicm_val + 0.2953 * uism_val + 3.5753 * uiconm_val
    return raw * 0.1289


niqe_metric = None


def load_model():
    precision_cfg = {
        "shared_stem": "int4",
        "expert_backbone": "binary",
        "encoder": "int8",
        "sfm": "int8",
        "recon": "int8",
    }

    model = PDSMPNetQuantLight64Enhanced(
        feat_ch=64,
        precision_cfg=precision_cfg
    ).to(DEVICE)
    state = torch.load(WEIGHT_PATH, map_location='cpu', weights_only=False)
    if 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def measure_fps(model, input_size, warmup=50, runs=1000):
    model.eval()
    input_tensor = torch.randn(input_size).to(DEVICE)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return runs / elapsed


def measure_flops(model, input_size):
    model_copy = copy.deepcopy(model).eval().to(DEVICE)
    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net
        def forward(self, x):
            return self.net(x)['enhanced']
    wrapped = Wrap(model_copy)
    x = torch.randn(input_size).to(DEVICE)
    flops, _ = profile(wrapped, inputs=(x,), verbose=False)
    return flops / 1e9


def evaluate_quality(model):
    to_tensor = transforms.ToTensor()
    img_files = sorted([f for f in os.listdir(TEST_INPUT_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    gt_files  = sorted([f for f in os.listdir(TEST_LABEL_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    assert len(img_files) == len(gt_files)
    lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)

    psnr_list, ssim_list, lpips_list, uiqm_list, niqe_list = [], [], [], [], []

    for inp_name, gt_name in tqdm(zip(img_files, gt_files), total=len(img_files)):
        
        inp_pil = Image.open(os.path.join(TEST_INPUT_DIR, inp_name)).convert('RGB')
        gt_pil  = Image.open(os.path.join(TEST_LABEL_DIR, gt_name)).convert('RGB')

        
        clahe_pil = clahe_enhance(inp_pil)
        wb_pil    = gray_world_balance(inp_pil)

        
        inp_tensor = torch.cat([
            to_tensor(inp_pil),
            to_tensor(clahe_pil),
            to_tensor(wb_pil)
        ], dim=0).unsqueeze(0).to(DEVICE)

        gt_tensor = to_tensor(gt_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp_tensor)['enhanced']
            out = torch.clamp(out, 0, 1)

        psnr_list.append(calculate_psnr(out, gt_tensor).item())
        ssim_list.append(calculate_ssim(out, gt_tensor).item())
        lpips_list.append(lpips_fn(out*2-1, gt_tensor*2-1).item())
        uiqm_list.append(uiqm_metric(out))
        niqe_list.append(niqe_metric(out).item())

    return {
        "PSNR": np.mean(psnr_list),
        "SSIM": np.mean(ssim_list),
        "LPIPS": np.mean(lpips_list),
        "UIQM": np.mean(uiqm_list),
        "NIQE": np.mean(niqe_list)
    }


def main():
    print("加载模型...")
    model = load_model()

    
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6
    print(f"Params: {params_m:.3f} M")

    
    print("测量 GPU FPS...")
    fps = measure_fps(model, INPUT_SIZE, warmup=FPS_WARMUP, runs=FPS_ITERS)
    print(f"FPS: {fps:.2f}")

    
    print("测量 FLOPs...")
    flops_g = measure_flops(model, INPUT_SIZE)
    print(f"FLOPs: {flops_g:.3f} G")

    
    print("评估 UIEB 测试集...")
    metrics = evaluate_quality(model)

    
    print("\n" + "="*60)
    print("64通道增强模型评估结果 (9通道输入，论文标准)")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<12}")
    print("-"*32)
    print(f"{'Params (M)':<20} {params_m:<12.3f}")
    print(f"{'FLOPs (G)':<20} {flops_g:<12.3f}")
    print(f"{'GPU FPS':<20} {fps:<12.2f}")
    print(f"{'PSNR (dB)':<20} {metrics['PSNR']:<12.4f}")
    print(f"{'SSIM':<20} {metrics['SSIM']:<12.4f}")
    print(f"{'LPIPS':<20} {metrics['LPIPS']:<12.4f}")
    print(f"{'UIQM':<20} {metrics['UIQM']:<12.4f}")
    print(f"{'NIQE':<20} {metrics['NIQE']:<12.4f}")
    print("="*60)

    
    with open(SAVE_PATH, "w") as f:
        f.write(f"Params (M): {params_m:.3f}\n")
        f.write(f"FLOPs (G): {flops_g:.3f}\n")
        f.write(f"GPU FPS: {fps:.2f}\n")
        f.write(f"PSNR: {metrics['PSNR']:.4f}\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"LPIPS: {metrics['LPIPS']:.4f}\n")
        f.write(f"UIQM: {metrics['UIQM']:.4f}\n")
        f.write(f"NIQE: {metrics['NIQE']:.4f}\n")
    print("\n结果已保存至 quant_light_64_enhanced_results.txt")

if __name__ == "__main__":
    main()