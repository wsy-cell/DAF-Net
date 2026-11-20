"""
完整评估脚本（适配 64通道9输入增强模型）
- 模型：PDSMPNetQuantLight64Enhanced
- 输入：9通道（原始+CLAHE+白平衡）
- 评估：Params、FLOPs、FPS、PSNR、SSIM、LPIPS、UIQM、NIQE
"""

import os
import time
import copy
from contextlib import nullcontext

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


TEST_INPUT_DIR = 'datasets/uieb/test/130-test-syn_256'
TEST_LABEL_DIR = 'datasets/uieb/test/130-test-clear_256'

WEIGHT_PATH = 'checkpoints/uieb/best_model_ema.pth'
SAVE_PATH = 'outputs/uieb/evaluation/results.txt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = (1, 9, 256, 256)

FPS_BATCH_SIZES = (1, 8, 16, 32, 64)
FPS_WARMUP = 30
FPS_ITERS = 100


FPS_USE_AMP = True

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def cuda_synchronize():
    if torch.cuda.is_available() and DEVICE.type == "cuda":
        torch.cuda.synchronize()


def autocast_context(enabled=True):
    if DEVICE.type == "cuda" and enabled:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def safe_torch_load(path, map_location="cpu"):
    """
    兼容不同 PyTorch 版本的安全加载方式。
    新版 PyTorch 支持 weights_only=True，旧版不支持。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def select_fps_by_batch(fps_results, target_bs=1):
    """
    从 benchmark_fps_praf 返回的 list[dict] 中提取指定 batch size 的 FPS。
    """
    for item in fps_results:
        if item["batch_size"] == target_bs and item["status"] == "ok":
            return item["fps"]

    raise RuntimeError(f"Batch size {target_bs} FPS was not successfully measured.")


def clahe_enhance(img_pil):
    img_np = np.array(img_pil)

    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge([l_clahe, a, b])
    out = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return Image.fromarray(out)


def gray_world_balance(img_pil):
    img_np = np.array(img_pil).astype(np.float32)

    rm = img_np[:, :, 0].mean()
    gm = img_np[:, :, 1].mean()
    bm = img_np[:, :, 2].mean()

    gray_mean = (rm + gm + bm) / 3.0

    img_np[:, :, 0] = np.clip(img_np[:, :, 0] * gray_mean / (rm + 1e-6), 0, 255)
    img_np[:, :, 1] = np.clip(img_np[:, :, 1] * gray_mean / (gm + 1e-6), 0, 255)
    img_np[:, :, 2] = np.clip(img_np[:, :, 2] * gray_mean / (bm + 1e-6), 0, 255)

    return Image.fromarray(img_np.astype(np.uint8))


def uicm(rgb):
    rgb_f = rgb.astype(np.float32) / 255.0

    R = rgb_f[:, :, 0]
    G = rgb_f[:, :, 1]
    B = rgb_f[:, :, 2]

    RG = R - G
    YB = (R + G) / 2.0 - B

    def trimmed_mean_std(x, alpha=0.01, beta=0.01):
        n = len(x)
        a = int(np.floor(alpha * n))
        b = int(np.floor(beta * n))

        if a + b >= n:
            return np.mean(x), np.std(x)

        sx = np.sort(x)
        trimmed = sx[a:n - b]

        return np.mean(trimmed), np.std(trimmed)

    mu_RG, sigma_RG = trimmed_mean_std(RG.flatten())
    mu_YB, sigma_YB = trimmed_mean_std(YB.flatten())

    return (
        -0.0268 * np.sqrt(mu_RG ** 2 + mu_YB ** 2)
        + 0.1586 * np.sqrt(sigma_RG ** 2 + sigma_YB ** 2)
    )


def uism(rgb):
    sobel_x = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32
    )

    sobel_y = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=np.float32
    )

    h, w = rgb.shape[:2]

    k1, k2 = 8, 8
    block_h = h // k1
    block_w = w // k2

    weights = [0.299, 0.587, 0.114]

    total_eme = 0.0

    for c in range(3):
        channel = rgb[:, :, c].astype(np.float32)

        grad_x = cv2.filter2D(channel, -1, sobel_x)
        grad_y = cv2.filter2D(channel, -1, sobel_y)

        edge = np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6

        eme_sum = 0.0

        for i in range(k1):
            for j in range(k2):
                block = edge[
                    i * block_h:(i + 1) * block_h,
                    j * block_w:(j + 1) * block_w
                ]

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
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    gray = gray.astype(np.float32)

    h, w = gray.shape

    k1, k2 = 8, 8
    block_h = h // k1
    block_w = w // k2

    total = 0.0
    count = 0

    for i in range(k1):
        for j in range(k2):
            block = gray[
                i * block_h:(i + 1) * block_h,
                j * block_w:(j + 1) * block_w
            ]

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
    """
    输入：
        img_tensor: Bx3xHxW，范围 [0, 1]
    输出：
        UIQM 标量
    """
    img_np = (
        img_tensor.squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        * 255.0
    )

    img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    uicm_val = uicm(img_np)
    uism_val = uism(img_np)
    uiconm_val = uiconm(img_np)

    raw = (
        0.0282 * uicm_val
        + 0.2953 * uism_val
        + 3.5753 * uiconm_val
    )

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

    state = safe_torch_load(WEIGHT_PATH, map_location="cpu")

    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)

    if len(missing_keys) > 0:
        print("\n[Warning] Missing keys when loading checkpoint:")
        for k in missing_keys:
            print(f"  {k}")

    if len(unexpected_keys) > 0:
        print("\n[Warning] Unexpected keys when loading checkpoint:")
        for k in unexpected_keys:
            print(f"  {k}")

    model.eval()

    return model


def measure_fps(model, input_size, warmup=50, runs=1000):
    model.eval()

    input_tensor = torch.randn(input_size).to(DEVICE)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    cuda_synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    cuda_synchronize()

    elapsed = time.time() - start

    return runs / elapsed


@torch.no_grad()
def benchmark_fps_praf(
    model,
    input_size=256,
    batch_sizes=(1, 8, 16, 32, 64),
    warmup=30,
    iters=100,
    amp=True
):
    model.eval()

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    results = []

    for bs in batch_sizes:
        try:
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            x = torch.randn(bs, 9, input_size, input_size, device=DEVICE)

            for _ in range(warmup):
                with autocast_context(enabled=amp):
                    _ = model(x)

            cuda_synchronize()

            start = time.perf_counter()

            for _ in range(iters):
                with autocast_context(enabled=amp):
                    _ = model(x)

            cuda_synchronize()

            end = time.perf_counter()

            total_time = end - start

            fps = bs * iters / total_time
            latency_ms_per_image = total_time / (bs * iters) * 1000.0

            if DEVICE.type == "cuda":
                peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            else:
                peak_mem_mb = None

            results.append({
                "batch_size": bs,
                "fps": fps,
                "latency_ms_per_image": latency_ms_per_image,
                "peak_mem_mb": peak_mem_mb,
                "status": "ok"
            })

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                results.append({
                    "batch_size": bs,
                    "fps": None,
                    "latency_ms_per_image": None,
                    "peak_mem_mb": None,
                    "status": "oom"
                })
            else:
                raise

    return results


def measure_flops(model, input_size):
    model_copy = copy.deepcopy(model).eval().to(DEVICE)

    class Wrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)["enhanced"]

    wrapped = Wrap(model_copy)

    x = torch.randn(input_size).to(DEVICE)

    flops, _ = profile(wrapped, inputs=(x,), verbose=False)

    return flops / 1e9


def evaluate_quality(model):
    to_tensor = transforms.ToTensor()

    img_files = sorted([
        f for f in os.listdir(TEST_INPUT_DIR)
        if f.lower().endswith(IMG_EXTS)
    ])

    gt_files = sorted([
        f for f in os.listdir(TEST_LABEL_DIR)
        if f.lower().endswith(IMG_EXTS)
    ])

    assert len(img_files) == len(gt_files), (
        f"Input image number {len(img_files)} != GT image number {len(gt_files)}"
    )

    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    lpips_fn.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    uiqm_list = []
    niqe_list = []

    for inp_name, gt_name in tqdm(zip(img_files, gt_files), total=len(img_files)):
        inp_path = os.path.join(TEST_INPUT_DIR, inp_name)
        gt_path = os.path.join(TEST_LABEL_DIR, gt_name)

        inp_pil = Image.open(inp_path).convert("RGB")
        gt_pil = Image.open(gt_path).convert("RGB")

        clahe_pil = clahe_enhance(inp_pil)
        wb_pil = gray_world_balance(inp_pil)

        inp_tensor = torch.cat([
            to_tensor(inp_pil),
            to_tensor(clahe_pil),
            to_tensor(wb_pil)
        ], dim=0).unsqueeze(0).to(DEVICE)

        gt_tensor = to_tensor(gt_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(inp_tensor)["enhanced"]
            out = torch.clamp(out, 0.0, 1.0)

        psnr_list.append(calculate_psnr(out, gt_tensor).item())
        ssim_list.append(calculate_ssim(out, gt_tensor).item())
        lpips_list.append(lpips_fn(out * 2.0 - 1.0, gt_tensor * 2.0 - 1.0).item())
        uiqm_list.append(uiqm_metric(out))
        niqe_list.append(niqe_metric(out).item())

    return {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "LPIPS": float(np.mean(lpips_list)),
        "UIQM": float(np.mean(uiqm_list)),
        "NIQE": float(np.mean(niqe_list)),
    }


def save_results(
    save_path,
    params_m,
    flops_g,
    fps_main,
    fps_results,
    metrics
):
    with open(save_path, "w") as f:
        f.write("64-channel enhanced model evaluation results\n")
        f.write("Input: 9-channel tensor = raw + CLAHE + gray-world white balance\n")
        f.write("\n")

        f.write("Efficiency Metrics:\n")
        f.write(f"Params (M): {params_m:.3f}\n")
        f.write(f"FLOPs (G): {flops_g:.3f}\n")
        f.write(f"GPU FPS BS=1: {fps_main:.2f}\n")
        f.write("\n")

        f.write("Batch FPS Benchmark:\n")
        for r in fps_results:
            if r["status"] == "ok":
                if r["peak_mem_mb"] is None:
                    peak_str = "N/A"
                else:
                    peak_str = f"{r['peak_mem_mb']:.1f} MB"

                f.write(
                    f"BS={r['batch_size']}: "
                    f"FPS={r['fps']:.2f}, "
                    f"lat/img={r['latency_ms_per_image']:.3f} ms, "
                    f"peak={peak_str}\n"
                )
            else:
                f.write(f"BS={r['batch_size']}: OOM\n")

        f.write("\nQuality Metrics:\n")
        f.write(f"PSNR: {metrics['PSNR']:.4f}\n")
        f.write(f"SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"LPIPS: {metrics['LPIPS']:.4f}\n")
        f.write(f"UIQM: {metrics['UIQM']:.4f}\n")
        f.write(f"NIQE: {metrics['NIQE']:.4f}\n")


def main():
    print("加载模型...")
    model = load_model()

    
    total_params = sum(p.numel() for p in model.parameters())
    params_m = total_params / 1e6

    print(f"Params: {params_m:.3f} M")

    
    print("测量 GPU FPS...")

    fps_results = benchmark_fps_praf(
        model,
        input_size=256,
        batch_sizes=FPS_BATCH_SIZES,
        warmup=FPS_WARMUP,
        iters=FPS_ITERS,
        amp=FPS_USE_AMP
    )

    for r in fps_results:
        if r["status"] == "ok":
            if r["peak_mem_mb"] is None:
                peak_str = "N/A"
            else:
                peak_str = f"{r['peak_mem_mb']:.1f} MB"

            print(
                f"BS={r['batch_size']:>3d} | "
                f"FPS={r['fps']:.2f} | "
                f"lat/img={r['latency_ms_per_image']:.3f} ms | "
                f"peak={peak_str}"
            )
        else:
            print(f"BS={r['batch_size']:>3d} | OOM")

    
    fps_main = select_fps_by_batch(fps_results, target_bs=1)

    
    print("测量 FLOPs...")
    flops_g = measure_flops(model, INPUT_SIZE)

    print(f"FLOPs: {flops_g:.3f} G")

    
    print("评估 UIEB 测试集...")
    metrics = evaluate_quality(model)

    
    print("\n" + "=" * 60)
    print("64通道增强模型评估结果 (9通道输入，论文标准)")
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':<12}")
    print("-" * 32)
    print(f"{'Params (M)':<20} {params_m:<12.3f}")
    print(f"{'FLOPs (G)':<20} {flops_g:<12.3f}")
    print(f"{'GPU FPS BS=1':<20} {fps_main:<12.2f}")
    print(f"{'PSNR (dB)':<20} {metrics['PSNR']:<12.4f}")
    print(f"{'SSIM':<20} {metrics['SSIM']:<12.4f}")
    print(f"{'LPIPS':<20} {metrics['LPIPS']:<12.4f}")
    print(f"{'UIQM':<20} {metrics['UIQM']:<12.4f}")
    print(f"{'NIQE':<20} {metrics['NIQE']:<12.4f}")
    print("=" * 60)

    
    save_results(
        save_path=SAVE_PATH,
        params_m=params_m,
        flops_g=flops_g,
        fps_main=fps_main,
        fps_results=fps_results,
        metrics=metrics
    )

    print(f"\n结果已保存至 {SAVE_PATH}")


if __name__ == "__main__":
    main()