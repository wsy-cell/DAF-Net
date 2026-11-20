"""
Stage 1: 量化水下图像退化程度（复用 loss.py 中的指标）
目标：为每张图计算色偏误差和散射误差，归一化，判定退化主导类型
产出：退化分布图 + 归一化参数 + CSV标签文件
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
from collections import Counter


from dafnet.losses import Grayworld_Loss          
from dafnet.losses import dark_channel            

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


grayworld_loss_fn = Grayworld_Loss().to(device)

def load_image(image_path):
    """加载图像并转为 tensor [1, 3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)  
    return img_tensor

def compute_color_error(img_tensor):
    """
    色偏误差：直接调用 Grayworld_Loss
    返回值：标量 float
    """
    return grayworld_loss_fn(img_tensor).item()

def compute_haze_error(img_tensor, patch_size=15):
    """
    散射误差：基于暗通道均值（暗通道越亮，散射越重）
    调用 loss.py 中的 dark_channel 函数，然后取均值
    """
    dark = dark_channel(img_tensor)  
    
    dark_pooled = F.avg_pool2d(dark.unsqueeze(1), 
                               kernel_size=patch_size, 
                               stride=1, 
                               padding=patch_size//2)
    return dark_pooled.mean().item()

def main():
    
    input_dir = INPUT_DIR
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 1: 量化水下图像退化程度")
    print("=" * 60)

    image_files = sorted([f for f in os.listdir(input_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    print(f"\n[1/4] 找到 {len(image_files)} 张训练图像")
    print(f"[2/4] 正在计算退化误差...")

    results = []
    color_errors = []
    haze_errors = []

    for img_name in tqdm(image_files, desc="Processing", ncols=80):
        img_path = os.path.join(input_dir, img_name)
        try:
            img_tensor = load_image(img_path)
            c_err = compute_color_error(img_tensor)
            h_err = compute_haze_error(img_tensor)
            color_errors.append(c_err)
            haze_errors.append(h_err)
            results.append({
                'filename': img_name,
                'color_error': c_err,
                'haze_error': h_err
            })
        except Exception as e:
            print(f"  警告: 处理 {img_name} 时出错: {e}")
            continue

    color_errors = np.array(color_errors)
    haze_errors = np.array(haze_errors)

    print(f"\n[3/4] 原始误差统计:")
    print(f"  Color Error - mean: {color_errors.mean():.6f}, std: {color_errors.std():.6f}")
    print(f"  Haze Error  - mean: {haze_errors.mean():.6f}, std: {haze_errors.std():.6f}")

    
    mu_c, sigma_c = color_errors.mean(), color_errors.std()
    mu_h, sigma_h = haze_errors.mean(), haze_errors.std()
    sigma_c = max(sigma_c, 1e-8)
    sigma_h = max(sigma_h, 1e-8)

    color_z = (color_errors - mu_c) / sigma_c
    haze_z = (haze_errors - mu_h) / sigma_h

    
    norm_params = {
        'color_mean': float(mu_c),
        'color_std': float(sigma_c),
        'haze_mean': float(mu_h),
        'haze_std': float(sigma_h)
    }

    
    margin = 0.1
    dominance_scores = color_z - haze_z
    labels = []
    for d in dominance_scores:
        if d > margin:
            labels.append('color_dominant')
        elif d < -margin:
            labels.append('haze_dominant')
        else:
            labels.append('balanced')

    label_counts = Counter(labels)
    print(f"\n[4/4] 退化类型分布:")
    print(f"  色偏主导 (color_dominant): {label_counts['color_dominant']} 张 ({label_counts['color_dominant']/len(labels)*100:.1f}%)")
    print(f"  散射主导 (haze_dominant):  {label_counts['haze_dominant']} 张 ({label_counts['haze_dominant']/len(labels)*100:.1f}%)")
    print(f"  均衡 (balanced):           {label_counts['balanced']} 张 ({label_counts['balanced']/len(labels)*100:.1f}%)")

    
    csv_path = os.path.join(output_dir, 'degradation_labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'color_error', 'haze_error',
                         'color_zscore', 'haze_zscore', 'dominance', 'label'])
        for i, res in enumerate(results):
            writer.writerow([
                res['filename'],
                f"{res['color_error']:.8f}",
                f"{res['haze_error']:.8f}",
                f"{color_z[i]:.4f}",
                f"{haze_z[i]:.4f}",
                f"{dominance_scores[i]:.4f}",
                labels[i]
            ])

    import json
    json_path = os.path.join(output_dir, 'normalization_params.json')
    with open(json_path, 'w') as f:
        json.dump(norm_params, f, indent=2)

    
    plt.rcParams['font.family'] = 'DeJavu Serif'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_map = {'color_dominant': 'red', 'haze_dominant': 'blue', 'balanced': 'gray'}
    colors_list = [colors_map[l] for l in labels]

    ax = axes[0]
    ax.scatter(color_errors, haze_errors, c=colors_list, alpha=0.6, s=20)
    ax.set_xlabel('Color Error (Gray World Loss)', fontsize=11)
    ax.set_ylabel('Haze Error (Dark Channel Mean)', fontsize=11)
    ax.set_title('Original Error Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(color_z, haze_z, c=colors_list, alpha=0.6, s=20)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Color Error (Z-score)', fontsize=11)
    ax.set_ylabel('Haze Error (Z-score)', fontsize=11)
    ax.set_title('Normalized Error Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    
    ax = axes[2]
    
    
    color_data = dominance_scores[np.array(labels) == 'color_dominant'].ravel()
    haze_data = dominance_scores[np.array(labels) == 'haze_dominant'].ravel()
    balanced_data = dominance_scores[np.array(labels) == 'balanced'].ravel()
    
    if len(color_data) > 0:
        ax.hist(color_data, bins=30, alpha=0.7, color='red', 
                label=f"Color-dominant ({len(color_data)})")
    if len(haze_data) > 0:
        ax.hist(haze_data, bins=30, alpha=0.7, color='blue', 
                label=f"Haze-dominant ({len(haze_data)})")
    if len(balanced_data) > 0:
        ax.hist(balanced_data, bins=30, alpha=0.7, color='gray', 
                label=f"Balanced ({len(balanced_data)})")
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Dominance Score (Color - Haze)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Degradation Dominance Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'degradation_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n所有结果已保存到: {output_dir}/")
    print("第一阶段完成！")

if __name__ == "__main__":
    main()