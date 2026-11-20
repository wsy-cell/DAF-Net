"""
PDS-MPNet Stage3 数据集
- 支持加载冻结预处理的辅助图像（CLAHE + 灰度世界）
- 支持加载退化分析标签用于渐进式 ASC 训练
- 支持数据增强（ColorJitter, RandomFlip, RandomCrop）
"""

import os
import csv
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from torchvision import transforms
import cv2


class PDSImageDataset(Dataset):
    def __init__(self, input_dir, label_dir, csv_path=None, transform=None, 
                 use_preprocess=True, augment=False, use_degradation_labels=True):
        """
        input_dir:  退化输入图像目录
        label_dir:  清晰 GT 图像目录
        csv_path:   退化分析标签 CSV（可选，用于渐进式 ASC 训练）
        transform:  基础图像预处理 (Resize + ToTensor)
        use_preprocess: 是否使用冻结预处理生成辅助通道
        augment:    是否启用数据增强
        use_degradation_labels: 是否加载退化标签
        """
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.use_preprocess = use_preprocess
        self.augment = augment
        self.use_degradation_labels = use_degradation_labels

        
        self.image_files = sorted([f for f in os.listdir(input_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        self.label_files = sorted([f for f in os.listdir(label_dir)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        assert len(self.image_files) == len(self.label_files), \
            "输入和 GT 图像数量不一致"

        
        self.degradation_map = {}
        if use_degradation_labels and csv_path and os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row['filename']
                    label = row.get('label', 'balanced')
                    
                    if label == 'color_dominant':
                        target_ws, target_wc = 0.3, 0.7
                    elif label == 'haze_dominant':
                        target_ws, target_wc = 0.7, 0.3
                    else:
                        target_ws, target_wc = 0.5, 0.5
                    self.degradation_map[fname] = (target_ws, target_wc)

        
        self.color_jitter = transforms.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.1, hue=0.02
        )
        self.random_hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.random_vflip = transforms.RandomVerticalFlip(p=0.2)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        gt_name  = self.label_files[idx]

        input_path = os.path.join(self.input_dir, img_name)
        label_path = os.path.join(self.label_dir, gt_name)

        
        input_img = Image.open(input_path).convert('RGB')
        label_img = Image.open(label_path).convert('RGB')

        
        if self.use_preprocess:
            
            input_np = np.array(input_img)
            lab = cv2.cvtColor(input_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            lab_clahe = cv2.merge([l_clahe, a, b])
            img_clahe = Image.fromarray(cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB))

            
            img_wb = self._gray_world_balance(input_img)

            
            if self.transform:
                img_raw_tensor  = self.transform(input_img)
                img_clahe_tensor = self.transform(img_clahe)
                img_wb_tensor    = self.transform(img_wb)
            else:
                img_raw_tensor   = transforms.ToTensor()(input_img)
                img_clahe_tensor = transforms.ToTensor()(img_clahe)
                img_wb_tensor    = transforms.ToTensor()(img_wb)

            input_tensor = torch.cat([img_raw_tensor, img_clahe_tensor, img_wb_tensor], dim=0)  
        else:
            
            if self.transform:
                input_tensor = self.transform(input_img)
            else:
                input_tensor = transforms.ToTensor()(input_img)

        if self.transform:
            label_tensor = self.transform(label_img)
        else:
            label_tensor = transforms.ToTensor()(label_img)

        
        if self.augment:
            
            if input_tensor.size(0) == 9:
                raw = input_tensor[:3]
                raw_aug = self.color_jitter(raw)
                input_tensor = torch.cat([raw_aug, input_tensor[3:]], dim=0)
            else:
                input_tensor = self.color_jitter(input_tensor)

            
            if torch.rand(1).item() < 0.5:
                input_tensor = torch.flip(input_tensor, dims=[2])
                label_tensor = torch.flip(label_tensor, dims=[2])
            if torch.rand(1).item() < 0.2:
                input_tensor = torch.flip(input_tensor, dims=[1])
                label_tensor = torch.flip(label_tensor, dims=[1])

        
        if self.use_degradation_labels and img_name in self.degradation_map:
            target_ws, target_wc = self.degradation_map[img_name]
        else:
            target_ws, target_wc = 0.5, 0.5  

        return (input_tensor, label_tensor,
                torch.tensor([target_ws], dtype=torch.float32),
                torch.tensor([target_wc], dtype=torch.float32),
                os.path.splitext(img_name)[0])

    def _gray_world_balance(self, img):
        """简单的灰度世界白平衡"""
        img_np = np.array(img).astype(np.float32)
        r_mean, g_mean, b_mean = img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()
        gray_mean = (r_mean + g_mean + b_mean) / 3.0
        scale_r = gray_mean / (r_mean + 1e-6)
        scale_g = gray_mean / (g_mean + 1e-6)
        scale_b = gray_mean / (b_mean + 1e-6)
        img_np[:,:,0] = np.clip(img_np[:,:,0] * scale_r, 0, 255)
        img_np[:,:,1] = np.clip(img_np[:,:,1] * scale_g, 0, 255)
        img_np[:,:,2] = np.clip(img_np[:,:,2] * scale_b, 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))