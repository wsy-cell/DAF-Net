import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor


def save_image(name, image_np, output_path=""):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def pil_to_np(img_PIL, with_transpose=True):
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]
    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()[0]


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def get_A(x):
    x = torch_to_np(x)
    x_np = np.clip(x, 0, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)
    return A.unsqueeze(0)


from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torch.nn.functional as F


def calculate_ssim(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return ssim(img1, img2)


def calculate_psnr(img1, img2):
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    mse = F.mse_loss(img1, img2)
    max_pixel_value = 1.0
    psnr = 10 * torch.log10(max_pixel_value ** 2 / (mse + 1e-8))
    return psnr


def rgb2xyz(rgb):
    mask = (rgb > .04045).type(torch.FloatTensor)
    if rgb.is_cuda:
        mask = mask.cuda()
    rgb = (((rgb + .055) / 1.055) ** 2.4) * mask + rgb / 12.92 * (1 - mask)
    x = .412453 * rgb[:, 0, :, :] + .357580 * rgb[:, 1, :, :] + .180423 * rgb[:, 2, :, :]
    y = .212671 * rgb[:, 0, :, :] + .715160 * rgb[:, 1, :, :] + .072169 * rgb[:, 2, :, :]
    z = .019334 * rgb[:, 0, :, :] + .119193 * rgb[:, 1, :, :] + .950227 * rgb[:, 2, :, :]
    out = torch.cat((x[:, None, :, :], y[:, None, :, :], z[:, None, :, :]), dim=1)
    return out


def xyz2lab(xyz):
    sc = torch.Tensor((0.95047, 1., 1.08883))[None, :, None, None]
    if xyz.is_cuda:
        sc = sc.cuda()
    xyz_scale = xyz / sc
    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if xyz_scale.is_cuda:
        mask = mask.cuda()
    xyz_int = xyz_scale ** (1 / 3.) * mask + (7.787 * xyz_scale + 16. / 116.) * (1 - mask)
    L = 116. * xyz_int[:, 1, :, :] - 16.
    a = 500. * (xyz_int[:, 0, :, :] - xyz_int[:, 1, :, :])
    b = 200. * (xyz_int[:, 1, :, :] - xyz_int[:, 2, :, :])
    out = torch.cat((L[:, None, :, :], a[:, None, :, :], b[:, None, :, :]), dim=1)
    return out


def rgb2lab(rgb, l_norm=100., l_cent=50., ab_norm=110.):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:, [0], :, :] - l_cent) / l_norm
    ab_rs = lab[:, 1:, :, :] / ab_norm
    out = torch.cat((l_rs, ab_rs), dim=1)
    return out


import sys
class Logger:
    """同时输出到控制台和日志文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8") 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()