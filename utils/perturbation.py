from typing import Tuple, Optional
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
import random


class GaussianBlur:
    """使用 torchvision.transforms.v2.GaussianBlur 包装，更高效且简洁。"""

    def __init__(self, kernel_size=(3, 5), sigma=(1.0, 2.0)):
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, input_tensor: torch.Tensor):
        B, T, H, W = input_tensor.shape
        # v2 变换默认处理 (..., C, H, W)，因此我们将 T 维度暂时合并到 B
        x = input_tensor.view(B * T, 1, H, W)
        x_blurred = self.transform(x)
        return x_blurred.view(B, T, H, W)


class DownsampleUpsample:
    def __init__(self, scale_factors: Tuple[int, ...] = (2, 3, 4)):
        self.scale_factors = scale_factors

    def __call__(self, tensor: torch.Tensor):
        B,T,H,W = tensor.shape
        scale_factor = random.choice(self.scale_factors)
        # 使用 avg_pool2d 进行降采样
        down = F.avg_pool2d(tensor, kernel_size=scale_factor)
        up = F.interpolate(
            down,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        return up


class TruncateExtremes:
    """按每个样本独立计算分位数并进行裁剪。"""

    def __init__(self, quantiles=(0.8, 0.9, 0.95)):
        self.quantiles = quantiles

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        # 将张量重塑为 (B*T, H*W)，以便按每个图像计算分位数

        q_val = random.choice(self.quantiles)

        return torch.clamp(x, max=q_val)


class AdditiveNoise:
    """添加高斯噪声。"""

    def __init__(self, std_dev_range=(0.01, 0.05)):
        self.std_dev_range = std_dev_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        std = random.uniform(*self.std_dev_range)
        noise = torch.randn_like(x) * std
        return x + noise


class AdditiveBias:
    """添加平滑的、低频的系统偏差场。"""

    def __init__(self, scale_factor=8, strength_range=(0.1, 0.3)):
        self.scale_factor = scale_factor
        self.strength_range = strength_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        strength = random.uniform(*self.strength_range)

        # 创建一个低分辨率的随机噪声场
        bias_low_res = torch.randn(
            B, T, H // self.scale_factor, W // self.scale_factor, device=x.device
        )

        # 通过插值放大到原始分辨率，形成平滑的偏差场
        bias_field = F.interpolate(
            bias_low_res, size=(H, W), mode="bilinear", align_corners=False
        )

        # 将偏差场缩放到数据标准差的一定比例
        data_std = x.std()
        if data_std > 1e-6:  # 避免除以零
            bias_field = bias_field * data_std * strength

        return x + bias_field


class Perturbation:
    def __init__(
        self, blur_p=0.5, downsample_p=0.5, truncate_p=0.3, noise_p=0.5, bias_p=0.3
    ):

        assert 0.0 <= blur_p <= 1.0, "blur_p must be in [0, 1]"
        assert 0.0 <= downsample_p <= 1.0, "downsample_p must be in [0, 1]"
        assert 0.0 <= truncate_p <= 1.0, "truncate_p must be in [0, 1]"
        assert 0.0 <= noise_p <= 1.0, "noise_p must be in [0, 1]"
        assert 0.0 <= bias_p <= 1.0, "bias_p must be in [0, 1]"

        self.transforms = transforms.Compose(
            [
                # 1. 核心退化操作：模糊和降采样，模拟分辨率损失
                transforms.RandomApply(
                    [GaussianBlur(kernel_size=(3, 5), sigma=(1.0, 2.5))], p=blur_p
                ),
                transforms.RandomApply(
                    [DownsampleUpsample(scale_factors=(2, 3, 4))], p=downsample_p
                ),
                # 2. 模拟数值偏差
                transforms.RandomApply(
                    [TruncateExtremes(quantiles=(0.85, 0.9, 0.95))], p=truncate_p
                ),
                transforms.RandomApply(
                    [AdditiveBias(scale_factor=8, strength_range=(0.1, 0.3))], p=bias_p
                ),
                # 3. 最后加入随机噪声
                transforms.RandomApply(
                    [AdditiveNoise(std_dev_range=(0.01, 0.05))], p=noise_p
                ),
            ]
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.transforms(tensor)
