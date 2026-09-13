from typing import Dict
from pytorch_lightning import LightningModule
import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanMetric, Metric
import numpy as np
from torch.nn import functional as F


def compute_psd(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    assert (
        pred.dim() == 4 and target.dim() == 4
    ), "Input tensors must be 4D (B, C, H, W)"

    _, _, h, w = pred.shape
    
    pred = pred.float()
    target = target.float()

    fft_pred = torch.fft.fft2(pred, dim=(-2, -1))
    fft_target = torch.fft.fft2(target, dim=(-2, -1))

    power_pred = torch.abs(fft_pred) ** 2 / (h * w)
    power_target = torch.abs(fft_target) ** 2 / (h * w)

    return torch.nn.functional.mse_loss(power_pred, power_target)


def compute_csi(
    pred: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    tp = torch.sum((pred == 1) & (target == 1))
    fp = torch.sum((pred == 1) & (target == 0))
    fn = torch.sum((pred == 0) & (target == 1))

    csi = tp / (tp + fp + fn)

    if (tp + fp + fn) > 0:
        return csi
    else:
        return torch.tensor([0.0], device=pred.device, dtype=pred.dtype)


def compute_csi_mean(
    pred: torch.Tensor, target: torch.Tensor,
):
    r = []
    for th in THRESHOLDS:
        r.append(compute_csi(pred, target, th))
    
    r = torch.tensor(r, device=pred.device).mean()
    
    return r


def log_loss(
    pred: torch.Tensor,
    truth: torch.Tensor,
    stage: str,
    lightning_module: LightningModule,
):
    # Root Mean Square Error
    rmse = compute_rmse(pred, truth)
    lightning_module.log(f"{stage}_rmse", rmse, sync_dist=True)

    # CSI Mean
    csi_mean = compute_csi_mean(pred * 255, truth * 255)
    lightning_module.log(f"{stage}_csi_mean", csi_mean, sync_dist=True, prog_bar=True)

    # CSI 181 Threshold
    csi_181 = compute_csi(pred * 255, truth * 255, threshold=THRESHOLDS[-2])
    lightning_module.log(f"{stage}_csi_181", csi_181, sync_dist=True)

    # CSI 219 Threshold
    csi_219 = compute_csi(pred * 255, truth * 255, threshold=THRESHOLDS[-1])
    lightning_module.log(f"{stage}_csi_219", csi_219, sync_dist=True)
    
    ssim_value = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
    ssim_value = ssim_value(pred, truth)
    lightning_module.log(f"{stage}_ssim", ssim_value, sync_dist=True)


def compute_loss(pred: torch.Tensor, truth: torch.Tensor):
    # CSI - 10mm Threshold
    csi_181 = compute_csi(pred * 255, truth * 255, threshold=181)

    # CSI - 25mm Threshold
    csi_219 = compute_csi(pred * 255, truth * 255, threshold=209.114269587373)


class RAPSDE:
    def __init__(self, img_size: int, pixel_spacing_distance: float = 3.0):
        self.size = img_size
        self.pixel_spacing = pixel_spacing_distance

        y, x = torch.meshgrid(
            torch.arange(img_size),
            torch.arange(img_size),
            indexing="ij"
        )

        center_y, center_x = (img_size - 1) / 2.0, (img_size - 1) / 2.0
        r = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        self.R = r.round().long()
        self.max_R = img_size // 2

        self.r_count_full = torch.bincount(self.R.flatten())
        self.valid_r = self.r_count_full[1: self.max_R + 1]

        k = torch.arange(1, self.max_R + 1)
        total_len_km = img_size * pixel_spacing_distance
        self.wave_length = total_len_km / k

    def compute(self, x: torch.Tensor):
        if x.dim() == 4:
            x = x.view(-1, self.size, self.size)

        device = x.device
        x = x - x.mean(dim=(1, 2), keepdim=True)

        fft_x = torch.fft.fft2(x)
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))
        power_2d = torch.abs(fft_shift) ** 2
        avg_power_2d = power_2d.mean(dim=0)

        radial_sum = torch.bincount(
            self.R.flatten().to(device), weights=avg_power_2d.flatten().to(device)
        )
        radial_engergy = radial_sum[1: self.max_R + 1]
        radial_engergy = radial_engergy / self.valid_r.to(device)

        return 10 * torch.log10(radial_engergy)


class SpectralMeanMetric(MeanMetric):

    def __init__(self, num_bins=64):
        super().__init__()
        self.add_state("sum", default=torch.zeros(num_bins), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, value: torch.Tensor):
        self.sum += value
        self.count += 1

    def compute(self):
        return self.sum / self.count


class CSI(Metric):

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def __str__(self):
        return f"CSI|{self.threshold}"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()

        self.tp += torch.sum((pred == 1) & (target == 1))
        self.fp += torch.sum((pred == 1) & (target == 0))
        self.fn += torch.sum((pred == 0) & (target == 1))

    def compute(self):
        denominator = self.tp + self.fp + self.fn
        if denominator > 0.0:
            return self.tp / denominator
        else:
            return torch.tensor(0.0)

class POD(Metric):
    """
    Probability of Detection (POD)
    """

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.tensor(0), dist_reduce_fx="sum")

    def __str__(self):
        return f"POD|{self.threshold}"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()

        self.tp += torch.sum((pred == 1) & (target == 1))
        self.fn += torch.sum((pred == 0) & (target == 1))

    def compute(self):
        denominator = self.tp + self.fn
        if denominator > 0.0:
            return self.tp / denominator
        else:
            return torch.tensor(0.0)


class FAR(Metric):
    """
    False Alarm Rate (FAR)
    """

    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")

    def __str__(self):
        return f"FAR|{self.threshold}"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()

        self.tp += torch.sum((pred == 1) & (target == 1))
        self.fp += torch.sum((pred == 1) & (target == 0))

    def compute(self):
        denominator = self.tp + self.fp
        if denominator > 0.0:
            return self.tp / denominator
        else:
            return torch.tensor(0.0)


class FAR(Metric):
    """
    False Alarm Rate (FAR)
    """
    
    def __init__(self, threshold: float):
        super().__init__()
        
        self.threshold = threshold
        
        
        self.add_state("tp", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def __str__(self):
        return f"FAR|{self.threshold}"
    
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        
    
        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()
    
        self.tp += torch.sum((pred == 1) & (target == 1))
        self.fp += torch.sum((pred == 1) & (target == 0))

    
    
    def compute(self):
        
        denominator = self.tp + self.fp
        
        if denominator > 0.0:
            return self.tp / denominator
        else:
            return torch.tensor(0.0)

THRESHOLDS = (16, 74, 133, 160, 181, 219)


class FSS(Metric):
    """
    Fraction Skill Score (FSS)
    """
    
    def __init__(self, threshold: float, scale: int, eps: float = 1e-8):
        """FSS

        Args:
            threshold (float): threshold for binarization
            scale (int): must be odd, for nerbourghood window size
        """
        super().__init__()
        self.threshold = threshold
        self.scale = scale
        self.eps = eps

        self.add_state("mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mse_ref", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def __str__(self):
        return f"FSS|{self.threshold}|scale-{self.scale}"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = (pred >= self.threshold).float()
        target = (target >= self.threshold).float()

        pad = (self.scale - 1) // 2
        pred = F.avg_pool2d(pred, kernel_size=self.scale, stride=1, padding=pad, count_include_pad=False)
        target = F.avg_pool2d(target, kernel_size=self.scale, stride=1, padding=pad, count_include_pad=False)

        mse = torch.mean((pred - target) ** 2, dim=(-2, -1))
        mse_ref = torch.mean(pred ** 2, dim=(-2, -1)) + torch.mean(target ** 2, dim=(-2, -1))

        self.mse += mse.sum()
        self.mse_ref += mse_ref.sum()
        self.count += mse.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(float('nan'))

        fss = 1.0 - self.mse / (self.mse_ref + self.eps)
        return fss.clamp(0.0, 1.0)




class CSIMean(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("tp", default=torch.zeros(len(THRESHOLDS)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(THRESHOLDS)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(THRESHOLDS)), dist_reduce_fx="sum")

    def __str__(self):
        return f"CSI-Mean"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for i, thre in enumerate(THRESHOLDS):
            p = (pred >= thre).float()
            t = (target >= thre).float()

            self.tp[i] += torch.sum((p == 1) & (t == 1))
            self.fp[i] += torch.sum((p == 1) & (t == 0))
            self.fn[i] += torch.sum((p == 0) & (t == 1))

    def compute(self, eps: float = 1e-6):
        csi_per_threshold = self.tp / (self.tp + self.fn + self.fp)
        return csi_per_threshold.mean()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor):
    return torch.sqrt(torch.mean((pred - target) ** 2))


class RMSE(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("rmse", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")

    def __str__(self):
        return f"RMSE"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.rmse += torch.sqrt(torch.mean(
            (unit_transform(pred) - unit_transform(target)) ** 2
        ))
        self.count += 1

    def compute(self):
        return self.rmse / self.count


class PeakRatio(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("pred_sum", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("truth_sum", torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.pred_sum += unit_transform(pred).max()
        self.truth_sum += unit_transform(target).max()

    def compute(self):
        return self.pred_sum / self.truth_sum


def unit_transform(x: torch.Tensor):
    result = torch.zeros_like(x)

    mask = (x > 5.0) & (x <= 18.0)
    result = torch.where(mask, (x - 2.0) / 90.66, result)

    mask = (x > 18.0) & (x <= 254)
    result = torch.where(mask, torch.exp((x - 83.9) / 38.9), result)

    return result


class QQStatistic:

    def __init__(self, device="cpu"):
        self.device = device
        self.pred_buffer = []
        self.truth_buffer = []

    def update(self, pred: torch.Tensor, truth: torch.Tensor):
        p_flat = pred.flatten()
        t_flat = truth.flatten()

        self.pred_buffer.append(p_flat[p_flat > 0.1])
        self.truth_buffer.append(t_flat[t_flat > 0.1])

    def compute(self):
        preds = torch.cat(self.pred_buffer).cpu().numpy()
        truth = torch.cat(self.truth_buffer).cpu().numpy()

        p_base = np.linspace(0, 0.9, 150)
        p_mid = np.linspace(0.9, 0.99, 50)
        p_tail = np.linspace(0.99, 1.0, 100)

        probs = np.concatenate([p_base[:-1], p_mid[:-1], p_tail])

        q_preds = np.quantile(preds, probs, method="linear")
        q_truth = np.quantile(truth, probs, method="linear")

        return torch.from_numpy(q_preds).to(self.device), torch.from_numpy(q_truth).to(self.device)


class SSIM(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("ssim", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", torch.tensor(0.0), dist_reduce_fx="sum")
        self.ssim_value = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def __str__(self):
        return f"SSIM"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        self.ssim += self.ssim_value(pred / 255, target / 255)
        self.count += 1

    def compute(self):
        return self.ssim / self.count


class HeidkeSkillScore(Metric):

    def __init__(self, thresholds: tuple = THRESHOLDS):
        super().__init__()
        self.thresholds = thresholds

        self.add_state("tp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(len(self.thresholds)), dist_reduce_fx="sum")

    def __str__(self):
        return f"HSS"

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        for i, thre in enumerate(self.thresholds):
            p = (pred >= thre).long()
            t = (target >= thre).long()

            self.tp[i] += torch.sum((p == 1) & (t == 1))
            self.fp[i] += torch.sum((p == 1) & (t == 0))
            self.fn[i] += torch.sum((p == 0) & (t == 1))
            self.tn[i] += torch.sum((p == 0) & (t == 0))

    def compute(self):
        numerator = 2 * (self.tp * self.tn - self.fp * self.fn)
        denominator = (self.tp + self.fn) * (self.fn + self.tn) + \
            (self.tp + self.fp) * (self.fp + self.tn)

        hss = numerator.float() / (denominator.float() + 1e-6)
        return hss.mean()