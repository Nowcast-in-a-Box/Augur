"""
MAU (Memory Attention Unit) PyTorch Lightning Module
将原始MAU模型封装为Lightning模块
"""

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from typing import Dict, Any, Optional, Tuple
from models.MAU.MAU import RNN
import math


class MAULightning(LightningModule):
    """
    MAU模型的PyTorch Lightning封装

    Args:
        img_width: 图像宽度
        img_height: 图像高度
        img_channel: 图像通道数（默认1，灰度图）
        patch_size: patch大小（默认1）
        num_layers: RNN层数（默认4）
        num_hidden: 隐藏层维度（默认64）
        filter_size: 卷积核大小（默认5）
        stride: 卷积步长（默认1）
        tau: 时间注意力窗口大小（默认5）
        cell_mode: 单元模式 ('residual' 或 'normal')
        model_mode: 模型模式 ('recall' 或 'normal')
        sr_size: 下采样倍数（默认1）
        input_length: 输入序列长度（默认10）
        total_length: 总序列长度（默认20）
        lr: 学习率（默认1e-3）
        lr_decay: 学习率衰减率（默认0.99）
        sampling_stop_iter: 停止采样的迭代次数（默认50000）
        delay_interval: 学习率衰减间隔（默认1000）
    """

    def __init__(
        self,
        img_width: int = 128,
        img_height: int = 128,
        img_channel: int = 1,
        patch_size: int = 1,
        num_layers: int = 4,
        num_hidden: int = 64,
        filter_size: int = 5,
        stride: int = 1,
        tau: int = 5,
        cell_mode: str = "residual",
        model_mode: str = "recall",
        sr_size: int = 1,
        input_length: int = 10,
        total_length: int = 20,
        lr: float = 1e-3,
        lr_decay: float = 0.99,
        sampling_stop_iter: int = 50000,
        sampling_start_value: float = 1.0,
        delay_interval: int = 1000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 创建配置对象（模拟原始configs）
        class Configs:
            pass

        self.configs = Configs()
        self.configs.img_width = img_width
        self.configs.img_height = img_height
        self.configs.img_channel = img_channel
        self.configs.patch_size = patch_size
        self.configs.num_layers = num_layers
        self.configs.num_hidden = num_hidden
        self.configs.filter_size = (filter_size, filter_size)
        self.configs.stride = stride
        self.configs.tau = tau
        self.configs.cell_mode = cell_mode
        self.configs.model_mode = model_mode
        self.configs.sr_size = sr_size
        self.configs.input_length = input_length
        self.configs.total_length = total_length
        self.configs.lr = lr
        self.configs.lr_decay = lr_decay
        self.configs.sampling_stop_iter = sampling_stop_iter
        self.configs.sampling_start_value = sampling_start_value
        self.configs.sampling_delta_per_iter = (
            sampling_start_value - 0.0
        ) / sampling_stop_iter
        self.configs.delay_interval = delay_interval
        self.configs.device = "cpu"  # Lightning会自动处理设备
        self.configs.model_name = "mau"

        # 创建隐藏层维度列表
        num_hidden_list = [num_hidden] * num_layers

        # 初始化MAU网络
        self.network = RNN(num_layers, num_hidden_list, self.configs)

        # 损失函数
        self.mse_criterion = nn.MSELoss()
        self.l1_criterion = nn.L1Loss()

        # 用于学习率调度的计数器
        self.training_step_count = 0

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            frames: 输入帧序列 (B, T, C, H, W)
            mask: 掩码 (B, target_length, H, W, C) MAU内部会permute

        Returns:
            预测的未来帧序列 (B, total_length-input_length, C, H, W)
            只返回未来的预测帧，不包括对输入帧的重建
        """
        # MAU network返回所有时间步的预测 (B, total_length-1, C, H, W)
        all_predictions = self.network(frames, mask)

        # 只返回未来的预测帧：从input_length-1开始的所有帧
        # all_predictions[:, 0] 对应 t=1（基于t=0预测）
        # all_predictions[:, input_length-1] 对应 t=input_length（第一个未来帧）
        # 提取索引 [input_length-1 : total_length-1]
        future_predictions = all_predictions[
            :, self.configs.input_length - 1 :, :, :, :
        ]

        return future_predictions

    def _shared_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix: str
    ) -> Dict[str, torch.Tensor]:
        """
        训练/验证的共享步骤

        Args:
            batch: 包含'inputs'和'targets'的字典
            batch_idx: 批次索引
            prefix: 'train'或'val'

        Returns:
            包含loss信息的字典
        """
        # 获取数据
        # batch['inputs']: (B, input_length, H, W)
        # batch['targets']: (B, target_length, H, W)
        inputs = batch["sequence"]  # (B, input_length, H, W)
        targets = batch["target"]  # (B, target_length, H, W)

        batch_size = inputs.shape[0]
        input_length = inputs.shape[1]
        target_length = targets.shape[1]

        # 验证输入尺寸是否与模型配置匹配
        assert (
            input_length == self.configs.input_length
        ), f"输入长度不匹配: 期望{self.configs.input_length}, 实际{input_length}"
        assert (
            input_length + target_length == self.configs.total_length
        ), f"总长度不匹配: 期望{self.configs.total_length}, 实际{input_length + target_length}"

        # 添加通道维度并合并
        inputs = inputs.unsqueeze(2)  # (B, input_length, 1, H, W)
        targets = targets.unsqueeze(2)  # (B, target_length, 1, H, W)

        # 合并inputs和targets为完整序列
        frames = torch.cat([inputs, targets], dim=1)  # (B, total_length, 1, H, W)

        # 创建mask (在训练时使用scheduled sampling，在验证时使用预测值)
        # mask需要是 (B, target_length, H, W, 1) 格式
        # 因为MAU内部会permute(0, 1, 4, 2, 3)将其转为(B, T, C, H, W)
        # mask=1: 使用真实值(teacher forcing), mask=0: 使用预测值(autoregressive)
        if prefix == "train":
            # Scheduled sampling: eta从1.0衰减到0.0
            eta = self.configs.sampling_start_value - (
                self.training_step_count * self.configs.sampling_delta_per_iter
            )
            eta = max(0.0, min(1.0, eta))  # 限制在[0,1]

            # eta概率使用真实值，(1-eta)概率使用预测值
            random_flip = torch.rand(
                batch_size, target_length, inputs.shape[3], inputs.shape[4], 1
            ).to(self.device)
            mask = (random_flip < eta).float()

            # 记录teacher forcing比例
            self.log("train_teacher_forcing_ratio", eta, prog_bar=True, on_step=True)
        else:
            # 验证时完全使用预测值（与训练后期一致）
            mask = torch.zeros(
                batch_size, target_length, inputs.shape[3], inputs.shape[4], 1
            ).to(self.device)

        # 前向传播
        predictions = self(frames, mask)  # (B, target_length, C, H, W)

        # 计算损失（只对未来的预测部分计算）
        # ground_truth: 未来的target_length帧
        ground_truth = frames[:, input_length:, :, :, :]  # (B, target_length, C, H, W)

        loss_mse = self.mse_criterion(predictions, ground_truth)
        loss_l1 = self.l1_criterion(predictions, ground_truth)

        # 主要损失是MSE
        loss = loss_mse

        # 记录指标
        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_mse", loss_mse, prog_bar=True)
        self.log(f"{prefix}_l1", loss_l1)

        return {
            "loss": loss,
            "mse": loss_mse,
            "l1": loss_l1,
            "predictions": predictions.detach(),
            "ground_truth": ground_truth.detach(),
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """训练步骤"""
        self.training_step_count += 1
        result = self._shared_step(batch, batch_idx, "train")
        return result["loss"]

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """验证步骤"""
        return self._shared_step(batch, batch_idx, "val")

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 每个step调用一次
                "frequency": self.hparams.delay_interval,  # 每delay_interval步调用一次
            },
        }

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """预测步骤"""
        inputs = batch["inputs"].unsqueeze(2)  # (B, input_length, 1, H, W)

        # 如果batch中有targets，使用它；否则创建零张量用于padding
        if "targets" in batch:
            targets = batch["targets"].unsqueeze(2)  # (B, target_length, 1, H, W)
        else:
            # 纯推理模式：创建零值targets
            target_length = self.configs.total_length - self.configs.input_length
            targets = torch.zeros(
                inputs.shape[0], target_length, 1, inputs.shape[3], inputs.shape[4]
            ).to(self.device)

        frames = torch.cat([inputs, targets], dim=1)

        # 预测时不使用真实值
        mask = torch.zeros(
            inputs.shape[0], targets.shape[1], inputs.shape[3], inputs.shape[4], 1
        ).to(self.device)

        predictions = self(frames, mask)  # 已经是未来帧 (B, target_length, C, H, W)
        return predictions

    def get_model_summary(self) -> str:
        """获取模型摘要"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        summary = f"""
MAU Model Summary:
==================
Architecture:
  - Image Size: {self.hparams.img_height}x{self.hparams.img_width}
  - Channels: {self.hparams.img_channel}
  - Patch Size: {self.hparams.patch_size}
  - SR Size: {self.hparams.sr_size}
  - Num Layers: {self.hparams.num_layers}
  - Hidden Dim: {self.hparams.num_hidden}
  - Filter Size: {self.hparams.filter_size}
  - Tau: {self.hparams.tau}
  - Cell Mode: {self.hparams.cell_mode}
  - Model Mode: {self.hparams.model_mode}

Sequence:
  - Input Length: {self.hparams.input_length}
  - Total Length: {self.hparams.total_length}
  - Output Length: {self.hparams.total_length - self.hparams.input_length}

Training:
  - Learning Rate: {self.hparams.lr}
  - LR Decay: {self.hparams.lr_decay}

Parameters:
  - Total: {total_params:,}
  - Trainable: {trainable_params:,}
==================
"""
        return summary


# 兼容旧版本的工厂函数
def create_mau_model(
    img_width: int = 128, img_height: int = 128, img_channel: int = 1, **kwargs
) -> MAULightning:
    """
    创建MAU Lightning模型的工厂函数

    Args:
        img_width: 图像宽度
        img_height: 图像高度
        img_channel: 图像通道数
        **kwargs: 其他参数传递给MAULightning

    Returns:
        MAULightning模型实例
    """
    return MAULightning(
        img_width=img_width, img_height=img_height, img_channel=img_channel, **kwargs
    )
