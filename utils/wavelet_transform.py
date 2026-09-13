from typing import Dict, Any, Literal, Tuple, NamedTuple
import torch
import ptwt
from collections import namedtuple

# 可选：为细节方向命名（提高可读性）
DETAIL_NAMES = ["horizontal", "vertical", "diagonal"]


class WaveletCoeffs(NamedTuple):
    """
    结构化小波系数容器（基于 NamedTuple 仅用于类型提示与文档）。
    实际运行时返回 dict，字段名动态生成。
    """

    A: torch.Tensor
    # D1, D2, ... 动态生成，类型为 torch.Tensor，shape: (B, C, 3, H, W)


# 类型别名：清晰表达返回结构
WaveletCoeffDict = Dict[str, torch.Tensor]


class WaveletTransform:
    def __init__(
        self,
        wavelet: str,
        level: int,
        mode: Literal["constant", "reflect", "zero", "periodic"] = "reflect",
    ) -> None:
        if level < 1:
            raise ValueError("Level must be >= 1")
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def transform(self, input_tensor: torch.Tensor) -> WaveletCoeffDict:
        """
        执行二维小波分解并返回结构化系数字典。

        参数:
            input_tensor: torch.Tensor - 输入张量，shape: (B, C, H, W)

        返回:
            WaveletCoeffDict - 包含以下键：
                - "A" : 近似系数 (approximation)
                - "D1", "D2", ..., "D{level}" : 各层细节系数
                  每个 D{l} 的 shape: (B, C, 3, H_l, W_l)
                  其中第3维为 [horizontal, vertical, diagonal]
        """
        coeffs = ptwt.wavedec2(
            input_tensor,
            wavelet=self.wavelet,
            level=self.level,
            mode=self.mode,
        )

        # coeffs[0]: approximation
        # coeffs[1:]: detail tuples per level
        result: WaveletCoeffDict = {"A": coeffs[0]}

        for l in range(1, self.level + 1):
            level_details = coeffs[l]  # Tuple[Tensor, Tensor, Tensor]
            if len(level_details) != 3:
                raise RuntimeError(
                    f"Expected 3 detail coefficients at level {l}, got {len(level_details)}"
                )
            # 拼接为 (B, C, 3, H, W)
            stacked = torch.stack(level_details, dim=2)  # dim=2 → 通道后插入方向维度
            result[f"D{l}"] = stacked

        return result

    def reverse(self, input: WaveletCoeffDict) -> torch.Tensor:
        """
        执行二维小波重构。

        参数:
            input: WaveletCoeffDict - 包含 "A" 和 "D1", "D2", ..., "D{level}" 的系数字典

        返回:
            torch.Tensor - 重构后的张量，shape: (B, C, H, W)
        """

        coeffs_list = [input["A"]]
        

        for l in range(1, self.level + 1):
            d_tensor = input[f"D{l}"]  # (B, C, 3, H, W)
            if d_tensor.shape[2] != 3:
                raise ValueError(f"D{l} must have 3 detail directions, got shape {d_tensor.shape}")

            # 拆分通道：horizontal, vertical, diagonal
            H, V, D = torch.unbind(d_tensor, dim=2)  # 各 (B, C, H, W)
            coeffs_list.append((H, V, D))

        reconstructed = ptwt.waverec2(coeffs_list, wavelet=self.wavelet)

        return reconstructed.clone().contiguous()