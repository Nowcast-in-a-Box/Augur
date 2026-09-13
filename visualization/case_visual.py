import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import os
from matplotlib import gridspec

# ==========================================
# 1. 物理数值转换 (保持不变，用于科学准确性)
# ==========================================


def vil_transform(x_uint8):
    """
    将 0-255 的像素值转换为物理 VIL 值 (kg/m²)
    """
    x = x_uint8.astype(np.float32)
    y = np.zeros_like(x)

    # 0 <= x <= 5 -> 0
    # 5 < x <= 18
    mask1 = (x > 5) & (x <= 18)
    y[mask1] = (x[mask1] - 2) / 90.66
    # 18 < x <= 255
    mask2 = x > 18
    y[mask2] = np.exp((x[mask2] - 83.9) / 38.9)
    return y


# ==========================================
# 2. 配色方案 (回归原始配色，但适配物理数值)
# ==========================================


def get_original_colormap_with_physics():
    """
    使用用户提供的原始 HEX 配色，但将边界映射到物理数值
    """
    # 原始提供的 HEX 颜色列表
    colors = [
        "#808080",  # 0-16
        "#90EE90",  # 16-31
        "#32CD32",  # 31-59
        "#228B22",  # 59-74
        "#006400",  # 74-100
        "#FFFF00",  # 100-133
        "#FFD700",  # 133-160
        "#FFA500",  # 160-181
        "#DC143C",  # 181-219
        "#8B008B",  # 219-255
    ]

    # 原始提供的像素边界
    pixel_boundaries = np.array([0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255])

    # 关键步骤：将像素边界转换为物理数值边界
    # 这样 Colorbar 的刻度显示的是真实的 kg/m^2，但颜色分布与原来完全一致
    phys_boundaries = vil_transform(pixel_boundaries)

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(phys_boundaries, cmap.N)

    return cmap, norm, phys_boundaries


# ==========================================
# 3. 数据加载与处理
# ==========================================


def load_data(case_study_path):
    pt_files = {}
    files = [f for f in os.listdir(case_study_path) if f.endswith(".pt")]

    print("正在加载数据...")
    for file in files:
        name = file.replace(".pt", "")
        path = os.path.join(case_study_path, file)
        data = torch.load(path, map_location="cpu", weights_only=False)

        # 统一转为物理数值
        data_np = data.squeeze().detach().numpy()
        # 确保先还原到 0-255 整数
        data_uint8 = (np.clip(data_np, 0, 1) * 255).astype(np.uint8)
        # 再转为物理值
        pt_files[name] = vil_transform(data_uint8)

    return pt_files


# ==========================================
# 4. 绘图主逻辑
# ==========================================


def visualize_layout_v3(case_study_path, save_path):
    # 1. 准备资源
    data = load_data(case_study_path)
    cmap, norm, boundaries = get_original_colormap_with_physics()

    # 2. 定义显示顺序
    # 逻辑：Sequence -> Observation -> 其他模型
    models_to_plot = []

    # 添加 Sequence
    if "sequence" in data:
        models_to_plot.append("Sequence")  # 临时改名用于显示，取数据时要注意

    # 添加 Observation (原 target)
    if "target" in data:
        data["Observation"] = data.pop("target")  # 重命名数据键
        models_to_plot.append("Observation")

    # 添加其他模型 (按字母或指定顺序)
    other_models = [
        "ConvLSTM",
        "UNet",
        "EarthFarseer",
        "SimVP",
        "AlphaPre",
        "StormWave",
    ]
    for m in other_models:
        if m in data:
            models_to_plot.append(m)

    # 3. 设置画板
    n_rows = len(models_to_plot)
    n_cols = 6

    # 动态调整高度
    fig = plt.figure(figsize=(16, 2.3 * n_rows))

    # GridSpec 布局：左侧留给名字，中间放图，右侧留给 Colorbar
    gs = gridspec.GridSpec(
        n_rows,
        n_cols + 2,
        width_ratios=[0.6] + [1] * 6 + [0.15],  # 第一列用于标签，稍微窄一点
        wspace=0.05,
        hspace=0.15,
        left=0.05,
        right=0.92,
        top=0.95,
        bottom=0.05,
    )

    # 4. 绘图循环
    # 时间标签定义
    labels_seq = ["T-50min", "T-40min", "T-30min", "T-20min", "T-10min", "T-0min"]
    labels_pred = [f"T+{i*10}min" for i in range(1, 7)]  # T+10min ...

    im_ref = None  # 用于 Colorbar

    for row_idx, name in enumerate(models_to_plot):
        # 获取数据 (处理 Sequence 键名不一致的问题)
        data_key = "sequence" if name == "Sequence" else name
        if data_key not in data:
            continue

        row_data = data[data_key]

        # --- A. 左侧标签 ---
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis("off")

        # 字体样式
        font_weight = "bold" if name in ["Sequence", "Observation"] else "normal"
        font_size = 15 if name in ["Sequence", "Observation"] else 14

        ax_label.text(
            0.5,
            0.5,
            name,
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight=font_weight,
            transform=ax_label.transAxes,
        )

        # --- B. 图像绘制 ---
        for col_idx in range(6):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])

            img = row_data[col_idx]
            im = ax.imshow(img, cmap=cmap, norm=norm, origin="upper")
            if im_ref is None:
                im_ref = im

            ax.set_xticks([])
            ax.set_yticks([])

            # --- 顶部时间标签 (仅对前两行特殊处理) ---
            if name == "Sequence":
                ax.set_title(labels_seq[col_idx], fontsize=13, fontweight="bold", pad=6)
            elif name == "Observation":
                ax.set_title(
                    labels_pred[col_idx], fontsize=13, fontweight="bold", pad=6
                )

            # --- 边框样式 ---
            # Sequence: 蓝色虚线
            # Observation: 红色虚线
            # 其他: 灰色实线
            if name == "Sequence":
                spine_color = "#1E90FF"  # DodgerBlue
                spine_style = "--"
                spine_width = 2.0
            elif name == "Observation":
                spine_color = "#DC143C"  # Crimson
                spine_style = "--"
                spine_width = 2.0
            else:
                spine_color = "gray"
                spine_style = "-"
                spine_width = 0.8
                # 如果是 StormWave (自家模型)，可以稍微加粗一点点实线强调
                if name == "StormWave":
                    spine_color = "black"
                    spine_width = 1.5

            for spine in ax.spines.values():
                spine.set_linestyle(spine_style)
                spine.set_linewidth(spine_width)
                spine.set_edgecolor(spine_color)

    # 5. Colorbar (右侧)
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])  # [left, bottom, width, height]

    # spacing='uniform' 这里的关键
    # 它会让非线性的物理区间在视觉上等距显示，方便对应颜色
    cbar = plt.colorbar(im_ref, cax=cax, spacing="uniform")

    # 设置刻度
    cbar.set_ticks(boundaries)

    # 格式化刻度标签 (去除太长的对应像素值，只保留物理含义)
    # 物理值: [0, 0.15, 0.32, 0.77, 1.5, 3.5, 7.0, 12.1, 32.2, 81.3]
    tick_labels = []
    for val in boundaries:
        if val < 0.1:
            tick_labels.append("0")
        elif val < 10:
            tick_labels.append(f"{val:.1f}")
        else:
            tick_labels.append(f"{int(round(val))}")

    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(r"VIL ($kg/m^2$)", fontsize=13, labelpad=12)

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches="tight", facecolor="white")
        print(f"图表已保存: {save_path}")


def main():
    case_name = "flashflood"
    case_path = f"case_study/{case_name}/"
    output_path = (
        f"out/Fig5_CaseStudy_{case_name}.jpg"
    )

    if not os.path.exists(case_path):
        print("错误: 找不到路径")
        return

    visualize_layout_v3(case_path, output_path)


if __name__ == "__main__":
    main()
