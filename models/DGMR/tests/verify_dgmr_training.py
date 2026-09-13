import torch
import pytorch_lightning as pl
from models.DGMR.dgmr import DGMR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# 1. 定义模拟数据集，模拟雷达帧数据 (Batch, Time, Channel, Height, Width)
class MockRadarDataset(Dataset):
    def __init__(self, num_samples=4, input_frames=4, target_frames=4, size=64):
        self.num_samples = num_samples
        self.input_frames = input_frames
        self.target_frames = target_frames
        self.size = size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回输入帧和目标帧
        x = torch.randn(self.input_frames, 1, self.size, self.size)
        y = torch.randn(self.target_frames, 1, self.size, self.size)
        return x, y

def verify_dgmr():
    print("正在初始化 DGMR 模型进行 CPU 测试...")
    
    # 使用较小的参数以便在 CPU 上快速测试
    config = {
        "forecast_steps": 4,
        "input_channels": 1,
        "output_shape": 64,
        "latent_channels": 128,
        "context_channels": 64,
        "num_samples": 2,
        "generation_steps": 2, # 减少生成步骤以节省 CPU 内存
        "gen_lr": 1e-4,
        "disc_lr": 4e-4
    }

    # 初始化模型
    # 注意：虽然类定义中继承了 PyTorchModelHubMixin，但我们直接使用标准构造函数，不调用 Hub 相关方法
    model = DGMR(**config)
    
    # 强制设为 CPU 模式
    model.cpu()

    # 2. 测试正向传播 (Inference)
    print("\n--- 步骤 1: 测试正向传播 ---")
    dummy_input = torch.randn(2, 4, 1, 64, 64) # (Batch=2, Time=4, C=1, H=64, W=64)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    
    assert output.shape == (2, 4, 1, 64, 64), f"输出形状错误: {output.shape}"
    assert not torch.isnan(output).any(), "输出中包含 NaN"
    print("正向传播测试成功！")

    # 3. 测试训练步骤 (Manual Optimization)
    print("\n--- 步骤 2: 测试模拟训练 (检查梯度) ---")
    model.train()
    dataset = MockRadarDataset(num_samples=4, size=64)
    dataloader = DataLoader(dataset, batch_size=2)

    # 使用 Lightning Trainer 进行单步训练测试
    # 这会触发 DGMR 类中定义的 training_step (包含手动 backward 和 optimizer.step)
    trainer = pl.Trainer(
        max_steps=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    trainer.fit(model, dataloader)

    # 检查生成器梯度
    gen_grad_found = False
    for name, param in model.generator.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            if norm > 0:
                gen_grad_found = True
                print(f"生成器梯度正常: {name[:30]}... norm = {norm:.6f}")
                break
    
    # 检查判别器梯度
    disc_grad_found = False
    for name, param in model.discriminator.named_parameters():
        if param.grad is not None:
            norm = param.grad.norm().item()
            if norm > 0:
                disc_grad_found = True
                print(f"判别器梯度正常: {name[:30]}... norm = {norm:.6f}")
                break

    assert gen_grad_found, "生成器未检测到有效梯度"
    assert disc_grad_found, "判别器未检测到有效梯度"
    print("训练步骤与梯度检查成功！")

    # 4. 测试标准权重加载 (排除 Mixin 干扰)
    print("\n--- 步骤 3: 测试标准 PyTorch 权重保存与加载 ---")
    save_path = "dgmr_debug.ckpt"
    # 使用标准的 torch.save 而不是 mixin 的 save_pretrained
    torch.save({"state_dict": model.state_dict(), "config": config}, save_path)
    
    # 创建新实例并加载
    checkpoint = torch.load(save_path, map_location="cpu")
    new_model = DGMR(**checkpoint["config"])
    new_model.load_state_dict(checkpoint["state_dict"])
    
    print("标准 load_state_dict 成功。PyTorchModelHubMixin 未介入此过程。")
    print("\n所有测试通过！")

if __name__ == "__main__":
    verify_dgmr()
