import torch
import pytorch_lightning as pl
from models.DGMR.dgmr import DGMR

def test_dgmr():
    # 1. 初始化模型
    # 我们通过新添加的 num_context_steps 参数将其设为 6
    model = DGMR(
        forecast_steps=6,
        input_channels=1,
        output_shape=128,
        gen_lr=5e-5,
        disc_lr=2e-4,
        num_samples=2,
        generation_steps=1, 
        num_context_steps=6, # 匹配输入帧数
    )

    # 2. 准备模拟数据 (B, T, C, H, W)
    batch_size = 2
    input_frames = 6
    forecast_frames = 6
    channels = 1
    height = 128
    width = 128

    images = torch.randn(batch_size, input_frames, channels, height, width)
    future_images = torch.randn(batch_size, forecast_frames, channels, height, width)

    print(f"Input images shape: {images.shape}")
    print(f"Future images shape: {future_images.shape}")

    # 3. Inference Test (Forward Pass)
    print("\n--- Running Inference ---")
    model.eval()
    with torch.no_grad():
        output = model(images)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, forecast_frames, channels, height, width), f"Expected {(batch_size, forecast_frames, channels, height, width)}, got {output.shape}"
    print("Inference successful!")

    # 4. Training Step Test
    print("\n--- Running Training Step ---")
    model.train()
    
    # We need to mock the optimizers because DGMR uses manual optimization
    # and training_step calls self.optimizers()
    
    # Use a dummy trainer to setup the model for manual optimization
    trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, enable_checkpointing=False, fast_dev_run=True)
    
    # We can use a simple DataLoader with 1 batch
    from torch.utils.data import DataLoader, Dataset
    class DummyDataset(Dataset):
        def __init__(self, images, future_images):
            self.images = images
            self.future_images = future_images
        def __len__(self): return 1
        def __getitem__(self, idx): return self.images[idx], self.future_images[idx]

    dataset = DummyDataset(images, future_images)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Run one step
    trainer.fit(model, train_dataloaders=dataloader)
    
    # Check if gradients are populated
    has_grads = False
    for name, param in model.generator.named_parameters():
        if param.grad is not None:
            has_grads = True
            print(f"Gradient found for {name}, mean: {param.grad.mean().item():.6f}")
            break
    
    if has_grads:
        print("Training step successful! Gradients populated.")
    else:
        print("Error: No gradients found after training step.")

if __name__ == "__main__":
    test_dgmr()
