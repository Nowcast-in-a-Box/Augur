import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import lightning as L
from networks.afnonet import AFNONet
import numpy as np

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, img_size=(160, 320), in_chans=2, out_chans=2, length=10):
        self.img_size = img_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.randn(self.in_chans, *self.img_size)
        y = torch.randn(self.out_chans, *self.img_size)
        return x, y

def test_inference():
    print("Testing inference...")
    # Use smaller size for quick testing on CPU
    img_size = (160, 320)
    patch_size = (16, 16)
    in_chans = 2
    out_chans = 2
    
    model = AFNONet(
        img_size=img_size, 
        patch_size=patch_size, 
        in_chans=in_chans, 
        out_chans=out_chans,
        embed_dim=128, # Smaller for speed
        depth=2,       # Smaller for speed
        num_blocks=4   # Smaller for speed
    )
    
    sample_input = torch.randn(1, in_chans, *img_size)
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    assert output.shape == (1, out_chans, *img_size)
    print("Inference test passed!")

def test_training():
    print("\nTesting training demo...")
    img_size = (64, 128)
    patch_size = (8, 8)
    in_chans = 2
    out_chans = 2
    
    model = AFNONet(
        img_size=img_size, 
        patch_size=patch_size, 
        in_chans=in_chans, 
        out_chans=out_chans,
        embed_dim=64,
        depth=2,
        num_blocks=4
    )
    
    dataset = DummyDataset(img_size=img_size, in_chans=in_chans, out_chans=out_chans, length=20)
    train_loader = DataLoader(dataset, batch_size=4)
    
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
    )
    
    # Check gradients before training
    initial_weights = model.blocks[0].filter.w1.clone()
    
    trainer.fit(model, train_loader)
    
    # Check if weights changed (meaning gradients were computed and optimizer stepped)
    final_weights = model.blocks[0].filter.w1
    weight_diff = torch.norm(final_weights - initial_weights)
    print(f"Weight difference after training: {weight_diff.item()}")
    
    assert weight_diff > 0, "Weights did not change, gradients might be zero or optimizer not stepping."
    print("Training demo passed!")

if __name__ == "__main__":
    try:
        test_inference()
        test_training()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
