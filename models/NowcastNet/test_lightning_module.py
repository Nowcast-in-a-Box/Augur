import torch
from torch.utils.data import DataLoader, Dataset
import lightning.pytorch as L
import argparse
from models.NowcastNet.nowcasting.models.lightning_module import NowcastNetLightning
import traceback

class MockDataset(Dataset):
    def __init__(self, length, img_height, img_width, num_samples=5):
        self.length = length
        self.img_height = img_height
        self.img_width = img_width
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # radar_frames has shape: (total_length, img_height, img_width, 2)
        # Channel 0: intensity, Channel 1: mask
        frames = torch.rand((self.length, self.img_height, self.img_width, 2), dtype=torch.float32)
        return {'radar_frames': frames}

def get_mock_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_length', type=int, default=9)
    parser.add_argument('--total_length', type=int, default=29)
    parser.add_argument('--img_height', type=int, default=128) # Smaller for fast CPU test
    parser.add_argument('--img_width', type=int, default=128)
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args([])
    
    args.evo_ic = args.total_length - args.input_length
    args.gen_oc = args.total_length - args.input_length
    args.ic_feature = args.ngf * 10
    return args

def main():
    print("Initializing mock configuration...")
    configs = get_mock_args()
    
    print("Initializing MockDataset...")
    train_dataset = MockDataset(configs.total_length, configs.img_height, configs.img_width, num_samples=4)
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    
    val_dataset = MockDataset(configs.total_length, configs.img_height, configs.img_width, num_samples=2)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size)
    
    print("Initializing NowcastNetLightning...")
    model = NowcastNetLightning(
        input_length=configs.input_length,
        total_length=configs.total_length,
        img_height=configs.img_height,
        img_width=configs.img_width,
        ngf=configs.ngf
    )
    
    print("\n--- Manual Gradient Check ---")
    # Fetch a single batch manually
    batch = next(iter(train_loader))
    
    # Forward pass
    try:
        model.train()
        # Test forward directly first to see tuple
        gen_out, evo_out = model(batch['radar_frames'])
        print(f"Model forward successful. Gen shape: {gen_out.shape}, Evo shape: {evo_out.shape}")
        
        loss = model.training_step(batch, 0)
        print(f"Training step successful. Initial Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        print("Backward pass successful.")
        
        # Check gradients
        has_grad = False
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                grad_norm += param.grad.data.norm(2).item() ** 2
        
        if has_grad:
            grad_norm = grad_norm ** 0.5
            print(f"Gradients computed successfully. Total gradient norm: {grad_norm:.4f}")
        else:
            print("WARNING: No gradients found after backward pass!")
            
    except Exception as e:
        print("Error during manual forward/backward check:")
        traceback.print_exc()
        return

    print("\n--- Lightning Trainer Test ---")
    # Using fast_dev_run=True runs 1 batch of training and validation to catch any bugs
    try:
        trainer = L.Trainer(fast_dev_run=True, accelerator='cpu')
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("Lightning Trainer test completed successfully.")
    except Exception as e:
        print("Error during Lightning Trainer run:")
        traceback.print_exc()

if __name__ == '__main__':
    main()
