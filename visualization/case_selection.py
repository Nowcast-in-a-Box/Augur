from dataset.DataReader import HDF5DataModule
import lightning.pytorch as pl
import torch

pl.seed_everything(42, workers=True)


dataset = HDF5DataModule("/data2/StormWave/SEVIR", batch_size=1, num_workers=10)

dataset.setup("fit")

dataloader = dataset.val_dataset



batch = dataloader[59]


sequence = batch["sequence"]
torch.save(sequence.unsqueeze(0), "case_study/sequence.pt")
target = batch["target"]
torch.save(target.unsqueeze(0), "case_study/target.pt")



for index in range(len(dataloader)):
    batch = dataloader[index]
    
    target:torch.Tensor = batch["target"]
    
    max_value = (target > 219 / 255).any()
    
    ratio_sum = (target >= 100 / 255).float().sum() / target.numel()
    
    max_delta = 0.0
    
    for i in range(5):
        delta = target[i+1, :, :] - target[i, :, :]
        
        max_num = delta.max().item()
        


        
        if max_num >= max_delta:
            max_delta = max_num
    
    if max_value and ratio_sum >= 0.3 and max_delta >= 180/255:
        print(f"{index}, ratio: {ratio_sum}, delta: {max_delta}")

# lucky number : 2605