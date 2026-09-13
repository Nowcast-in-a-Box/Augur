from models.AlphaPre.models.alphapre import get_model


from typing import Any
from lightning.pytorch import LightningModule
from models.UNet.vanilla_UNet import UNet as VanillaUNet
from utils.metrics import log_loss
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR

class AlphaPre(LightningModule):
    def __init__(self, 
                n_layers: int = 3,
                spec_num: int = 20,
                pha_weight: float = 0.01,
                anet_weight: float = 0.1,
                amp_weight: float = 0.01,
                aweight_stop_steps: int = 10000,
                lr: float = 1e-4,
                img_size: int = 128,
                T_in: int = 6,
                T_out: int = 6
        ):
        super().__init__()
        self.lr = lr
        
        self.model = get_model(
            img_channels=1,
            T_in=T_in,
            T_out=T_out,
            input_shape=(img_size, img_size),
            n_layers=n_layers,
            spec_num=spec_num,
            pha_weight=pha_weight,
            anet_weight=anet_weight,
            amp_weight=amp_weight,
            aweight_stop_steps=aweight_stop_steps,
            
        )
    
    
    def forward(self, data: torch.Tensor, 
                truth: torch.Tensor | None = None, 
                compute_loss: bool = False):
        return self.model.predict(frames_in=data, 
                                  frames_gt=truth,
                                  compute_loss=compute_loss)
    
    def training_step(self, batch):
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred, loss = self.forward(data=sequence.unsqueeze(2), truth=target.unsqueeze(2), compute_loss=True)
        pred = pred.squeeze(2)
        
        log_loss(
            pred=pred,
            truth=target,
            stage="train",
            lightning_module=self
        )
        total_loss = loss["total_loss"]
        self.log("train_loss", total_loss,  sync_dist=True)
        return total_loss
        
        

    def validation_step(self, batch):
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred, loss = self.forward(data=sequence.unsqueeze(2), truth=target.unsqueeze(2), compute_loss=True)
        pred = pred.squeeze(2)
        
        log_loss(
            pred=pred,
            truth=target,
            stage="val",
            lightning_module=self
        )
        
        total_loss = loss["total_loss"]
        self.log("val_loss", total_loss,  sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
        )

        scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=300,
        )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        ]

    