from typing import Any
from lightning.pytorch import LightningModule
from models.UNet.new_UNet import Unet as VanillaUNet
from utils.metrics import log_loss
import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR

class UNet(LightningModule):

    def __init__(self, input_channel=6, size=384, lr=1e-4):
        super(UNet, self).__init__()
        self.input_channel = input_channel
        self.size = size
        self.lr = lr

        self.model = VanillaUNet(
            in_chan=input_channel,
            out_chan=input_channel
        )

    def forward(self, data: torch.Tensor):
        return self.model.forward(x=data)
    
    def training_step(self, batch):
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred = self.forward(data=sequence)
        
        log_loss(
            pred=pred,
            truth=target,
            stage="train",
            lightning_module=self
        )
        
        loss = F.mse_loss(pred, target, reduce="mean")
        self.log("train_loss", loss, sync_dist=True)
        return loss
        
        

    def validation_step(self, batch):
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred = self.forward(data=sequence)
        
        log_loss(
            pred=pred,
            truth=target,
            stage="val",
            lightning_module=self
        )
        
        loss = F.mse_loss(pred, target, reduce="mean")
        self.log("val_loss", loss, sync_dist=True)
        
        return loss

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
