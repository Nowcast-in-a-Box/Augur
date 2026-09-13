from .models import PhyCell, ConvLSTM, EncoderRNN

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lightning.pytorch import LightningModule

from utils.metrics import log_loss

class PhyDNet(LightningModule):
    
    def __init__(self,
            input_channels = 6,
            input_height = 128,
            input_width = 128,
            phycell_input_dim = 64,  # 经过encoder_E后的通道数
            phycell_F_hidden_dim = [49, 49],  # 物理单元的隐藏维度（两层）
            phycell_n_layers = 2,
            phycell_kernel_size = (7, 7),
            convcell_hidden_dims = [64, 64],  # ConvLSTM隐藏维度（两层，都是64）
            convcell_n_layers = 2,
            convcell_kernel_size = (3, 3),
            learning_rate: float = 0.001
        ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.lr = learning_rate
        

        encoder_output_height = self.hparams.input_height // 4  # 128 / 4 = 32
        encoder_output_width = self.hparams.input_width // 4    # 128 / 4 = 32
        phycell_input_height = encoder_output_height  # 32
        phycell_input_width = encoder_output_width    # 32
        
        phycell = PhyCell(
            input_shape=(phycell_input_height, phycell_input_width),
            input_dim=self.hparams.phycell_input_dim,
            F_hidden_dims=self.hparams.phycell_F_hidden_dim,
            n_layers=self.hparams.phycell_n_layers,
            kernel_size=self.hparams.phycell_kernel_size,
        )

        # 2. 创建 ConvLSTM
        convcell = ConvLSTM(
            input_shape=(phycell_input_height, phycell_input_width),
            input_dim=self.hparams.phycell_input_dim,
            hidden_dims=self.hparams.convcell_hidden_dims,
            n_layers=self.hparams.convcell_n_layers,
            kernel_size=self.hparams.convcell_kernel_size,
        )
        
        self.convcell = convcell
        self.phycell = phycell

        # 3. 创建完整模型
        self.encoder = EncoderRNN(phycell, convcell, 
                                  in_channels=self.hparams.input_channels, out_channels=self.hparams.input_channels)



        self.criterion = nn.MSELoss()

        

    
    def forward(self, x: torch.Tensor):
        
        out_phys, hidden1, output_image, out_phys_detail, out_conv = self.encoder(
            x,
            first_timestep=True,
            decoding=False
        )
        
        
        
        return output_image
    
    def training_step(self, batch):
        
        
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred = self.forward(sequence)
        
        loss = self.criterion(pred, target)
        
        self.log("train_loss", loss)
        
        return loss

    
    def validation_step(self, batch):
        
        
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred = self.forward(sequence)
        
        loss = self.criterion(pred, target)
        
        self.log("val_loss", loss)
        
        log_loss(
            pred=pred,
            truth=target,
            stage="val",
            lightning_module=self
        )
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1,verbose=True)
        
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss"
            }
        }

    

