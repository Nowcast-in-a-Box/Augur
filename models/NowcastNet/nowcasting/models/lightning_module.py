import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as L
from models.NowcastNet.nowcasting.models.nowcastnet import Net

class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        # prediction, target shape: (B, T, H, W, 1)
        B, T, H, W, C = prediction.shape
        # Flatten B and T, remove C
        pred = prediction.reshape(-1, H, W)
        tgt = target.reshape(-1, H, W)

        # Compute 2D FFT
        # Using rfftn for efficiency as inputs are real
        pred_fft = torch.fft.rfftn(pred, dim=(-2, -1))
        tgt_fft = torch.fft.rfftn(tgt, dim=(-2, -1))

        # Power spectral density or magnitude
        pred_mag = torch.abs(pred_fft)
        tgt_mag = torch.abs(tgt_fft)

        # Log-spectral loss (L1 distance in log-frequency domain)
        # log1p is used for numerical stability
        loss = torch.mean(torch.abs(torch.log1p(pred_mag) - torch.log1p(tgt_mag)))
        return loss

class NowcastNetLightning(L.LightningModule):
    def __init__(self, 
                 input_length: int = 9, 
                 total_length: int = 29, 
                 img_height: int = 512, 
                 img_width: int = 512, 
                 ngf: int = 32,
                 lr: float = 1e-4,
                 lambda_evo: float = 1.0,
                 lambda_spec: float = 0.01):
        """
        Initialize NowcastNet Lightning Module with official loss components.

        Args:
            input_length (int): Number of input radar frames.
            total_length (int): Total number of frames (input + prediction).
            img_height (int): Height of the input radar images.
            img_width (int): Width of the input radar images.
            ngf (int): Number of generative filters (base channels).
            lr (float): Learning rate.
            lambda_evo (float): Weight for evolution loss.
            lambda_spec (float): Weight for spectral loss.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Derived parameters
        evo_ic = total_length - input_length
        gen_oc = total_length - input_length
        ic_feature = ngf * 10
        
        self.model = Net(
            input_length=input_length,
            total_length=total_length,
            img_height=img_height,
            img_width=img_width,
            ngf=ngf,
            evo_ic=evo_ic,
            gen_oc=gen_oc,
            ic_feature=ic_feature
        )
        
        self.spec_criterion = SpectralLoss()

    def forward(self, x):
        """
        Forward pass for inference.
        Returns: gen_result, evo_result
        """
        return self.model(x)

    def _compute_weighted_loss(self, prediction, target, mask):
        # prediction, target shape: (B, T, H, W, 1)
        # mask shape: (B, T, H, W, 1)
        diff = (prediction - target) ** 2
        weighted_diff = diff * mask
        loss = weighted_diff.sum() / (mask.sum() + 1e-8)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch['radar_frames']
        # x shape: (B, total_length, H, W, 2)
        # Channel 0: intensity, Channel 1: mask
        
        targets = x[:, self.hparams.input_length:, :, :, 0:1]
        masks = x[:, self.hparams.input_length:, :, :, 1:2]
        
        # Forward pass returns both results
        gen_predictions, evo_predictions = self(x)
        
        # 1. Evolution Loss (MSE)
        loss_evo = self._compute_weighted_loss(evo_predictions, targets, masks)
        
        # 2. Generative Loss (MSE)
        loss_gen = self._compute_weighted_loss(gen_predictions, targets, masks)
        
        # 3. Spectral Loss
        loss_spec = self.spec_criterion(gen_predictions, targets)
        
        # Combined Unified Loss
        total_loss = self.hparams.lambda_evo * loss_evo + loss_gen + self.hparams.lambda_spec * loss_spec
        
        # Logging
        self.log("train_loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss_evo", loss_evo, on_epoch=True)
        self.log("loss_gen", loss_gen, on_epoch=True)
        self.log("loss_spec", loss_spec, on_epoch=True)
        
        return total_loss
        
    def validation_step(self, batch, batch_idx):
        x = batch['radar_frames']
        targets = x[:, self.hparams.input_length:, :, :, 0:1]
        masks = x[:, self.hparams.input_length:, :, :, 1:2]
        
        gen_predictions, evo_predictions = self(x)
        
        loss_evo = self._compute_weighted_loss(evo_predictions, targets, masks)
        loss_gen = self._compute_weighted_loss(gen_predictions, targets, masks)
        loss_spec = self.spec_criterion(gen_predictions, targets)
        
        val_loss = self.hparams.lambda_evo * loss_evo + loss_gen + self.hparams.lambda_spec * loss_spec
        
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_loss_evo", loss_evo, on_epoch=True)
        self.log("val_loss_gen", loss_gen, on_epoch=True)
        
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
