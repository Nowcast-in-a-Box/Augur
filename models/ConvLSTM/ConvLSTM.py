import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.nn import init
from lightning.pytorch import LightningModule

from utils.metrics import log_loss

from models.ConvLSTM.ConvLSTMCell import ConvLSTMCell


class EncoderDecoderConvLSTM(LightningModule):
    def __init__(self, nf: int, in_chan: int, T: int):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.save_hyperparameters()
        
        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        self.T = T
        self.encoder_1_convlstm = ConvLSTMCell(
            input_dim=in_chan, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.encoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_1_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True  # nf + 1
        )

        self.decoder_2_convlstm = ConvLSTMCell(
            input_dim=nf, hidden_dim=nf, kernel_size=(3, 3), bias=True
        )

        self.decoder_CNN = nn.Conv3d(
            in_channels=nf, out_channels=1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

    def autoencoder(
        self, x, seq_len, future_step, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
    ):

        outputs = []

        # encoder
        for t in range(seq_len):
            h_t, c_t = self.encoder_1_convlstm(
                input_tensor=x[:, t, :, :], cur_state=[h_t, c_t]
            )  # we could concat to provide skip conn here
            h_t2, c_t2 = self.encoder_2_convlstm(
                input_tensor=h_t, cur_state=[h_t2, c_t2]
            )  # we could concat to provide skip conn here

        # encoder_vector
        encoder_vector = h_t2

        # decoder
        for t in range(future_step):
            h_t3, c_t3 = self.decoder_1_convlstm(
                input_tensor=encoder_vector, cur_state=[h_t3, c_t3]
            )  # we could concat to provide skip conn here
            h_t4, c_t4 = self.decoder_2_convlstm(
                input_tensor=h_t3, cur_state=[h_t4, c_t4]
            )  # we could concat to provide skip conn here
            encoder_vector = h_t4
            outputs += [h_t4]  # predictions

        outputs = torch.stack(outputs, 1)
        outputs = outputs.permute(0, 2, 1, 3, 4)
        outputs = self.decoder_CNN(outputs)
        outputs = torch.nn.Sigmoid()(outputs)

        return outputs

    def forward(self, x):
        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # find size of different input dimensions
        b, seq_len, _, h, w = x.size()

        # initialize hidden states
        h_t, c_t = self.encoder_1_convlstm.init_hidden(batch_size=b, image_size=(h, w))
        h_t2, c_t2 = self.encoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )
        h_t3, c_t3 = self.decoder_1_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )
        h_t4, c_t4 = self.decoder_2_convlstm.init_hidden(
            batch_size=b, image_size=(h, w)
        )

        # autoencoder forward
        outputs = self.autoencoder(
            x, seq_len, self.T, h_t, c_t, h_t2, c_t2, h_t3, c_t3, h_t4, c_t4
        )

        return outputs

    def training_step(self, batch):
        sequence = batch["sequence"]
        target = batch["target"]
        
        pred = self.forward(sequence.unsqueeze(2)).squeeze(1)

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
        
        pred = self.forward(sequence.unsqueeze(2)).squeeze(1)



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
        
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
        )
        
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer,
        #     T_max=50,
        #     eta_min=1e-6
        # )
        
        # return [optimizer], [
        #     {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #     }
        # ]
        return optimizer