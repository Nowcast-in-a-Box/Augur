import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.NowcastNet.nowcasting.layers.utils import warp, make_grid
from models.NowcastNet.nowcasting.layers.generation.generative_network import Generative_Encoder, Generative_Decoder
from models.NowcastNet.nowcasting.layers.evolution.evolution_network import Evolution_Network
from models.NowcastNet.nowcasting.layers.generation.noise_projector import Noise_Projector

class Net(nn.Module):
    def __init__(self, 
                 input_length: int = 9, 
                 total_length: int = 29, 
                 img_height: int = 512, 
                 img_width: int = 512, 
                 ngf: int = 32, 
                 evo_ic: int = 20, 
                 gen_oc: int = 20, 
                 ic_feature: int = 320):
        """
        NowcastNet model implementation.

        Args:
            input_length (int): Number of input radar frames.
            total_length (int): Total number of frames (input + prediction).
            img_height (int): Height of the input images.
            img_width (int): Width of the input images.
            ngf (int): Number of generative filters (base channels).
            evo_ic (int): Input channels for the evolution component (typically total_length - input_length).
            gen_oc (int): Output channels for the generative component (typically total_length - input_length).
            ic_feature (int): Input channels for the generative decoder (typically ngf * 10).
        """
        super(Net, self).__init__()
        self.input_length = input_length
        self.total_length = total_length
        self.img_height = img_height
        self.img_width = img_width
        self.ngf = ngf
        
        self.pred_length = self.total_length - self.input_length

        self.evo_net = Evolution_Network(self.input_length, self.pred_length, base_c=32)
        self.gen_enc = Generative_Encoder(self.total_length, base_c=self.ngf)
        self.gen_dec = Generative_Decoder(ngf=self.ngf, gen_oc=gen_oc, ic_feature=ic_feature, evo_ic=evo_ic)
        self.proj = Noise_Projector(self.ngf)

        sample_tensor = torch.zeros(1, 1, self.img_height, self.img_width)
        self.grid = make_grid(sample_tensor)

    def forward(self, all_frames):
        all_frames = all_frames[:, :, :, :, :1]

        frames = all_frames.permute(0, 1, 4, 2, 3)
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Input Frames
        input_frames = frames[:, :self.input_length]
        input_frames = input_frames.reshape(batch, self.input_length, height, width)

        # Evolution Network
        intensity, motion = self.evo_net(input_frames)
        motion_ = motion.reshape(batch, self.pred_length, 2, height, width)
        intensity_ = intensity.reshape(batch, self.pred_length, 1, height, width)
        series = []
        last_frames = all_frames[:, (self.input_length - 1):self.input_length, :, :, 0]
        grid = self.grid.repeat(batch, 1, 1, 1)
        for i in range(self.pred_length):
            last_frames = warp(last_frames, motion_[:, i], grid.to(input_frames.device), mode="nearest", padding_mode="border")
            last_frames = last_frames + intensity_[:, i]
            series.append(last_frames)
        evo_result = torch.cat(series, dim=1)

        evo_result = evo_result/128
        
        # Generative Network
        evo_feature = self.gen_enc(torch.cat([input_frames, evo_result], dim=1))

        noise = torch.randn(batch, self.ngf, height // 32, width // 32, device=input_frames.device)
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8).permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)

        feature = torch.cat([evo_feature, noise_feature], dim=1)
        gen_result = self.gen_dec(feature, evo_result)

        return gen_result.unsqueeze(-1), evo_result.unsqueeze(-1)