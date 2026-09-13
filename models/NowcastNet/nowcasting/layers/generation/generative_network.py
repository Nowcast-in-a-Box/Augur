import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.NowcastNet.nowcasting.layers.generation.module import GenBlock
from models.NowcastNet.nowcasting.layers.evolution.module import *

class Generative_Encoder(nn.Module):
    def __init__(self, n_channels, base_c=64):
        super(Generative_Encoder, self).__init__()
        base_c = base_c
        self.inc = DoubleConv(n_channels, base_c, kernel=3)
        self.down1 = Down(base_c * 1, base_c * 2, 3)
        self.down2 = Down(base_c * 2, base_c * 4, 3)
        self.down3 = Down(base_c * 4, base_c * 8, 3)

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        return x

class Generative_Decoder(nn.Module):
    def __init__(self, ngf: int = 32, gen_oc: int = 20, ic_feature: int = 320, evo_ic: int = 20):
        """
        Generative Decoder module.

        Args:
            ngf (int): Number of generative filters.
            gen_oc (int): Number of output channels for the generated image.
            ic_feature (int): Number of input channels for the initial convolution.
            evo_ic (int): Number of input channels for SPADE normalization (from evolution result).
        """
        super().__init__()
        self.ngf = ngf
        self.gen_oc = gen_oc
        self.ic_feature = ic_feature
        self.evo_ic = evo_ic

        nf = ngf

        self.fc = nn.Conv2d(ic_feature, 8 * nf, 3, padding=1)

        self.head_0 = GenBlock(8 * nf, 8 * nf, evo_ic=evo_ic)

        self.G_middle_0 = GenBlock(8 * nf, 4 * nf, evo_ic=evo_ic, double_conv=True)
        self.G_middle_1 = GenBlock(4 * nf, 4 * nf, evo_ic=evo_ic, double_conv=True)

        self.up_0 = GenBlock(4 * nf, 2 * nf, evo_ic=evo_ic)

        self.up_1 = GenBlock(2 * nf, 1 * nf, evo_ic=evo_ic, double_conv=True)
        self.up_2 = GenBlock(1 * nf, 1 * nf, evo_ic=evo_ic, double_conv=True)

        final_nc = nf * 1

        self.conv_img = nn.Conv2d(final_nc, gen_oc, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, x, evo):
        x = self.fc(x)
        x = self.head_0(x, evo)
        x = self.up(x)
        x = self.G_middle_0(x, evo)
        x = self.G_middle_1(x, evo)
        x = self.up(x)
        x = self.up_0(x, evo)
        x = self.up(x)
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        return x