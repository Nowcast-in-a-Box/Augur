from lightning.pytorch import LightningModule

import torch
import torch.nn.functional as F

from models.EarthFarseer.model import Earthfarseer_model

class EarthFarseer(LightningModule):
    
    def __init__(self, shape_in, shape_out):
        super().__init__()
        
        self.model = Earthfarseer_model(
            shape_in=shape_in,
            shape_out=shape_out
        )
        
    
    def forward(self, data: torch.Tensor):
        return self.model.forward(data)
    

