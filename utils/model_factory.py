from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import torch
from torchinfo import summary

from utils.model_params import (
    ALPHAPRE_KWARGS,
    CONVLSTM_KWARGS,
    EARTHFARSEER_KWARGS,
    EXPRECAST_KWARGS,
    MAU_KWARGS,
    NOWCASTNET_KWARGS,
    PHYDNET_KWARGS,
    SIMVP_KWARGS,
    UNET_KWARGS,
    WADEPRE_KWARGS,
)

# Importing models
try:
    from models.UNet.UNet import UNet
    from models.MAU.mau_lightning import MAULightning
    from models.EarthFarseer.model import Earthfarseer_model
    from models.AlphaPre.AlphaPre import AlphaPre
    from models.SimVP.simvpp import SimVP
    from models.ConvLSTM.ConvLSTM import EncoderDecoderConvLSTM
    from models.WADEPre.WADEPre import WADEPre
    from models.exPreCast.exPreCast import exPreCast
    from models.PhyDNet.PhyDNet import PhyDNet
    from models.NowcastNet.nowcasting.models.lightning_module import NowcastNetLightning
except ImportError as e:
    print(f"Model class importing error: {e}")

class ModelSummaryWrapper(torch.nn.Module):
    """Wrapper to make strategy.model_call compatible with torchinfo.summary."""
    def __init__(self, model: torch.nn.Module, strategy: 'ModelStrategy'):
        super().__init__()
        self.model = model
        self.strategy = strategy

    def forward(self, x: torch.Tensor) -> Any:
        return self.strategy.model_call(self.model, x)

def summary_print(model: torch.nn.Module, strategy: 'ModelStrategy', input_data: Any) -> str:
    """Calculates model summary and returns only total params and mult-adds."""
    wrapper = ModelSummaryWrapper(model, strategy)
    results = summary(
        model=wrapper,
        input_data=input_data,
        col_names=("num_params", "mult_adds"),
        depth=0,
        verbose=0
    )
    return (f"Total Parameters: {results.total_params:,}\n"
            f"Total Mult-Adds: {results.total_mult_adds:,} (MACs)")

class ModelStrategy(ABC):
    """Base class for all model inference strategies."""
    
    @abstractmethod
    def load_checkpoint(self, ckpt_path: str) -> torch.nn.Module:
        """Loads the model from a checkpoint."""
        pass
    
    def preprocess(self, data: torch.Tensor) -> Any:
        """Transforms raw input data (B, T, H, W) into model-ready input."""
        return data

    def model_call(self, model: torch.nn.Module, x: Any) -> Any:
        """Executes the core model forward pass/logic."""
        return model(x)

    def postprocess(self, output: Any) -> torch.Tensor:
        """Transforms model output back to the standard (B, T, H, W) format."""
        return output

    def inference(self, model: torch.nn.Module, data: torch.Tensor) -> torch.Tensor:
        """Standardized inference pipeline."""

        x = self.preprocess(data)
        out = self.model_call(model, x)
        return self.postprocess(out)

# Strategy Registry
STRATEGY_REGISTRY: Dict[str, Type[ModelStrategy]] = {}

def register_strategy(name: str):
    """Decorator to register a strategy class."""
    def decorator(cls: Type[ModelStrategy]):
        STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


@register_strategy("UNet")
class UNetStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return UNet.load_from_checkpoint(
            ckpt_path,
            **UNET_KWARGS,
        )

    def model_call(self, model: UNet, x):
        return model.forward(x)
    
@register_strategy("NowcastNet")
class NowcastNetStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return NowcastNetLightning.load_from_checkpoint(
            ckpt_path,
            **NOWCASTNET_KWARGS,
        )
    
    def model_call(self, model: NowcastNetLightning, x):
        return model.forward(x)

@register_strategy("MAU")
class MAUStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return MAULightning.load_from_checkpoint(
            ckpt_path,
            **MAU_KWARGS,
        )
    
    def model_call(self, model, x):
        # MAU uses a shared_step for inference in its lightning module
        return model._shared_step(
            batch={"sequence": x, "target": x.detach().clone()},
            batch_idx=1, prefix="val"
        )["predictions"]

    def postprocess(self, output):
        return output.squeeze(2)

@register_strategy("AlphaPre")
class AlphaPreStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return AlphaPre.load_from_checkpoint(
            ckpt_path,
            **ALPHAPRE_KWARGS,
        )
    
    def preprocess(self, data):
        return data.unsqueeze(2)

    def model_call(self, model, x):
        pred, _ = model.forward(x, truth=None, compute_loss=False)
        return pred

    def postprocess(self, output):
        return output.squeeze(2)

@register_strategy("EarthFarseer")
class EarthFarseerStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return Earthfarseer_model.load_from_checkpoint(
            ckpt_path,
            **EARTHFARSEER_KWARGS,
        )
    
    def preprocess(self, data):
        return data.unsqueeze(2)

    def postprocess(self, output):
        return output.squeeze(2)

@register_strategy("ConvLSTM")
class ConvLSTMStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return EncoderDecoderConvLSTM.load_from_checkpoint(
            ckpt_path,
            **CONVLSTM_KWARGS,
        )
    
    def preprocess(self, data):
        return data.unsqueeze(2)

    def postprocess(self, output):
        return output.squeeze(1)

@register_strategy("SimVP")
class SimVPStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        return SimVP.load_from_checkpoint(
            ckpt_path,
            **SIMVP_KWARGS,
        )

    def preprocess(self, data):
        return data.unsqueeze(2)

    def postprocess(self, output):
        return output.squeeze(2)

@register_strategy("WADEPre")
class WADEPreStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)["state_dict"]
        model = WADEPre(**WADEPRE_KWARGS)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def model_call(self, model: WADEPre, x):
        pred, _ = model.forward(x)
        return pred

@register_strategy("exPreCast")
class exPreCastStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        model = exPreCast(**EXPRECAST_KWARGS)
        model.load_pretrained(pretrained_path=ckpt_path)
        return model

    def preprocess(self, data):
        return data.unsqueeze(1)

    def postprocess(self, output):
        return output.squeeze(1)

@register_strategy("PhyDNet")
class PhyDNetStrategy(ModelStrategy):
    def load_checkpoint(self, ckpt_path):
        model = PhyDNet(**PHYDNET_KWARGS)
        model.setup("")
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
        return model


class ModelFactory:
    """Factory class to create and manage models based on their strategies."""
    
    def __init__(self, model_name: str, ckpt_path: str, device):
        if model_name not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(f"Unknown model: {model_name}. Available: {available}")
        
        self.model_name = model_name
        self.strategy = STRATEGY_REGISTRY[model_name]()
        self.model = self.strategy.load_checkpoint(ckpt_path)
        self.device = device
        self._summary_result = None
        
        if device:
            self.model.to(device)
            
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self._summary_result is None:
            try:
                sample = data[:1].detach().clone()
                input_data = self.strategy.preprocess(sample)
                
                self._summary_result = summary_print(self.model, self.strategy, input_data)
                self.model.eval()
            except Exception as e:
                self._summary_result = f"Summary calculation failed: {e}"
            
        return self.strategy.inference(model=self.model, data=data)

    def print_summary(self, console: Any):
        """Prints total params and mult-adds."""
        if self._summary_result:
            console.print(f"\n[bold cyan]Model Statistics: {self.model_name}[/bold cyan]")
            console.print(self._summary_result)
        else:
            console.print(f"[yellow]No summary info available for {self.model_name}[/yellow]")
