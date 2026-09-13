from abc import ABC, abstractmethod
from typing import Any, Type

from utils.model_params import (
    ALPHAPRE_KWARGS,
    CONVLSTM_KWARGS,
    DGMR_KWARGS,
    EARTHFARSEER_KWARGS,
    EXPRECAST_KWARGS,
    FOURCASTNET_KWARGS,
    MAU_KWARGS,
    NOWCASTNET_KWARGS,
    PHYDNET_KWARGS,
    SIMVP_KWARGS,
    UNET_KWARGS,
)


class ModelConfig(ABC):
    precision = "bf16-mixed"

    @abstractmethod
    def build_model(self) -> Any:
        pass


MODEL_REGISTRY: dict[str, Type[ModelConfig]] = {}


def register_model(name: str):
    def decorator(cls: Type[ModelConfig]) -> Type[ModelConfig]:
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


@register_model("UNet")
class UNetConfig(ModelConfig):
    def build_model(self):
        from models.UNet.UNet import UNet

        return UNet(**UNET_KWARGS)


@register_model("DGMR")
class DGMRConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.DGMR.dgmr.dgmr import DGMR

        return DGMR(**DGMR_KWARGS)


@register_model("NowcastNet")
class NowcastNetConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.NowcastNet.nowcasting.models.lightning_module import NowcastNetLightning

        return NowcastNetLightning(**NOWCASTNET_KWARGS)


@register_model("FourCastNet")
class FourCastNetConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.FourCastNet.networks.afnonet import AFNONet

        return AFNONet(**FOURCASTNET_KWARGS)


@register_model("exPreCast")
class exPreCastConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.exPreCast.exPreCast import exPreCast

        return exPreCast(**EXPRECAST_KWARGS)


@register_model("PhyDNet")
class PhyDNetConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.PhyDNet.PhyDNet import PhyDNet

        return PhyDNet(**PHYDNET_KWARGS)


@register_model("MAU")
class MAUConfig(ModelConfig):
    def build_model(self):
        from models.MAU.mau_lightning import MAULightning

        return MAULightning(**MAU_KWARGS)


@register_model("SimVP")
class SimVPConfig(ModelConfig):
    def build_model(self):
        from models.SimVP.simvpp import SimVP

        return SimVP(**SIMVP_KWARGS)


@register_model("ConvLSTM")
class ConvLSTMConfig(ModelConfig):
    def build_model(self):
        from models.ConvLSTM.ConvLSTM import EncoderDecoderConvLSTM

        return EncoderDecoderConvLSTM(**CONVLSTM_KWARGS)


@register_model("EarthFarseer")
class EarthFarseerConfig(ModelConfig):
    def build_model(self):
        from models.EarthFarseer.model import Earthfarseer_model

        return Earthfarseer_model(**EARTHFARSEER_KWARGS)


@register_model("AlphaPre")
class AlphaPreConfig(ModelConfig):
    precision = "32"

    def build_model(self):
        from models.AlphaPre.AlphaPre import AlphaPre

        return AlphaPre(**ALPHAPRE_KWARGS)
