# EarthFarseer package
"""
EarthFarseer: A deep learning model for spatiotemporal forecasting.
"""

__version__ = "1.0.0"
__author__ = "EarthFarseer Team"

# Import main modules to make them available at package level
from .model import *
from .modules import *
from .FoTF_module import *
from .Temporal_block import *
from .utils import *
from .Fourier_computing_unit import *
from .Global_Fourier_Transformer import *
from .Local_CNN_Branch import *
