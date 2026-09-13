from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import numpy as np
import torch
from einops import rearrange
from torchvision.utils import save_image

COLORS = (
    "#E0FFFF",
    "#87CEEB",
    "#4682B4",
    "#4169E1",
    "#228B22",
    "#32CD32",
    "#9ACD32",
    "#FFA500",
    "#FF8C00",
    "#FF4500"
)


cmap = ListedColormap(colors=COLORS)
cmap.set_bad(color="white")

boundries = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 10000])
norm = BoundaryNorm(boundries, ncolors=len(COLORS), clip=True)


def apply_colormap(tensor: torch.Tensor):
    array = tensor.cpu().numpy()
    
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    color_array = mappable.to_rgba(array)
    
    color_array = color_array[:, :, :, :, :3]
    
    color_array = rearrange(color_array, "b t h w c -> (b t) c h w")
    
    
    return torch.from_numpy(color_array)


