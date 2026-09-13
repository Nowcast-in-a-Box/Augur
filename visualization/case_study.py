from typing import Dict
import torch
from dataset.SEVIR import SEVIRLightningDataModule
from src.BrowianBridgeDiffusion import BrownianBridgeFramework
from utils.metrics import compute_loss, RMSE, CSI, CSIMean, HeidkeSkillScore, SSIM
from torch import no_grad, set_float32_matmul_precision, cat
from utils.gray_2_color import apply_colormap
from torchvision.utils import save_image

from utils.model_factory import ModelFactory
from utils.statistic import print_result

from torchmetrics import MeanMetric, MetricCollection
from rich.console import Console

from rich.progress import Progress

from torchinfo import summary

import time

set_float32_matmul_precision("high")
console = Console()

# =====================================

SHOW_SUMMARY = True

# =====================================



lead_times = (10, 20, 30 ,40 ,50 ,60)

CASE_NAME = "tornado"

DEVICE = "cuda:7"


def get_metrics_collections():
    return  {
        "rmse": RMSE(),
        "ssim": SSIM(),
        "hss": HeidkeSkillScore(),
        "csi_mean": CSIMean(),
        "csi_h": CSI(threshold=181),
        "csi_e": CSI(threshold=219),
    }



CHECKPOINTS = {
    "ConvLSTM": "checkpoints/ConvLSTM/last.ckpt",
    "MAU": "checkpoints/MAU/last.ckpt",
    "SimVP": "checkpoints/SimVP/last-v2.ckpt",
    "EarthFarseer": "checkpoints/EarthFarseer/last.ckpt",
    
    "AlphaPre": "checkpoints/AlphaPre/last.ckpt",
    "StormWave": "checkpoints/StormWave/last.ckpt"
    # "pySTEPS": ""
}



sequence = torch.load(f"case_study/{CASE_NAME}/sequence.pt", map_location=DEVICE).float()
target = torch.load(f"case_study/{CASE_NAME}/target.pt", map_location=DEVICE).float()
for k,v in CHECKPOINTS.items():
    

    DETERNIMISTIC_MODEL = k


    model = ModelFactory(
        DETERNIMISTIC_MODEL, 
        ckpt_path=CHECKPOINTS[DETERNIMISTIC_MODEL], 
        device=DEVICE if DETERNIMISTIC_MODEL != "pySTEPS" else ""
    )


    # FLOPS
    

    # Meterics:




    metric_collection = {
        f"lead_{h}": MetricCollection(get_metrics_collections()).to(DEVICE)
        for h in lead_times
    }

    start = time.time()
    pred = model(sequence, summary=False)
    end = time.time()
    print(f"{DETERNIMISTIC_MODEL}: {end - start} s")
    
    torch.save(pred, f"case_study/{CASE_NAME}/{DETERNIMISTIC_MODEL}.pt")

    for idx, lead in enumerate(lead_times):
        lead_pred = pred[:, idx: idx+1, ...]
        lead_target = target[:, idx: idx+1, ...]

        
        
        
        

        det_metrics = compute_loss(
            pred=lead_pred,
            truth=lead_target
        )

        collection = metric_collection[f"lead_{lead}"]
        
        for metric_name, metric_value in collection.items():

            metric_value.update(lead_pred * 255, lead_target * 255)

    table = print_result(model_name=DETERNIMISTIC_MODEL, metric_collection=metric_collection, metric_names=tuple(get_metrics_collections().keys()), lead_times=lead_times)

    # print(f"Model: {DETERNIMISTIC_MODEL}")
    console.print(table)
