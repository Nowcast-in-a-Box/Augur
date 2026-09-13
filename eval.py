import torch
import copy
from typing import Dict, List, Tuple
from torch import set_float32_matmul_precision
from lightning.pytorch import seed_everything
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from torchmetrics import MetricCollection

# Assuming local imports follow the user's project structure
from dataset.DataReader import HDF5DataModule
from utils.model_factory import ModelFactory
from utils.statistic import print_result
from utils.metrics import (
    SSIM,
    HeidkeSkillScore,
    THRESHOLDS,
    CSI,
    CSIMean,
    RMSE,
    POD,
    FAR,
    FSS,
)

# Configuration
set_float32_matmul_precision("high")
console = Console()
seed_everything(42)

# Global Constants
DEVICE = torch.device("cuda:3")
DTYPE = torch.float32
LEAD_TIMES = (10, 20, 30, 40, 50, 60)


# Define Metrics
METRICS_PROTOTYPE = [
    RMSE(),
    SSIM(),
    HeidkeSkillScore(),
    CSIMean(),
    CSI(threshold=181),
    CSI(threshold=219),
    POD(threshold=181),
    POD(threshold=219),
    FAR(threshold=181),
    FAR(threshold=219),
    FSS(threshold=181, scale=3),
    FSS(threshold=219, scale=3),
]

# Create a mapping of metric names for sorting/display
METRIC_NAMES = [str(m) for m in METRICS_PROTOTYPE]

CHECKPOINTS = {
    "UNet": "checkpoints/UNet/last-v2.ckpt",
    "MAU": "checkpoints/MAU/last.ckpt",
    "ConvLSTM": "checkpoints/ConvLSTM/last.ckpt",
    "EarthFarseer": "checkpoints/EarthFarseer/last.ckpt",
    "SimVP": "checkpoints/SimVP/last-v2.ckpt",
    "AlphaPre": "checkpoints/AlphaPre/last.ckpt",
    "WADEPre": "/checkpoints/WADEPre/WADEPre-1.ckpt",
    "exPreCast": "checkpoints/exPreCast/exPreCast-v1.ckpt",
    "PhyDNet": "checkpoints/PhyDNet/last-v2.ckpt",
    "NowcastNet": "checkpoints/NowcastNetLightning/last.ckpt",
}


def setup_metrics(
    lead_times: Tuple[int, ...], prototype: List
) -> Dict[str, MetricCollection]:
    """Creates a dictionary of MetricCollections, one for each lead time."""
    metric_collection = {}
    for lt in lead_times:
        copied_metrics = {str(m): copy.deepcopy(m) for m in prototype}
        metric_collection[f"lead_{lt}"] = MetricCollection(copied_metrics).to(DEVICE)
    return metric_collection


def main():
    # import torch.backends.cudnn as cudnn

    # print(torch.get_float32_matmul_precision(), cudnn.allow_tf32, cudnn.deterministic, cudnn.benchmark)

    # # Setup Dataset
    dataset = HDF5DataModule("/data2/StormWave/SEVIR", batch_size=25, num_workers=128)
    dataset.setup("fit")
    val_loader = dataset.val_dataloader()
    total_batches = len(val_loader)

    for model_name, ckpt_path in CHECKPOINTS.items():
        console.rule(f"[bold yellow]Evaluating Model: {model_name}")

        # Initialize Model using Factory
        model = ModelFactory(model_name=model_name, ckpt_path=ckpt_path, device=DEVICE)

        # Initialize Metrics for this model
        metrics_per_lead = setup_metrics(LEAD_TIMES, METRICS_PROTOTYPE)

        # Inference Loop with Progress Bar
        with torch.inference_mode():
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"[cyan]Inference {model_name}...", total=total_batches
                )

                for batch in val_loader:
                    sequence = (
                        batch["sequence"]
                        .to(device=DEVICE, dtype=DTYPE, non_blocking=True)
                        .contiguous()
                    )
                    target = batch["target"].to(
                        device=DEVICE, dtype=DTYPE, non_blocking=True
                    )

                    # Model forward pass
                    pred = model(sequence.contiguous())

                    for idx, lt in enumerate(LEAD_TIMES):
                        lead_pred = pred[:, idx : idx + 1, ...]
                        lead_target = target[:, idx : idx + 1, ...]
                        metrics_per_lead[f"lead_{lt}"].update(
                            lead_pred * 255, lead_target * 255
                        )

                    progress.advance(task)

        # Compute and Print Results
        table = print_result(
            model_name=model_name,
            metric_collection=metrics_per_lead,
            lead_times=LEAD_TIMES,
            metric_order=METRIC_NAMES,
        )

        console.print(table)

        # Print Model Statistics (Params, FLOPs)
        model.print_summary(console)
        console.print("\n")

        # Cleanup for next model
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
