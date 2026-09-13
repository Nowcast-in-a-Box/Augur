"""Editable experiment file.

Change the constants below for the current run, then execute:

    python train.py
"""

import warnings
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
import torch

from training.data_config import build_datamodule
from training.model_config import MODEL_REGISTRY

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
torch.set_float32_matmul_precision("medium")


# ── Model and dataset ─────────────────────────────────────────────────
# Fixed model parameters live in training/model_config.py and utils/model_params.py.
# Fixed dataset parameters live in training/data_config.py.
MODEL_NAME = "UNet"
DATASET_NAME = "hdf5"

MODEL_CONFIG = MODEL_REGISTRY[MODEL_NAME]()
MODEL = MODEL_CONFIG.build_model()


# ── Trainer runtime settings ──────────────────────────────────────────
SEED = 42  # Random seed used by Lightning and SEVIR data splitting.
BATCH_SIZE = 100  # Per-device batch size passed to the datamodule.
NUM_WORKERS = 128  # DataLoader worker count.
MAX_EPOCHS = 300  # Maximum training epochs before Trainer stops.
DEVICES = [0, 1]  # GPU ids passed to Lightning Trainer.
PRECISION = MODEL_CONFIG.precision  # Usually keep the model default precision.
LOG_EVERY_N_STEPS = 20  # Lightning logging interval.

DATAMODULE = build_datamodule(
    dataset_name=DATASET_NAME,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=SEED,
)

pl.seed_everything(SEED, workers=True)

# ── Callbacks ─────────────────────────────────────────────────────────
CALLBACKS: list[Any] = [
    ModelCheckpoint(
        dirpath=f"checkpoints/{DATASET_NAME}/{MODEL_NAME}/",
        filename="{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    ),
    LearningRateMonitor(logging_interval="step"),
    RichProgressBar(),
    RichModelSummary(max_depth=4),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
    ),
]


# ── Logger and strategy ───────────────────────────────────────────────
LOGGER = CSVLogger(
    save_dir=f"logs/{DATASET_NAME}",
    name=MODEL_NAME,
    flush_logs_every_n_steps=50,
)
STRATEGY = DDPStrategy(find_unused_parameters=True)


# ── Trainer ───────────────────────────────────────────────────────────
TRAINER = pl.Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="gpu",
    devices=DEVICES,
    precision=PRECISION,
    log_every_n_steps=LOG_EVERY_N_STEPS,
    enable_model_summary=True,
    callbacks=CALLBACKS,
    logger=[LOGGER],
    strategy=STRATEGY,
)


if __name__ == "__main__":
    TRAINER.fit(model=MODEL, datamodule=DATAMODULE)
