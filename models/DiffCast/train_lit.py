"""Top-to-bottom training script for DiffCast (Lightning port).

This file is intentionally linear: every hyperparameter is spelled out as a
plain Python literal at the top, then the script builds the LightningModule,
the Trainer, and calls ``fit``. No argparse, no helper builders, no
indirection. Edit the literals to reconfigure.

The caller is responsible for providing dataloaders. Wire them up in the
``Dataloaders`` block below.
"""
from __future__ import annotations

from typing import Sequence

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.data import DataLoader

from .ema import EMACallback
from .module import DiffCast


# =====================================================================
# Hyperparameters
# =====================================================================

# ---- data shape ----
IMG_CHANNEL: int = 1
IMG_SIZE: int = 128
FRAMES_IN: int = 5
FRAMES_OUT: int = 20

# ---- model ----
USE_DIFF: bool = True              # False -> SimVP only; True -> DiffCast (SimVP + residual diffusion)

# ---- diffusion-specific ----
ALPHA: float = 0.5                 # weight on the denoising loss in L = L_P + alpha * L_eps
SAMPLING_TIMESTEPS: int = 250
DIFF_DIM: int = 64
DIFF_DIM_MULTS: Sequence[int] = (1, 2, 4, 8)

# ---- optimizer ----
LR: float = 1e-4
LR_BETA1: float = 0.90
LR_BETA2: float = 0.95
WEIGHT_DECAY: float = 0.0

# ---- schedule ----
SCHEDULER: str = 'cosine'          # 'constant' | 'linear' | 'cosine'
WARMUP_STEPS: int = 1000

# ---- training schedule (mirrors run.py:282-286) ----
EPOCHS: int = 20
TRAINING_STEPS: int = 200_000      # max_steps = max(EPOCHS * len(train_loader), TRAINING_STEPS)
GRAD_ACC_STEPS: int = 1
GRADIENT_CLIP_VAL: float = 1.0

# ---- EMA ----
EMA_DECAY: float = 0.95
EMA_UPDATE_EVERY: int = 20

# ---- runtime ----
SEED: int = 0
RESUME_CKPT: str | None = None     # path to a Lightning .ckpt to resume from
CKPT_DIR: str | None = None        # set to a path to control where checkpoints go


# =====================================================================
# Dataloaders (provided externally - this script does not build them)
# =====================================================================
#
# Replace these two assignments with your real dataloaders. Each must yield
# tensors of shape ``[B, FRAMES_IN + FRAMES_OUT, IMG_CHANNEL, IMG_SIZE, IMG_SIZE]``
# in [0, 1].

train_loader: DataLoader | None = None
val_loader: DataLoader | None = None


def main() -> None:
    if train_loader is None:
        raise RuntimeError(
            "train_loader is None. Edit train_lit.py to construct your "
            "dataloaders, or import this module and assign train_loader / "
            "val_loader before calling main()."
        )

    pl.seed_everything(SEED, workers=True)

    model = DiffCast(
        img_channel=IMG_CHANNEL,
        img_size=IMG_SIZE,
        frames_in=FRAMES_IN,
        frames_out=FRAMES_OUT,
        use_diff=USE_DIFF,
        alpha=ALPHA,
        sampling_timesteps=SAMPLING_TIMESTEPS,
        diff_dim=DIFF_DIM,
        diff_dim_mults=DIFF_DIM_MULTS,
        lr=LR,
        lr_beta1=LR_BETA1,
        lr_beta2=LR_BETA2,
        weight_decay=WEIGHT_DECAY,
        scheduler=SCHEDULER,
        warmup_steps=WARMUP_STEPS,
    )

    steps_per_epoch = len(train_loader)
    max_steps = max(EPOCHS * steps_per_epoch, TRAINING_STEPS)

    callbacks = [
        EMACallback(decay=EMA_DECAY, update_every=EMA_UPDATE_EVERY),
        ModelCheckpoint(
            dirpath=CKPT_DIR,
            filename='ckpt-{step}',
            every_n_epochs=1,
            save_top_k=-1,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(
        max_steps=max_steps,
        max_epochs=-1,
        accumulate_grad_batches=GRAD_ACC_STEPS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=RESUME_CKPT)


if __name__ == '__main__':
    main()
