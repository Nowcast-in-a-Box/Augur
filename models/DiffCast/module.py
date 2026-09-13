"""LightningModule wrapping the DiffCast model.

The wrapped ``self.model`` is one of:

    * SimVP backbone alone                   when ``use_diff = False``
    * ``GaussianDiffusion(SimVP)`` (DiffCast) when ``use_diff = True``

Training loss:

    * ``use_diff=False`` -> ``backbone.predict(fin, fout, compute_loss=True)``.
    * ``use_diff=True``  -> :func:`residual_diffusion_loss` (paper Algorithm 2).

Validation re-runs the same loss with EMA-smoothed weights and logs
``val/loss``. EMA is created and updated by :class:`EMACallback`.
"""
from __future__ import annotations

from typing import Sequence

import lightning.pytorch as pl
import torch
from torch import Tensor

from .diffcast import get_model as get_diffcast
from .diffusion_loss import residual_diffusion_loss
from .ema import ConstantEMA
from .schedules import SCHEDULE_FACTORIES
from .simvp import get_model as get_simvp


class DiffCast(pl.LightningModule):
    def __init__(
        self,
        # ---- data shape ----
        img_channel: int = 1,
        img_size: int = 128,
        frames_in: int = 5,
        frames_out: int = 20,
        # ---- model selection ----
        use_diff: bool = False,
        # ---- diffusion-specific ----
        alpha: float = 0.5,
        sampling_timesteps: int = 250,
        diff_dim: int = 64,
        diff_dim_mults: Sequence[int] = (1, 2, 4, 8),
        # ---- optimizer ----
        lr: float = 1e-4,
        lr_beta1: float = 0.90,
        lr_beta2: float = 0.95,
        weight_decay: float = 0.0,
        # ---- learning rate schedule ----
        scheduler: str = 'cosine',
        warmup_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Cache the few hyperparameters that the step methods need on the hot path.
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.use_diff = use_diff
        self.alpha = alpha

        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps

        # ---- build SimVP backbone ----
        backbone = get_simvp(
            in_shape=(img_channel, img_size, img_size),
            T_in=frames_in,
            T_out=frames_out,
        )

        if use_diff:
            diff_model = get_diffcast(
                img_channels=img_channel,
                dim=diff_dim,
                dim_mults=tuple(diff_dim_mults),
                T_in=frames_in,
                T_out=frames_out,
                sampling_timesteps=sampling_timesteps,
            )
            diff_model.load_backbone(backbone)
            self.model: torch.nn.Module = diff_model
        else:
            self.model = backbone

        # Populated by EMACallback.setup; remains None when EMACallback is not used.
        self.ema: ConstantEMA | None = None

    # ------------------------------------------------------------------ loss

    def _compute_loss(self, model: torch.nn.Module, batch: Tensor) -> Tensor:
        """Single source of truth for both training and validation loss.

        ``batch`` shape: ``[B, frames_in + frames_out, C, H, W]`` in [0, 1].
        """
        frames_in = batch[:, : self.frames_in]   # [B, T_in,  C, H, W]
        frames_gt = batch[:, self.frames_in :]   # [B, T_out, C, H, W]

        if self.use_diff:
            return residual_diffusion_loss(model, frames_in, frames_gt, alpha=self.alpha)

        _, loss = model.predict(frames_in, frames_gt, compute_loss=True)
        return loss

    # ------------------------------------------------------------------ steps

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = self._compute_loss(self.model, batch)
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        # Use EMA-smoothed weights when available; before the first EMA update
        # we fall back to the live weights so validation always has a model.
        eval_model = self.ema.module if self.ema is not None else self.model
        loss = self._compute_loss(eval_model, batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)

    # ------------------------------------------------------------------ optim

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            weight_decay=self.weight_decay,
        )

        if self.scheduler_name not in SCHEDULE_FACTORIES:
            raise ValueError(
                f"Unknown scheduler {self.scheduler_name!r}; "
                f"expected one of {sorted(SCHEDULE_FACTORIES)}"
            )
        factory = SCHEDULE_FACTORIES[self.scheduler_name]

        sched_kwargs: dict = {'num_warmup_steps': self.warmup_steps}
        if self.scheduler_name != 'constant':
            # Lightning resolves max_steps from Trainer; this property gives us
            # the same number the scheduler needs to span warmup + decay.
            sched_kwargs['num_training_steps'] = self.trainer.estimated_stepping_batches

        scheduler = factory(optimizer, **sched_kwargs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
