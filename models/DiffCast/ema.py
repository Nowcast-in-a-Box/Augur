"""Constant-decay EMA built on ``torch.optim.swa_utils.AveragedModel``.

This replaces the third-party ``ema_pytorch`` package. The original code uses
its simplest configuration (``beta=0.95``, ``update_every=20``), which is
just a constant exponential moving average with a fixed update cadence:

    new = decay * new + (1 - decay) * cur     (every ``update_every``-th call)

Usage:

    pl_module.ema = ConstantEMA(model, decay=0.95, update_every=20)
    # ... in on_train_batch_end:
    pl_module.ema.update()
    # ... at validation time:
    eval_model = pl_module.ema.module

``ConstantEMA.module`` is the model whose parameters are the EMA-averaged
copies (``AveragedModel`` exposes the wrapped network as ``.module``).
"""
from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class ConstantEMA:
    """Constant-decay EMA with throttled updates.

    Behaviour matches ``ema_pytorch.EMA(beta=decay, update_every=update_every)``
    in its simplest configuration: every ``update_every``-th call mixes the
    *current* live weights into the running average.

    The very first non-throttled call seeds the average by copying the live
    weights (this is what ``AveragedModel`` does on the first
    ``update_parameters`` invocation), so the EMA starts from a meaningful
    state rather than the raw initialization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.95,
        update_every: int = 20,
    ) -> None:
        self.update_every = update_every
        self._call_count = 0
        # Hold a reference to the live model so update_parameters reads fresh weights.
        self._live_model = model
        # PyTorch's helper builds the canonical EMA averaging function for us.
        self._averaged = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))

    @property
    def module(self) -> torch.nn.Module:
        """The EMA-smoothed model, suitable for evaluation."""
        return self._averaged.module

    def update(self) -> None:
        self._call_count += 1
        if self._call_count % self.update_every != 0:
            return
        self._averaged.update_parameters(self._live_model)

    # ----- checkpoint helpers ----------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            'averaged': self._averaged.state_dict(),
            'call_count': self._call_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._averaged.load_state_dict(state['averaged'])
        self._call_count = state.get('call_count', 0)


class EMACallback(pl.Callback):
    """Lightning callback that owns a :class:`ConstantEMA` and persists it.

    The EMA is created in ``setup`` (so device placement is correct by then)
    and advanced in ``on_train_batch_end`` so it tracks training and is
    *not* updated during validation/test (which has no gradient and does not
    invoke ``on_train_batch_end``).
    """

    def __init__(self, decay: float = 0.95, update_every: int = 20) -> None:
        super().__init__()
        self.decay = decay
        self.update_every = update_every

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        stage: str,
    ) -> None:
        if getattr(pl_module, 'ema', None) is None:
            pl_module.ema = ConstantEMA(
                pl_module.model,
                decay=self.decay,
                update_every=self.update_every,
            )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        pl_module.ema.update()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        checkpoint['ema'] = pl_module.ema.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, Any],
    ) -> None:
        if 'ema' in checkpoint and getattr(pl_module, 'ema', None) is not None:
            pl_module.ema.load_state_dict(checkpoint['ema'])
