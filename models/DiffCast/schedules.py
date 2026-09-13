"""Local replacements for the three diffusers schedules.

Each function returns a ``torch.optim.lr_scheduler.LambdaLR`` whose multiplier
is computed from the current optimizer step. The formulas match the
``diffusers``/``transformers`` reference implementations exactly:

    constant : 1.0 once warmup completes.
    linear   : decays linearly to 0 over the remaining steps.
    cosine   : single half-cosine from 1.0 to 0.0 over the remaining steps.

A shared warmup ramp ``step / max(1, num_warmup_steps)`` is applied for
``step < num_warmup_steps``.
"""
from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _warmup_factor(step: int, num_warmup_steps: int) -> float:
    return step / max(1, num_warmup_steps)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return _warmup_factor(step, num_warmup_steps)
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return _warmup_factor(step, num_warmup_steps)
        remaining = num_training_steps - num_warmup_steps
        progress = (step - num_warmup_steps) / max(1, remaining)
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return _warmup_factor(step, num_warmup_steps)
        remaining = num_training_steps - num_warmup_steps
        progress = (step - num_warmup_steps) / max(1, remaining)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return LambdaLR(optimizer, lr_lambda)


SCHEDULE_FACTORIES = {
    'constant': get_constant_schedule_with_warmup,
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
}
