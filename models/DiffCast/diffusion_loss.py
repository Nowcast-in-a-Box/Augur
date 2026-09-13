"""Residual diffusion training loss for DiffCast (paper Algorithm 2).

Computes ``L = L_P + alpha * L_eps`` *without* modifying ``diffcast.py``:

    * ``L_P``    is the deterministic backbone loss (e.g. SimVP MSE on mu vs y).
    * ``L_eps``  is the denoising loss on the residual ``r = y - mu``, evaluated
                 over every length-``T_in`` segment with a single shared
                 diffusion timestep.

Teacher forcing
---------------
At inference time, the per-segment ``cond`` argument fed to the U-Net is the
*predicted* residual of the previous segment (``pre_frag - pre_mu`` inside
``GaussianDiffusion.sample``). At training time we instead use the
ground-truth previous-segment residual; this is the standard
non-autoregressive surrogate and matches Algorithm 2 in the paper.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .diffcast import extract

if TYPE_CHECKING:
    from .diffcast import GaussianDiffusion


def _q_sample(
    gd: 'GaussianDiffusion',
    x_start: Tensor,
    t: Tensor,
    noise: Tensor,
) -> Tensor:
    """Forward diffusion ``q(x_t | x_0)``.

    Mirrors the closed-form used elsewhere in ``diffcast.py``.

    Shapes:
        x_start, noise: ``[B, T_in, C, H, W]``
        t:              ``[B]`` (long, in ``[0, num_timesteps)``)
        returns:        same shape as ``x_start``
    """
    return (
        extract(gd.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + extract(gd.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )


def _denoising_target(
    gd: 'GaussianDiffusion',
    x_start: Tensor,
    t: Tensor,
    noise: Tensor,
) -> Tensor:
    """Pick the U-Net regression target according to ``gd.objective``."""
    if gd.objective == 'pred_noise':
        return noise
    if gd.objective == 'pred_x0':
        return x_start
    if gd.objective == 'pred_v':
        return gd.predict_v(x_start, t, noise)
    raise ValueError(f"Unknown diffusion objective: {gd.objective!r}")


def residual_diffusion_loss(
    gd: 'GaussianDiffusion',
    frames_in: Tensor,
    frames_gt: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """Compute ``L = L_P + alpha * L_eps`` for one minibatch.

    Args:
        gd: a ``GaussianDiffusion`` instance with a backbone already attached
            via ``load_backbone``.
        frames_in:  ``[B, T_in,  C, H, W]`` input frames in [0, 1].
        frames_gt:  ``[B, T_out, C, H, W]`` ground-truth future frames in [0, 1].
        alpha: weight of the denoising loss in the combined objective.

    Returns:
        Scalar tensor (the combined loss).
    """
    # ---------- L_P : deterministic backbone loss ---------------------------
    # The backbone's predict() returns (mu, loss) when compute_loss=True, so
    # we run the backbone exactly once and reuse mu in the diffusion branch.
    mu, l_p = gd.backbone_net.predict(frames_in, frames_gt, compute_loss=True)
    # mu: [B, T_out, C, H, W]    l_p: scalar

    # ---------- normalize to [-1, 1] for the diffusion U-Net ----------------
    x_norm = gd.normalize(frames_in)    # [B, T_in,  C, H, W]
    mu_norm = gd.normalize(mu)          # [B, T_out, C, H, W]
    gt_norm = gd.normalize(frames_gt)   # [B, T_out, C, H, W]

    # ---------- multi-scale ConvGRU context over [x ; mu] -------------------
    # ContextNet returns (global_ctx, local_ctx); each is a list of multi-
    # scale hidden-state tensors (one per ConvGRU layer). The first segment
    # uses local_ctx (state at t=T_in, i.e. just the input frames), subsequent
    # segments use global_ctx (state at the end of the full sequence). This
    # matches the dispatch in GaussianDiffusion.sample().
    global_ctx, local_ctx = gd.ctx_net.scan_ctx(torch.cat([x_norm, mu_norm], dim=1))

    # ---------- residual sequence r = y - mu, then split into K segments ----
    residual = gt_norm - mu_norm                     # [B, T_out, C, H, W]
    B, T_out = residual.shape[0], residual.shape[1]
    T_in = frames_in.shape[1]
    K = T_out // T_in                                # number of segments

    # Shared diffusion timestep across all segments (paper Algorithm 2, line 7).
    t = torch.randint(
        0, gd.num_timesteps, (B,),
        device=residual.device, dtype=torch.long,
    )                                                # [B]

    l_eps: Tensor = residual.new_zeros(())
    prev_residual: Tensor | None = None
    for j in range(K):
        s_j = residual[:, j * T_in : (j + 1) * T_in]   # [B, T_in, C, H, W]

        cond = prev_residual if prev_residual is not None else torch.zeros_like(s_j)
        ctx = local_ctx if j == 0 else global_ctx
        idx = torch.full((B,), j, device=s_j.device, dtype=torch.long)  # [B]

        noise = torch.randn_like(s_j)                  # [B, T_in, C, H, W]
        s_t = _q_sample(gd, s_j, t, noise)             # [B, T_in, C, H, W]

        pred = gd.model(s_t, t, cond=cond, ctx=ctx, idx=idx)   # [B, T_in, C, H, W]
        target = _denoising_target(gd, s_j, t, noise)          # [B, T_in, C, H, W]

        l_eps = l_eps + F.mse_loss(pred, target)

        prev_residual = s_j   # teacher forcing with the GT residual

    return l_p + alpha * (l_eps / K)
