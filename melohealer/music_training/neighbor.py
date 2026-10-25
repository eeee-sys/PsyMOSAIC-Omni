"""Neighbor GRPO geometry from the PsyMOSAIC design, without SDE conversion.

Distances are squared L2 SUMS over the complete latent, with no temperature.
Only current anchor predictions contribute gradients. Rollout candidates, old
predictions, and reward-derived advantages are fixed targets.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _CorePath
for _core_root in _CorePath(__file__).resolve().parents:
    if (_core_root / "project_paths.py").is_file():
        if str(_core_root) not in _sys.path:
            _sys.path.insert(0, str(_core_root))
        break
from project_paths import (load_config, expand_paths, get_api_key, LORA_ROOT,
                           MODEL_ROOT, OUTPUT_ROOT, RAW_DATA_ROOT, VENDOR_ROOT,
                           CODE_ROOT as PROJECT_CODE_ROOT, DATA_ROOT as PROJECT_DATA_ROOT)


from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class AdvantageResult:
    standardized: Tensor
    reweighted: Tensor
    reward_mean: Tensor
    reward_std: Tensor
    quasi_norm: Tensor
    skip_group: bool


@dataclass(frozen=True)
class NeighborLossResult:
    loss: Tensor
    ratios: Tensor
    log_ratios: Tensor
    log_probs: Tensor
    old_log_probs: Tensor
    distances: Tensor
    old_distances: Tensor
    entropy: Tensor
    normalized_entropy: Tensor
    saturated_fraction: Tensor
    clip_fraction: Tensor
    skip_group: bool


def sample_correlated_noise(
    group_size: int,
    latent_shape: Sequence[int],
    sigma: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return [G, *latent_shape] standard normal marginals with corr=1-sigma²."""
    if group_size < 2:
        raise ValueError("A neighbor group needs at least two candidates")
    if not 0.0 <= sigma <= 1.0 or not math.isfinite(sigma):
        raise ValueError("sigma must be finite and in [0, 1]")
    shape = tuple(int(n) for n in latent_shape)
    if not shape or any(n < 1 for n in shape):
        raise ValueError("latent_shape must contain positive dimensions")
    shared = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    independent = torch.randn(
        (group_size, *shape), device=device, dtype=dtype, generator=generator
    )
    return math.sqrt(1.0 - sigma * sigma) * shared + sigma * independent


def group_advantages(
    rewards: Tensor, *, p: float = 0.8, std_epsilon: float = 1e-6
) -> AdvantageResult:
    """Standardize within one group, then apply the design's Lp quasi-norm.

    Population standard deviation (correction=0) makes the definition explicit.
    A constant reward group has no preference signal and is marked for skipping.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("rewards must be a vector with at least two entries")
    if not 0.0 < p <= 2.0 or not math.isfinite(p):
        raise ValueError("p must be finite and in (0, 2]")
    if std_epsilon <= 0.0 or not math.isfinite(std_epsilon):
        raise ValueError("std_epsilon must be positive and finite")
    values = rewards.detach().to(
        dtype=torch.float64 if rewards.dtype == torch.float64 else torch.float32
    )
    if not torch.isfinite(values).all():
        raise ValueError("Reward values must be finite")
    mean = values.mean()
    std = values.std(correction=0)
    standardized = (values - mean) / (std + std_epsilon)
    norm = standardized.abs().pow(p).sum().pow(1.0 / p)
    skip = not bool(torch.any(standardized != 0).item())
    reweighted = torch.zeros_like(values) if skip else standardized / norm
    return AdvantageResult(standardized, reweighted, mean, std, norm, skip)


def squared_l2_log_policy(
    anchor_predictions: Tensor, candidate_next_latents: Tensor
) -> tuple[Tensor, Tensor]:
    """Compute [B, G] log-softmax(-raw squared L2 sum) and distances.

    Candidates have shape [G, *latent_shape]. Anchors may be [*latent_shape]
    or [B, *latent_shape], including the ACE layout [B, T, 64]. Distances are
    evaluated in FP32 (FP64 when either input is FP64), including under autocast.
    """
    candidates = candidate_next_latents.detach()
    if candidates.ndim < 2 or candidates.shape[0] < 2:
        raise ValueError("candidates need shape [G, *latent_shape] with G >= 2")
    anchors = anchor_predictions
    if anchors.shape == candidates.shape[1:]:
        anchors = anchors.unsqueeze(0)
    if anchors.ndim != candidates.ndim or anchors.shape[1:] != candidates.shape[1:]:
        raise ValueError("Anchor and candidate latent shapes must match exactly")
    if anchors.shape[0] < 1 or anchors.device != candidates.device:
        raise ValueError("Anchors must be nonempty and share the candidate device")
    dtype = (
        torch.float64
        if torch.float64 in (anchors.dtype, candidates.dtype)
        else torch.float32
    )
    # Direct subtraction avoids cancellation in ||a||² + ||b||² - 2a·b.
    difference = anchors.to(dtype).unsqueeze(1) - candidates.to(dtype).unsqueeze(0)
    distances = difference.square().flatten(start_dim=2).sum(dim=-1)
    if not torch.isfinite(distances).all():
        raise FloatingPointError("Nonfinite raw squared L2 neighbor distances")
    return torch.log_softmax(-distances, dim=-1), distances


def neighbor_grpo_loss(
    current_anchor: Tensor,
    old_anchor: Tensor,
    candidate_next_latents: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float = 0.2,
    entropy_floor: float = 0.01,
) -> NeighborLossResult:
    """Mean PPO-clipped neighbor objective for one sampled ODE time step.

    Log-softmax differences preserve ratios even when probabilities underflow to
    zero. Ratios are never silently clamped beyond the specified PPO clipping.
    A nonfinite ratio raises an explicit error instead of modifying the method.
    Low entropy is diagnostic, not a reason to skip a group: the all-candidate
    objective can retain a useful gradient when the surrogate is near one-hot.
    """
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    if not 0.0 <= entropy_floor <= 1.0:
        raise ValueError("entropy_floor must be in [0, 1]")
    if advantages.ndim != 1 or advantages.numel() != candidate_next_latents.shape[0]:
        raise ValueError("advantages must contain one value per candidate")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite")
    log_probs, distances = squared_l2_log_policy(current_anchor, candidate_next_latents)
    with torch.no_grad():
        old_log_probs, old_distances = squared_l2_log_policy(
            old_anchor.detach(), candidate_next_latents
        )
    if old_log_probs.shape != log_probs.shape:
        raise ValueError("Old and current anchor counts must match")
    log_ratios = log_probs - old_log_probs
    ratios = log_ratios.exp()
    if not torch.isfinite(ratios).all():
        raise FloatingPointError(
            "Neighbor policy ratio overflow; inspect step size and log-ratio diagnostics"
        )
    fixed_advantages = advantages.detach().to(device=ratios.device, dtype=ratios.dtype)
    skip = not bool(torch.any(fixed_advantages != 0).item())
    unclipped = ratios * fixed_advantages
    clipped = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * fixed_advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    with torch.no_grad():
        probabilities = log_probs.detach().exp()
        entropy = -(probabilities * log_probs.detach()).sum(dim=-1)
        normalized_entropy = entropy / math.log(log_probs.shape[-1])
        saturated_fraction = (normalized_entropy < entropy_floor).float().mean()
        clip_fraction = ((ratios.detach() - 1.0).abs() > clip_epsilon).float().mean()
    return NeighborLossResult(
        loss, ratios, log_ratios, log_probs, old_log_probs, distances,
        old_distances, entropy, normalized_entropy, saturated_fraction,
        clip_fraction, skip,
    )
