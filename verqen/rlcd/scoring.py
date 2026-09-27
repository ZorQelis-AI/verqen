"""Strictly Proper Scoring Rules for Verqen-RLCD as per Section 61 & 62."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict


def log_scoring_rule(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-12,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """Logarithmic scoring rule (information gain): S(p, y) = log p_y."""
    p_masked = (probs * mask).clamp_min(eps)
    log_p = torch.log(p_masked).clamp_min(log_floor)
    score = (targets * log_p).sum(dim=-1)
    return score


def brier_scoring_rule(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Brier scoring rule: S(p, y) = - ||p - y||^2."""
    diff_sq = ((probs - targets) ** 2) * mask
    return -diff_sq.sum(dim=-1)


def spherical_scoring_rule(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Spherical scoring rule: S(p, y) = p_y / ||p||_2."""
    p_masked = probs * mask
    norm = p_masked.norm(dim=-1).clamp_min(eps)
    dot = (targets * p_masked).sum(dim=-1)
    return dot / norm


def ranked_probability_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Ranked Probability Score (RPS) for ordinal scales."""
    k = mask.sum(dim=-1).clamp(min=2).float()
    cdf_p = torch.cumsum(probs * mask, dim=-1)
    cdf_t = torch.cumsum(targets * mask, dim=-1)
    rps = (((cdf_p - cdf_t) ** 2) * mask).sum(dim=-1) / (k - 1)
    return rps
