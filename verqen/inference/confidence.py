"""Confidence estimation, Expected Calibration Error (ECE), and escalation utilities.

Inspired by Laya Framework's RLCD calibration and entropy-based confidence scoring:
- confidence_from_probs: Normalized Shannon entropy confidence: 1 - H(p) / log(k)
- margin_from_probs: top1 - top2 probability separation margin
- ece_score: Expected Calibration Error across confidence bins
- should_escalate: deterministic gating rule
- DecisionEscalator: unified escalation evaluation runtime
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch


def confidence_from_probs(
    probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    k: Optional[int] = None,
) -> float:
    """Compute normalized Shannon entropy confidence: 1 - H(p) / log(k).

    Guarantees:
    - Uniform distribution -> 0.0 confidence (maximum uncertainty)
    - One-hot distribution -> 1.0 confidence (absolute certainty)
    - Always bounded in [0.0, 1.0]
    """
    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().float().numpy()
    elif isinstance(probs, (list, tuple)):
        p = np.array(probs, dtype=np.float32)
    else:
        p = np.asarray(probs, dtype=np.float32)

    if k is None:
        k = len(p)

    if k < 2:
        return 1.0

    p_valid = p[:k]
    s = p_valid.sum()
    if s > 0:
        p_norm = p_valid / s
    else:
        return 0.0

    p_safe = np.clip(p_norm, 1e-12, 1.0)
    entropy = float(-np.sum(p_safe * np.log(p_safe)))
    max_entropy = math.log(max(k, 2))

    conf = 1.0 - (entropy / max_entropy)
    return float(np.clip(conf, 0.0, 1.0))


def margin_from_probs(
    probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
    k: Optional[int] = None,
) -> float:
    """Compute top1 - top2 margin (separation between best and runner-up options)."""
    if isinstance(probs, torch.Tensor):
        p = probs.detach().cpu().float().numpy()
    elif isinstance(probs, (list, tuple)):
        p = np.array(probs, dtype=np.float32)
    else:
        p = np.asarray(probs, dtype=np.float32)

    if k is not None:
        p = p[:k]

    if len(p) < 2:
        return 1.0

    sorted_p = np.sort(p)[::-1]
    return float(np.clip(sorted_p[0] - sorted_p[1], 0.0, 1.0))


def ece_score(
    conf: np.ndarray,
    correct: np.ndarray,
    bins: int = 15,
) -> float:
    """Expected Calibration Error (ECE) across confidence bins.

    ECE = sum_{b} (|B_b| / N) * |acc(B_b) - conf(B_b)|
    """
    if len(conf) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    total = len(conf)

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(conf[mask]))
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return float(ece)


def should_escalate(confidence: float, threshold: float = 0.70) -> bool:
    """Determine whether a decision should be escalated for human / frontier LLM review."""
    return bool(confidence < threshold)


class DecisionEscalator:
    """Evaluates whether to automate (ACT) or route to human review (ESCALATE)."""

    def __init__(self, confidence_threshold: float = 0.70, min_margin: float = 0.15) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_margin = min_margin

    def evaluate(
        self,
        probs: Union[torch.Tensor, np.ndarray, Sequence[float]],
        question_type: str = "choice",
        act_logits: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """Evaluate escalation status using both rule-based and neural signals.

        Returns:
            dict containing:
                confidence: float in [0, 1]
                margin: float in [0, 1]
                should_escalate: bool
                reason: str explanation
        """
        conf = confidence_from_probs(probs)
        margin = margin_from_probs(probs)

        # 1. Check neural act head if available
        neural_escalate = False
        if act_logits is not None:
            act_p = torch.softmax(act_logits.float(), dim=-1)
            # Class 1 is ESCALATE
            if act_p.size(-1) >= 2 and act_p[1].item() > 0.5:
                neural_escalate = True

        # 2. Threshold gating
        threshold_escalate = conf < self.confidence_threshold
        margin_escalate = margin < self.min_margin

        escalate = neural_escalate or threshold_escalate or margin_escalate

        reason = "Automated"
        if neural_escalate:
            reason = "Escalated by neural escalation head"
        elif threshold_escalate:
            reason = f"Confidence ({conf:.2f}) below threshold ({self.confidence_threshold:.2f})"
        elif margin_escalate:
            reason = f"Decision margin ({margin:.2f}) below minimum ({self.min_margin:.2f})"

        return {
            "confidence": round(conf, 4),
            "margin": round(margin, 4),
            "should_escalate": escalate,
            "reason": reason,
        }
