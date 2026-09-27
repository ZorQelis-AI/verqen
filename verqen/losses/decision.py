"""Decision loss: strictly proper scoring rules per question, averaged over questions.

    log score (soft cross-entropy) + brier_weight * Brier
    + rps_weight * RPS           (ordinal ``score`` questions only)
    + selective_weight * BCE     (selective head vs. whether top-1 was correct)

Every term is minimized in expectation by the true distribution, so the model is
pushed toward calibrated probabilities rather than just a correct argmax. The
reported ``kl`` subtracts the target entropy, so it reaches 0 at a perfect fit
(the raw CE does not when targets are soft).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from verqen.config import LossConfig
from verqen.model.decision_head import QTYPE_SCORE


def correctness(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """1.0 where the predicted top-1 is one of the target's argmax options."""
    pred = probs.masked_fill(~mask, -1.0).argmax(-1)
    best = target.masked_fill(~mask, -1.0).max(-1).values
    return (target.gather(1, pred[:, None]).squeeze(1) >= best - 1e-6).float()


def decision_loss(out: dict, target: torch.Tensor, qtype: torch.Tensor, cfg: LossConfig) -> dict:
    mask = out["opt_mask"]
    logp = out["logp"]
    probs = out["probs"]
    t = target
    if cfg.label_smoothing > 0:
        k = mask.sum(-1, keepdim=True).float()
        t = (1 - cfg.label_smoothing) * t + cfg.label_smoothing * mask.float() / k

    safe_logp = logp.masked_fill(~mask, 0.0)
    ce = -(t * safe_logp).sum(-1)
    ent = -(t * torch.log(t.clamp_min(1e-12))).sum(-1)
    brier = ((probs - t) ** 2 * mask).sum(-1)

    is_score = (qtype == QTYPE_SCORE).float()
    k = mask.sum(-1).clamp_min(2).float()
    rps = ((torch.cumsum(probs, -1) - torch.cumsum(t, -1)) ** 2 * mask).sum(-1) / (k - 1)

    correct = correctness(probs.detach(), target, mask)
    sel = F.binary_cross_entropy_with_logits(out["selective_logit"], correct, reduction="none")

    per_q = ce + cfg.brier_weight * brier + cfg.rps_weight * rps * is_score
    loss = per_q.mean() + cfg.selective_weight * sel.mean()
    return {
        "loss": loss,
        "ce": ce.mean().detach(),
        "kl": (ce - ent).mean().detach(),
        "brier": brier.mean().detach(),
        "acc": correct.mean().detach(),
        "selective_bce": sel.mean().detach(),
        "n_questions": torch.tensor(float(len(qtype))),
    }
