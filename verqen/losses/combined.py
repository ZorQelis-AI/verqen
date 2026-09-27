"""Back-compat wrapper around :func:`verqen.losses.decision.decision_loss`."""

from __future__ import annotations

import torch.nn as nn

from verqen.config import LossConfig
from verqen.losses.decision import correctness, decision_loss


class CombinedLoss(nn.Module):
    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, outputs: dict, batch: dict) -> dict:
        return decision_loss(outputs, batch["target"], batch["qtype"], self.config)


__all__ = ["CombinedLoss", "decision_loss", "correctness"]
