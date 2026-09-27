"""Post-hoc temperature scaling for probability calibration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import optim


class TemperatureScaler(nn.Module):
    """Calibrates output distributions via learned temperature T: p = softmax(z / T)."""

    def __init__(self) -> None:
        super().__init__()
        # Parameterize as log_temperature to guarantee T > 0 strictly
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_temperature).clamp(min=0.05, max=20.0)
        return logits / T

    def fit(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        lr: float = 0.05,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature parameter based on NLL on a validation set."""
        # Clean logits to avoid inf/overflow
        clean_logits = np.clip(logits, -50.0, 50.0)
        logits_tensor = torch.tensor(clean_logits, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.long)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam([self.log_temperature], lr=lr)

        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled = self.temperature_scale(logits_tensor)
            loss = nll_criterion(scaled, targets_tensor)
            if torch.isnan(loss):
                break
            loss.backward()
            optimizer.step()

        return self.temperature

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature and return scaled probabilities."""
        with torch.no_grad():
            clean_logits = np.clip(logits, -50.0, 50.0)
            logits_tensor = torch.tensor(clean_logits, dtype=torch.float32)
            scaled = self.temperature_scale(logits_tensor)
            probs = torch.softmax(scaled, dim=-1)
            return probs.numpy()
