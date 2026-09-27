"""Shared training utilities: optimizer groups, time-budgeted WSD schedule, logging."""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    """AdamW with decay only on matrix weights (not norms, biases, embeddings)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embedding" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = {"lr": lr, "betas": tuple(betas)}
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


class WSDSchedule:
    """Warmup -> stable -> linear decay, where the end can be set by a wall-clock budget.

    With ``budget_s > 0`` the total number of steps is re-estimated from measured step
    time, so training always finishes its decay phase inside the budget.
    """

    def __init__(self, max_lr: float, warmup: int, max_steps: int, decay_ratio: float = 0.2,
                 min_ratio: float = 0.1, budget_s: float = 0.0):
        self.max_lr, self.warmup, self.max_steps = max_lr, warmup, max_steps
        self.decay_ratio, self.min_ratio, self.budget_s = decay_ratio, min_ratio, budget_s
        self.t0 = time.time()
        self.total = max_steps
        self._decay_start: int | None = None

    def update_total(self, step: int) -> None:
        if self.budget_s <= 0 or step < 50 or self._decay_start is not None:
            return
        per_step = (time.time() - self.t0) / step
        self.total = max(step + 1, min(self.max_steps, int(self.budget_s / per_step)))

    def lr(self, step: int) -> float:
        self.update_total(step)
        if step < self.warmup:
            return self.max_lr * (step + 1) / self.warmup
        decay_start = self._decay_start if self._decay_start is not None else int(self.total * (1 - self.decay_ratio))
        if step >= decay_start:
            if self._decay_start is None:
                self._decay_start = decay_start  # freeze the plan once decay begins
            frac = min(1.0, (step - decay_start) / max(1, self.total - decay_start))
            return self.max_lr * (1 - frac * (1 - self.min_ratio))
        return self.max_lr

    def done(self, step: int) -> bool:
        return step >= self.total


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path

    def log(self, **kw: Any) -> None:
        kw = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in kw.items()}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), **kw}, default=_jsonable) + "\n")
        print(" | ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def _jsonable(o):
    """numpy / torch scalars (e.g. per-source accuracies) -> plain Python numbers."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def autocast(device: torch.device, dtype: torch.dtype = torch.bfloat16):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


def maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return module
    try:
        return torch.compile(module)
    except Exception as e:  # compile is an optimization, never a requirement
        print(f"[compile] disabled: {e}")
        return module


def cosine(x: float) -> float:
    return 0.5 * (1 + math.cos(math.pi * x))
