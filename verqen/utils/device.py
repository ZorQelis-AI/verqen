"""Device and dtype detection utilities."""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(mixed_precision: str = "auto") -> torch.dtype:
    """Return the AMP dtype based on *mixed_precision* config string.

    "auto" — picks bf16 if CUDA supports it, else fp16 on CUDA, else fp32.
    "bf16" / "fp16" / "fp32" — explicit selection.
    """
    if mixed_precision == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return torch.float32
