"""Self-contained checkpoints: weights + model config + tokenizer + calibration in one directory.

    checkpoint-XXXX/
        model.safetensors
        model_config.json
        tokenizer.json        (copied, so a checkpoint can never load the wrong vocabulary)
        calibration.json      (optional: temperatures, conformal/selective thresholds)
        training_state.pt     (optional: optimizer + step)
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import struct
from typing import Any, Optional

import safetensors.torch
import torch


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, output_dir: str, tokenizer_path: str | None = None, optimizer=None, step: int = 0,
                    extra: dict[str, Any] | None = None, calibration: dict[str, Any] | None = None) -> str:
    model = _unwrap(model)
    os.makedirs(output_dir, exist_ok=True)
    state = {k.replace("_orig_mod.", ""): v.detach().contiguous() for k, v in model.state_dict().items()}
    safetensors.torch.save_file(state, os.path.join(output_dir, "model.safetensors"))
    with open(os.path.join(output_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(model.config), f, indent=2)
    if tokenizer_path and os.path.exists(tokenizer_path):
        dst = os.path.join(output_dir, "tokenizer.json")
        if os.path.abspath(tokenizer_path) != os.path.abspath(dst):
            shutil.copyfile(tokenizer_path, dst)
    if calibration is not None:
        with open(os.path.join(output_dir, "calibration.json"), "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
    if optimizer is not None or extra:
        torch.save({"step": step, "optimizer": optimizer.state_dict() if optimizer else None, **(extra or {})},
                   os.path.join(output_dir, "training_state.pt"))
    return output_dir


_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
}


CHECKPOINT_FILES = ("model.safetensors", "model_config.json", "tokenizer.json", "calibration.json")


def resolve_checkpoint(path: str, revision: str | None = None) -> str:
    """A local directory as is, or a Hugging Face repo id ("zorqelis-ai/VerQen") downloaded to the
    local cache (only the four files a checkpoint needs; later calls reuse the cache)."""
    if os.path.isdir(path):
        return path
    if path.count("/") == 1 and not os.path.exists(path):
        from huggingface_hub import snapshot_download
        return snapshot_download(path, revision=revision, allow_patterns=list(CHECKPOINT_FILES))
    raise FileNotFoundError(f"no checkpoint directory or Hugging Face repo id: {path}")


def load_model(checkpoint_dir: str, device: torch.device | str = "cpu"):
    from verqen.config import ModelConfig
    from verqen.model.verqen import VerqenModel

    with open(os.path.join(checkpoint_dir, "model_config.json"), encoding="utf-8") as f:
        cfg = ModelConfig.from_dict(json.load(f))
    target_device = torch.device(device)
    with torch.device(target_device):
        model = VerqenModel(cfg)

    model_path = os.path.join(checkpoint_dir, "model.safetensors")
    sd = model.state_dict()

    # Stream tensors directly from binary file without CreateFileMappingW to avoid Windows OS Error 1455
    loaded = False
    try:
        with open(model_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size).decode("utf-8"))
            base = 8 + header_size
            with torch.no_grad():
                for name, meta in header.items():
                    if name == "__metadata__":
                        continue
                    if name in sd:
                        start, end = meta["data_offsets"]
                        f.seek(base + start)
                        buf = bytearray(f.read(end - start))
                        dtype = _DTYPE_MAP.get(meta["dtype"], torch.float32)
                        t = torch.frombuffer(buf, dtype=dtype).reshape(meta["shape"])
                        sd[name].copy_(t.to(target_device))
                        del buf, t
            loaded = True
    except Exception:
        pass

    if not loaded:
        backend = "pread" if os.name == "nt" else "mmap"
        with safetensors.safe_open(model_path, framework="pt", device="cpu", backend=backend) as sf:
            with torch.no_grad():
                for k in sf.keys():
                    if k in sd:
                        t = sf.get_tensor(k)
                        sd[k].copy_(t.to(target_device))
                        del t

    return model


def load_calibration(checkpoint_dir: str) -> Optional[dict[str, Any]]:
    p = os.path.join(checkpoint_dir, "calibration.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(model, checkpoint_dir: str, optimizer=None, device=None) -> int:
    """Back-compat: load weights (and optimizer) into an existing model; returns step."""
    safetensors.torch.load_model(_unwrap(model), os.path.join(checkpoint_dir, "model.safetensors"))
    p = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(p):
        st = torch.load(p, map_location="cpu", weights_only=False)
        if optimizer is not None and st.get("optimizer"):
            optimizer.load_state_dict(st["optimizer"])
        return int(st.get("step", 0))
    return 0
