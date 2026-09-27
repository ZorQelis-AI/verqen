"""VerQen-T1 configuration dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """Backbone + decision head configuration."""
    name: str = "verqen-t1"
    hidden_size: int = 576
    num_layers: int = 12
    num_heads: int = 9
    head_dim: int = 64
    ffn_size: int = 1536
    vocab_size: int = 32768
    max_seq_len: int = 512
    rope_base: float = 10000.0
    num_segments: int = 6          # pad, bos, state, question, option, decide
    set_layers: int = 2            # permutation-equivariant option-comparison layers
    max_options: int = 64
    init_std: float = 0.02
    norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True    # MLM head shares the input embedding

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    decay_ratio: float = 0.2        # WSD schedule: final fraction spent decaying
    grad_clip: float = 1.0
    mixed_precision: str = "bf16"
    batch_rows: int = 64            # rows of max_seq_len tokens per optimizer step
    micro_rows: int = 32
    decision_micro_rows: int = 0    # decisions / RLCD rows are data.max_seq_len long, not pretrain.seq_len: sized separately (0 = micro_rows)
    max_steps: int = 10000
    time_budget_hours: float = 0.0  # >0: stop (and still decay) when this wall-clock budget is spent
    betas: tuple[float, float] = (0.9, 0.95)
    eval_every: int = 1000
    compile: bool = True


@dataclass
class LossConfig:
    brier_weight: float = 0.5
    rps_weight: float = 1.0
    selective_weight: float = 0.1
    label_smoothing: float = 0.0


@dataclass
class DataConfig:
    tokenizer_path: str = "data/tokenizer/verqen_32k.json"
    pretrain_bin: str = "data/pretrain/fineweb_edu_full.bin"
    decision_dir: str = "data/decision/v2"
    max_seq_len: int = 512
    num_workers: int = 4
    mlm_probability: float = 0.3


@dataclass
class LoggingConfig:
    output_dir: str = "runs/verqen-t1"
    log_every: int = 50
    save_every: int = 2000


@dataclass
class PretrainConfig:
    """T2 pretraining (verqen/training/pretrain_v3.py). Ignored by the T1 pipeline."""
    objective: str = "rtd"              # "rtd" (ELECTRA/DeBERTa-v3 replaced-token detection) | "mlm"
    seq_len: int = 512                  # pretraining rows; decisions/RLCD use data.max_seq_len
    mask_prob: float = 0.15             # rtd: generator masking rate; mlm: use ~0.3
    gen_layers: int = 6                 # rtd generator depth (same width; ~1/4 of the discriminator)
    rtd_weight: float = 50.0            # ELECTRA's discriminator loss weight
    sources: list = field(default_factory=list)  # [{"path": bin, "weight": w, "name": str}]
    val_holdout_tokens: int = 2_000_000 # tail of every source reserved for validation
    save_minutes: float = 20.0          # wall-clock checkpoint cadence (atomic, resumable)
    ext_seq_len: int = 2048             # context-extension stage
    ext_rope_base: float = 80000.0
    ext_micro_rows: int = 8


@dataclass
class VerqenConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "VerqenConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VerqenConfig":
        config = cls()
        for section in ("model", "training", "loss", "data", "logging", "pretrain"):
            target = getattr(config, section)
            for k, v in (raw.get(section) or {}).items():
                if hasattr(target, k):
                    if k == "betas" and isinstance(v, list):
                        v = tuple(v)
                    setattr(target, k, v)
        return config

    def to_dict(self) -> dict:
        def _clean(obj):
            if dataclasses.is_dataclass(obj):
                return {f.name: _clean(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            if isinstance(obj, (list, tuple)):
                return [_clean(x) for x in obj]
            return obj
        return _clean(self)

    def save_yaml(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
