"""T2 RLCD stage: RLCD from the decision checkpoint, resumable, single GPU (rank 0).

RLCD is kept single-GPU on purpose: its environment rollouts are sequential Python and it
runs ~2 h, so data parallelism buys little, while GPU 1 is free to evaluate the stage-2
checkpoint at the same time (production/run_all.sh).
"""

from __future__ import annotations

import os

from verqen.config import VerqenConfig
from verqen.data.store import load_or_build
from verqen.rlcd.trainer import RLCDConfig, run_rlcd
from verqen.tokenizer import VerqenTokenizer
from verqen.training import distributed as D
from verqen.utils.checkpoint import load_model


def run_rlcd_v3(cfg: VerqenConfig, init: str, out_dir: str, budget_hours: float, info: D.DistInfo,
                steps: int = 100000) -> str:
    if not info.is_main:
        return os.path.join(out_dir, "final")
    train = load_or_build(os.path.join(cfg.data.decision_dir, "train.jsonl"), cfg.data.tokenizer_path, cfg.data.max_seq_len)
    val_path = os.path.join(cfg.data.decision_dir, "val.jsonl")
    val = load_or_build(val_path, cfg.data.tokenizer_path, cfg.data.max_seq_len) if os.path.exists(val_path) else None
    model, ref = load_model(init, info.device), load_model(init, info.device)
    # RLCD holds a second (reference) model and runs the environment gradient on top of the decision
    # step, so it needs far less per forward than the decisions stage: 16 rows at 1024 tokens OOM'd a
    # 5090 on 25 Sep 2026 (31.30 of 31.36 GiB) 30 s into the stage. A third of the decisions micro size
    # for decision rows, half of that per environment chunk: same updates, smaller pieces.
    rows = max(4, min(8, (cfg.training.decision_micro_rows or 24) // 3))
    rc = RLCDConfig(budget_hours=budget_hours, steps=steps, decision_rows=rows, env_chunk_rows=max(2, rows // 2))
    return run_rlcd(model, ref, train, VerqenTokenizer.from_pretrained(cfg.data.tokenizer_path), out_dir, info.device, rc,
                    cfg.data.tokenizer_path, cfg.data.max_seq_len, resume_dir=os.path.join(out_dir, "resume"),
                    save_minutes=cfg.pretrain.save_minutes, val=val)
