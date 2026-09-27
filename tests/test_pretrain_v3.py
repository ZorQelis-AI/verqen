"""T2 pretraining stage: objectives, export round-trip, resume, GDES semantics (CPU, tiny model)."""

import json
import os

import numpy as np
import pytest
import torch

from verqen.config import VerqenConfig
from verqen.training import distributed as D
from verqen.training.pretrain_v3 import GDESEmbedding, RTDCore, clean_state, run_pretrain_v3
from verqen.utils.checkpoint import load_model

TOK = "data/tokenizer/verqen_32k.json"


def tiny_cfg(tmp_path, objective: str) -> VerqenConfig:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(2):
        p = tmp_path / f"src{i}.bin"
        rng.integers(20, 30000, size=40_000, dtype=np.uint16).tofile(p)
        paths.append({"path": str(p), "weight": 1 + i, "name": f"s{i}"})
    cfg = VerqenConfig.from_dict({
        "model": {"hidden_size": 64, "num_layers": 2, "num_heads": 2, "head_dim": 32, "ffn_size": 128, "set_layers": 1},
        "training": {"batch_rows": 4, "micro_rows": 2, "max_steps": 6, "warmup_steps": 2, "eval_every": 3,
                     "compile": False, "learning_rate": 1e-3},
        "data": {"tokenizer_path": TOK, "max_seq_len": 64},
        "logging": {"log_every": 2},
        "pretrain": {"objective": objective, "seq_len": 64, "gen_layers": 1, "sources": paths, "val_holdout_tokens": 5000,
                     "save_minutes": 0.0},
    })
    return cfg


@pytest.mark.parametrize("objective", ["rtd", "mlm"])
def test_pretrain_runs_exports_and_reloads_strictly(tmp_path, objective):
    cfg = tiny_cfg(tmp_path, objective)
    info = D.DistInfo()
    out = run_pretrain_v3(cfg, str(tmp_path / "run"), budget_hours=0.0, info=info, max_steps=6)
    m = load_model(out)  # raises on any key mismatch
    assert m.config.num_layers == 2
    metrics = [json.loads(l) for l in open(tmp_path / "run" / "metrics.jsonl")]
    key = "rtd_loss" if objective == "rtd" else "mlm_loss"
    assert any(key in r for r in metrics) and all(np.isfinite(r[key]) for r in metrics if key in r)
    assert any(r["stage"] == "pretrain_eval" for r in metrics)


def test_crash_resume_continues_the_same_plan(tmp_path, monkeypatch):
    """A crash at step 4 (after a save at step 3) resumes at 3 and finishes the original 6-step plan."""
    import verqen.training.common as common
    cfg = tiny_cfg(tmp_path, "rtd")
    info = D.DistInfo()
    real_log = common.JsonlLogger.log

    def crashing_log(self, **kw):
        if kw.get("stage") == "pretrain" and kw.get("step") == 4:
            raise RuntimeError("simulated crash")
        return real_log(self, **kw)

    monkeypatch.setattr(common.JsonlLogger, "log", crashing_log)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pretrain_v3(cfg, str(tmp_path / "run"), 0.0, info, max_steps=6)
    assert torch.load(tmp_path / "run" / "resume" / "resume.pt", weights_only=False)["step"] == 3
    assert not (tmp_path / "run" / "final").exists()
    monkeypatch.setattr(common.JsonlLogger, "log", real_log)
    run_pretrain_v3(cfg, str(tmp_path / "run"), 0.0, info, max_steps=6)
    assert torch.load(tmp_path / "run" / "resume" / "resume.pt", weights_only=False)["step"] == 6
    steps = [json.loads(l)["step"] for l in open(tmp_path / "run" / "metrics.jsonl") if '"pretrain"' in l]
    assert steps[-1] == 6


def test_finished_run_is_idempotent(tmp_path):
    cfg = tiny_cfg(tmp_path, "mlm")
    info = D.DistInfo()
    run_pretrain_v3(cfg, str(tmp_path / "run"), 0.0, info, max_steps=3)
    run_pretrain_v3(cfg, str(tmp_path / "run"), 0.0, info, max_steps=3)  # nothing left to do, no error
    assert torch.load(tmp_path / "run" / "resume" / "resume.pt", weights_only=False)["step"] == 3


def test_context_extension_initializes_from_previous_stage(tmp_path):
    cfg = tiny_cfg(tmp_path, "rtd")
    info = D.DistInfo()
    run_pretrain_v3(cfg, str(tmp_path / "pre"), 0.0, info, max_steps=3)
    cfg.model.rope_base = 80000.0
    out = run_pretrain_v3(cfg, str(tmp_path / "ext"), 0.0, info, stage="pretrain_ext",
                          init_core=str(tmp_path / "pre" / "resume"), seq_len=128, micro_rows=1, max_steps=2)
    m = load_model(out)
    assert m.config.rope_base == 80000.0 and m.config.max_seq_len >= 128


def test_gdes_blocks_discriminator_gradient_into_generator_embedding():
    src = torch.nn.Embedding(10, 4)
    g = GDESEmbedding(src)
    out = g(torch.tensor([[1, 2, 3]]))
    out.sum().backward()
    assert src.weight.grad is None, "discriminator loss must not reach the generator embedding"
    assert g.delta.weight.grad is not None
    assert torch.allclose(g.baked(), src.weight.detach())  # delta starts at zero


def test_clean_state_strips_compile_and_ddp_prefixes():
    s = clean_state({"module.disc._orig_mod.layers.0.w": 1, "gen.transformer._orig_mod.x": 2})
    assert set(s) == {"disc.layers.0.w", "gen.transformer.x"}
