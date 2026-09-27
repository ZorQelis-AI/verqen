"""T2 decision stage: rank sharding of the batch stream, crash-resume, final checkpoint (CPU, tiny)."""

import json
import os

import pytest
import torch

from verqen.config import VerqenConfig
from verqen.training import distributed as D
from verqen.training.decisions_v3 import run_decisions_v3, sharded
from verqen.training.pretrain_v3 import run_pretrain_v3
from verqen.utils.checkpoint import load_model

SMOKE = "data/decision/v3_smoke"


def test_sharding_partitions_the_global_stream_and_resumes_exactly():
    stream = lambda: iter([[i] for i in range(40)])
    r0, r1 = list(sharded(stream(), 0, 2)), list(sharded(stream(), 1, 2))
    assert sorted(x[0] for x in r0 + r1) == list(range(40)) and not set(map(tuple, r0)) & set(map(tuple, r1))
    # resuming after 10 consumed global batches continues exactly where the ranks left off
    assert list(sharded(stream(), 0, 2, skip=10)) == r0[5:]
    assert list(sharded(stream(), 1, 2, skip=10)) == r1[5:]


@pytest.fixture(scope="module")
def pretrained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pre")
    cfg = VerqenConfig.from_yaml("configs/v3/smoke.yaml")
    cfg.training.compile = False
    return run_pretrain_v3(cfg, str(tmp / "pre"), 0.0, D.DistInfo(), max_steps=2)


@pytest.mark.skipif(not os.path.exists(SMOKE), reason="smoke decision data not built")
def test_decisions_crash_resume(tmp_path, pretrained, monkeypatch):
    import verqen.training.common as common
    cfg = VerqenConfig.from_yaml("configs/v3/smoke.yaml")
    cfg.training.compile = False
    cfg.training.eval_every = 100
    real_log = common.JsonlLogger.log

    def crashing_log(self, **kw):
        if kw.get("stage") == "decisions" and kw.get("step") == 10:
            raise RuntimeError("simulated crash")
        return real_log(self, **kw)

    monkeypatch.setattr(common.JsonlLogger, "log", crashing_log)
    with pytest.raises(RuntimeError):
        run_decisions_v3(cfg, pretrained, str(tmp_path / "dec"), 0.0, D.DistInfo(), max_steps=12, save_minutes=0.0)
    ck = torch.load(tmp_path / "dec" / "resume" / "resume.pt", weights_only=False)
    assert ck["step"] == 9 and ck["consumed"] == 9 * (cfg.training.batch_rows // cfg.training.micro_rows)
    monkeypatch.setattr(common.JsonlLogger, "log", real_log)
    out = run_decisions_v3(cfg, pretrained, str(tmp_path / "dec"), 0.0, D.DistInfo(), max_steps=12, save_minutes=0.0)
    assert torch.load(tmp_path / "dec" / "resume" / "resume.pt", weights_only=False)["step"] == 12
    load_model(out)
    sel = json.load(open(tmp_path / "dec" / "selection.json"))
    assert sel["final_val_acc"] is not None
