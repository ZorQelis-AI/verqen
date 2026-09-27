"""Diagnose RLCD's decision-vs-environment gradient balance (no training; the model is not updated).

    python scripts/audit/rlcd_grad_balance.py --checkpoint runs/fix/AB/decisions/best --steps 12

OBSERVED (AB_rc, 24 Sep): the decision objective's own expected reward fell 0.92 -> 0.83 and
training-stream decision accuracy 0.96 -> 0.90 while environment return rose. Hypothesis: the
two losses are summed and clipped jointly at norm 1.0, so a larger environment policy-gradient
sets both the direction and the clip scale. This measures, per step, on identical parameters:
  |g_dec|, |g_env|, cos(g_dec, g_env), and how much of the clipped joint step is decision gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402

from verqen.config import VerqenConfig  # noqa: E402
from verqen.data.collator import collate_rows, pack_rows, to_device  # noqa: E402
from verqen.training.decisions import token_budget_batches  # noqa: E402
from verqen.data.store import load_or_build  # noqa: E402
from verqen.rlcd.trainer import (EnvRunner, RLCDConfig, _forward, decision_rlcd_loss,  # noqa: E402
                                 env_rlcd_backward)
from verqen.tokenizer import VerqenTokenizer  # noqa: E402
from verqen.training.common import autocast  # noqa: E402
from verqen.utils.checkpoint import load_model  # noqa: E402


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2/t1.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="runs/audit/rlcd_grad_balance.json")
    a = ap.parse_args()
    cfg = VerqenConfig.from_yaml(a.config)
    dev = torch.device("cuda")
    model, ref = load_model(a.checkpoint, dev), load_model(a.checkpoint, dev)
    ref.eval()
    train = load_or_build(os.path.join(cfg.data.decision_dir, "train.jsonl"), cfg.data.tokenizer_path, cfg.data.max_seq_len)
    rc = RLCDConfig()
    runner = EnvRunner(model, VerqenTokenizer.from_pretrained(cfg.data.tokenizer_path), dev, cfg.data.max_seq_len)
    params = [p for n, p in model.named_parameters() if not n.startswith("mlm_head")]
    batches = iter(token_budget_batches(train, rc.decision_rows, cfg.data.max_seq_len, seed=77))
    rng = random.Random(1234)
    rows = []
    for step in range(a.steps):
        model.train()
        idx = next(batches)
        packed, _ = pack_rows([train[i] for i in idx], cfg.data.max_seq_len, rc.decision_rows)
        b = to_device(collate_rows(packed, cfg.data.max_seq_len, 0, rc.decision_rows), dev)
        model.zero_grad(set_to_none=True)
        with autocast(dev):
            out = _forward(model, b)
            with torch.no_grad():
                r = _forward(ref, b)
        loss, _ = decision_rlcd_loss(out, r["logp"], b["target"], out["opt_mask"], rc)
        loss.backward()
        gd = flat_grad(params)
        model.zero_grad(set_to_none=True)
        em = env_rlcd_backward(model, ref, runner, rc, rng, dev)
        ge = flat_grad(params)
        model.zero_grad(set_to_none=True)
        nd, ne = gd.norm().item(), ge.norm().item()
        joint = gd + ge
        nj = joint.norm().item()
        scale = min(1.0, 1.0 / max(nj, 1e-12))  # clip_grad_norm_(.., 1.0)
        rows.append({"step": step, "g_dec": round(nd, 4), "g_env": round(ne, 4), "ratio_env_over_dec": round(ne / max(nd, 1e-12), 3),
                     "cos": round(torch.nn.functional.cosine_similarity(gd, ge, dim=0).item(), 4),
                     "dec_share_of_joint": round((gd @ joint).item() / max(nj ** 2, 1e-12), 4),
                     "clip_scale": round(scale, 4), "env_transitions": em.get("env_transitions")})
        print(json.dumps(rows[-1]), flush=True)
    import statistics as st
    summ = {k: round(st.median(r[k] for r in rows), 4) for k in ("g_dec", "g_env", "ratio_env_over_dec", "cos", "dec_share_of_joint", "clip_scale")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"checkpoint": a.checkpoint, "median": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print("MEDIAN", json.dumps(summ))


if __name__ == "__main__":
    main()
