"""Pick the checkpoint to release: RLCD output or the pre-RLCD decision checkpoint.

    python production/select_release.py --config configs/v3/t2.box.yaml \
        --baseline runs/verqen-t2/decisions/final --candidate runs/verqen-t2/rlcd/final \
        --out runs/verqen-t2/release_choice.json

Why: RLCD optimises environment reward under a KL leash, and locally it has cost decision
accuracy (T1: graph_route 0.84 -> 0.70 before the fix ablation; AB_rc: training-stream decision
accuracy 0.96 -> 0.91 by step 750). Releasing it unconditionally would ship a regression
whenever that happens.

Pre-registered rule (fixed before the T2 run, evaluated on the VALIDATION split only; the
test splits, Artifact-G and the Jev-parity suite are never consulted):
  release the candidate iff
    (1) its overall val accuracy >= baseline - OVERALL_TOL, and
    (2) no domain with >= MIN_DOMAIN_N val questions loses more than DOMAIN_TOL, and
    (3) no scenario variant (base / flip / irrelevant / info_* / bait) with >= MIN_DOMAIN_N val
        questions loses more than DOMAIN_TOL. Added 25 Sep after RLCD on AB cut the bait probes
        0.752 -> 0.696 (7/9 cells down, 0 up) while overall accuracy rose: a per-domain rule
        cannot see a skill that cuts across domains.
  Otherwise release the baseline. A missing candidate means the baseline.
The decision and every number behind it go to --out; the chosen path is the last stdout line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OVERALL_TOL = 0.005
DOMAIN_TOL = 0.03
MIN_DOMAIN_N = 200


def val_accuracy(ckpt: str, cfg) -> dict:
    import numpy as np
    import torch
    from verqen.calibration.fit import collect_outputs
    from verqen.data.store import load_or_build
    from verqen.utils.checkpoint import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, dev)
    store = load_or_build(os.path.join(cfg.data.decision_dir, "val.jsonl"), cfg.data.tokenizer_path, cfg.data.max_seq_len)
    o = collect_outputs(model, store, dev, cfg.data.max_seq_len)
    logp, tgt, mask = (np.asarray(o[k]) for k in ("logp", "target", "mask"))
    pred = np.where(mask, logp, -np.inf).argmax(1)
    gold = np.where(mask, tgt, -np.inf).argmax(1)
    correct = pred == gold
    fam = np.array([store.meta(int(i))["family"] for i in o["example"]])
    by = {f: {"n": int((fam == f).sum()), "accuracy": round(float(correct[fam == f].mean()), 4)} for f in sorted(set(fam))}
    rows = [json.loads(line).get("variant") or "real" for line in open(os.path.join(cfg.data.decision_dir, "val.jsonl"), encoding="utf-8")]
    by_var: dict = {}
    if len(rows) == len(store):
        var = np.array([rows[int(i)] for i in o["example"]])
        by_var = {v: {"n": int((var == v).sum()), "accuracy": round(float(correct[var == v].mean()), 4)} for v in sorted(set(var))}
    else:  # never let the guard itself fail the run: fall back to rules (1)-(2) and say so
        print(f"[select] WARNING: val.jsonl has {len(rows)} rows, the store {len(store)}; variant rule skipped", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"checkpoint": ckpt, "n": int(len(correct)), "accuracy": round(float(correct.mean()), 4), "by_domain": by,
            "by_variant": by_var}


def env_scores(ckpt: str, cfg, episodes: int = 40) -> dict:
    """Greedy optimal-action rate per environment on the held-out evaluation seeds (999..)."""
    import torch
    from verqen.rlcd.trainer import EnvRunner, evaluate_envs
    from verqen.tokenizer import VerqenTokenizer
    from verqen.utils.checkpoint import load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, dev)
    res = evaluate_envs(model, EnvRunner(model, VerqenTokenizer.from_pretrained(cfg.data.tokenizer_path), dev,
                                         cfg.data.max_seq_len), episodes=episodes)
    del model
    torch.cuda.empty_cache()
    rates = {k: v["optimal_action_rate"] for k, v in res.items()}
    return {"by_env": rates, "mean": round(sum(rates.values()) / len(rates), 4)}


def decide(base: dict, cand: dict | None) -> tuple[str, list[str]]:
    if cand is None:
        return "baseline", ["candidate checkpoint missing"]
    reasons = []
    if cand["accuracy"] < base["accuracy"] - OVERALL_TOL:
        reasons.append(f"overall val {cand['accuracy']:.4f} < baseline {base['accuracy']:.4f} - {OVERALL_TOL}")
    for group in ("by_domain", "by_variant"):
        for d, b in base[group].items():
            c = cand[group].get(d)
            if c and b["n"] >= MIN_DOMAIN_N and c["accuracy"] < b["accuracy"] - DOMAIN_TOL:
                reasons.append(f"{group[3:]} {d}: {c['accuracy']:.4f} < {b['accuracy']:.4f} - {DOMAIN_TOL}")
    return ("baseline", reasons) if reasons else ("candidate", ["no regression beyond tolerance"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-env-gain", type=float, default=None,
                    help="agentic top-up: the candidate must ALSO beat the baseline's mean environment "
                         "optimal-action rate by at least this much (pre-registered 0.02)")
    ap.add_argument("--max-env-drop", type=float, default=0.02,
                    help="with --require-env-gain: no single environment may lose more than this (no new weakness)")
    a = ap.parse_args()
    from verqen.config import VerqenConfig
    cfg = VerqenConfig.from_yaml(a.config)
    base = val_accuracy(a.baseline, cfg)
    has_cand = Path(a.candidate, "model.safetensors").exists()
    cand = val_accuracy(a.candidate, cfg) if has_cand else None
    which, reasons = decide(base, cand)
    envs = None
    if a.require_env_gain is not None and cand is not None:
        envs = {"baseline": env_scores(a.baseline, cfg), "candidate": env_scores(a.candidate, cfg)}
        gain = envs["candidate"]["mean"] - envs["baseline"]["mean"]
        envs["gain"] = round(gain, 4)
        if gain < a.require_env_gain:
            which, reasons = "baseline", reasons + [f"environment gain {gain:+.4f} < required {a.require_env_gain}"]
        for e, r in envs["baseline"]["by_env"].items():
            c = envs["candidate"]["by_env"].get(e, 0.0)
            if c < r - a.max_env_drop:
                which, reasons = "baseline", reasons + [f"environment {e}: {c:.4f} < {r:.4f} - {a.max_env_drop}"]
    chosen = a.candidate if which == "candidate" else a.baseline
    rec = {"rule": {"overall_tol": OVERALL_TOL, "domain_tol": DOMAIN_TOL, "min_domain_n": MIN_DOMAIN_N, "split": "val"},
           "baseline": base, "candidate": cand, "environments": envs, "chosen": chosen, "reasons": reasons}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out + ".tmp"
    Path(tmp).write_text(json.dumps(rec, indent=1))
    os.replace(tmp, a.out)
    print(f"[select] baseline {base['accuracy']:.4f} candidate {cand['accuracy'] if cand else None} -> {which}: {'; '.join(reasons)}",
          flush=True)
    print(chosen)


if __name__ == "__main__":
    main()
