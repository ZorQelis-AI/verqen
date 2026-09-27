"""Reproduce the Artifact-G benchmark run for VerQen-T1 and store per-case artifacts.

Uses the benchmark's own harness functions (unchanged) for decisions and metrics, and records
for every case: raw (uncalibrated) probabilities, calibrated probabilities, p_correct,
escalation flag, decision, and the exact configuration. Output: runs/audit/<tag>/.

    python scripts/audit/repro_benchmark.py --checkpoint runs/verqen-t1-v2/release --tag baseline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from verqen.inference.engine import VerqenInferenceEngine  # noqa: E402
from verqen_benchmark.harness.metrics import compute_comprehensive_metrics  # noqa: E402

BENCH = "verqen_benchmark/dataset/benchmark_cases.jsonl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def run(engine, cases, drop_state=False, cost_aware=False):
    raw_engine_cal = engine.calibration
    out = []
    for c in cases:
        q = c["input"]["questions"][0]
        qtype = q.get("type", "choice").lower()
        opts = q.get("options", []) if qtype != "noul" else ["no", "yes"]
        st = "" if drop_state else c["input"]["state"]
        req = {"type": qtype, "text": q["text"], "options": opts}
        if cost_aware:  # opt-in configuration: the case's declared decision costs are passed as an input
            dc = c["decision_cost"]
            req["cost"] = {"error_cost": dc["error_cost"], "escalation_cost": dc["escalation_cost"]}
        t0 = time.perf_counter()
        res = engine.decide_multi(st, [req])[0]
        lat = (time.perf_counter() - t0) * 1000
        engine.calibration = None
        raw = engine.decide_multi(st, [req])[0]
        engine.calibration = raw_engine_cal
        sel = ("yes" if res.get("value") else "no") if qtype == "noul" else res["selected"]
        out.append({
            "case_id": c["case_id"], "case_family": c["case_family"], "domain": c["domain"], "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"], "selected": sel, "is_correct": sel == c["ground_truth"],
            "probabilities": res["probabilities"], "raw_probabilities": raw["probabilities"],
            "confidence": res["confidence"], "p_correct": res["p_correct"], "should_escalate": res["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False), "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}), "latency_ms": round(lat, 2), "split": c.get("split"),
            "options": opts,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/verqen-t1-v2/release")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--risk-mode", default="marginal", choices=["marginal", "worst_group"])
    ap.add_argument("--cost-aware", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    out_dir = Path("runs/audit") / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = [json.loads(l) for l in open(a.bench, encoding="utf-8") if l.strip()]
    engine = VerqenInferenceEngine.from_checkpoint(a.checkpoint, "auto", risk_mode=a.risk_mode, fast=a.fast)
    config = {
        "checkpoint": a.checkpoint, "checkpoint_sha": sha(Path(a.checkpoint) / "model.safetensors"),
        "calibration_sha": sha(Path(a.checkpoint) / "calibration.json"), "benchmark": a.bench, "benchmark_sha": sha(a.bench),
        "n_cases": len(cases), "device": str(engine.device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "weights_dtype": "float32", "autocast_dtype": str(engine.dtype), "batch_size": 1, "risk": engine.risk,
        "torch": torch.__version__, "python": platform.python_version(), "seed": 0, "engine_fast_path": engine.fast,
        "risk_mode": a.risk_mode, "cost_aware": a.cost_aware,
    }
    preds = run(engine, cases, cost_aware=a.cost_aware)
    nostate = run(engine, cases, drop_state=True, cost_aware=a.cost_aware)
    json.dump(config, open(out_dir / "config.json", "w"), indent=2)
    json.dump(preds, open(out_dir / "predictions.json", "w"), indent=1)
    json.dump(nostate, open(out_dir / "predictions_nostate.json", "w"), indent=1)
    m, mn = compute_comprehensive_metrics(preds), compute_comprehensive_metrics(nostate)
    json.dump({"full": m, "no_state": mn}, open(out_dir / "harness_metrics.json", "w"), indent=2)
    print(json.dumps({"overall": m["overall_accuracy"], "ece": m["calibration"], "selective": m["selective_abstention"],
                      "invariants": m["behavioral_invariants"], "no_state": mn["overall_accuracy"],
                      "families": {k: v["accuracy"] for k, v in m["family_breakdown"].items()}}, indent=1))


if __name__ == "__main__":
    main()
