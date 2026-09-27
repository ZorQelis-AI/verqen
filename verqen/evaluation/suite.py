"""Stage 5: evaluation that can't be fooled the way v1's was.

Reports, per held-out split (official benchmark validation/test splits never trained on,
procedural IID / unseen-phrasing / prompt-injected tests):

* accuracy with a 95% Wilson interval, chance level, NLL, Brier, ECE (raw and calibrated)
* selective accuracy and coverage at the risk-controlled operating point (escalate the rest)
* conformal prediction-set coverage and average size

Plus:
* shortcut probes (model-based): state removed, and state + question removed. On data
  where the answer depends on the state these must fall toward chance.
* invariance checks: option reversal and prepending an unrelated question (should be ~0)
* environment rollouts (greedy): oracle-optimal action rate and mean return
* speed: single-request latency, batched throughput, cached-state throughput, and a
  cost model with every price stated as an explicit, editable assumption.
"""

from __future__ import annotations

import glob
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

from verqen.calibration.fit import TYPES, _softmax_T, collect_outputs
from verqen.calibration.metrics import expected_calibration_error, wilson_interval
from verqen.data.store import build_store, read_jsonl
from verqen.utils.checkpoint import load_calibration, load_model

# Cost assumptions (edit freely; they only feed the cost table).
ASSUMPTIONS = {
    "local_gpu": "RTX 5060 Ti 16GB",
    "local_gpu_usd_per_hour": 0.12,          # ~180 W at ~$0.25/kWh plus amortized hardware
    "cloud_gpu_usd_per_hour": 0.80,          # typical on-demand L4-class price
    "frontier_input_usd_per_mtok": 3.00,
    "frontier_output_usd_per_mtok": 15.00,
    "frontier_tokens_per_decision_in": 400,  # prompt with state, question and options
    "frontier_tokens_per_decision_out": 20,  # short structured answer
    "frontier_latency_s": 1.0,               # typical time-to-complete for a short API call (range 0.5-2 s)
}


def _probs(out, calib):
    p = np.zeros_like(out["target"])
    for t_id, name in enumerate(TYPES):
        sel = out["qtype"] == t_id
        if sel.any():
            T = calib["temperature"][name] if calib else 1.0
            p[sel] = _softmax_T(out["logp"][sel], out["mask"][sel], T)
    return p


def split_metrics(out: dict, calib: dict | None) -> dict[str, Any]:
    probs = _probs(out, calib)
    t, m = out["target"], out["mask"]
    pred = np.where(m, probs, -1).argmax(-1)
    best = np.where(m, t, -1).max(-1)
    correct = (t[np.arange(len(pred)), pred] >= best - 1e-6)
    n, k = len(correct), int(correct.sum())
    lo, hi = wilson_interval(k, n)
    chance = float(np.mean([(t[i][m[i]] >= best[i] - 1e-6).sum() / m[i].sum() for i in range(n)]))
    nll = float(-(t * np.log(np.clip(probs, 1e-12, 1))).sum(-1).mean())
    brier = float((((probs - t) ** 2) * m).sum(-1).mean())
    res = {"n": n, "accuracy": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)], "chance": round(chance, 4),
           "nll": round(nll, 4), "brier": round(brier, 4),
           "ece_raw": round(expected_calibration_error(np.where(m, np.exp(out["logp"]), 0), t, m), 4),
           "ece_calibrated": round(expected_calibration_error(probs, t, m), 4)}
    if calib:
        a, b = calib["selective_platt"]
        pc = 1 / (1 + np.exp(-(a * out["sel"] + b)))
        thr = calib["risk_control"]["by_risk"][calib["default_risk"]]["threshold"]
        keep = pc >= thr
        res["selective"] = {"target_risk": float(calib["default_risk"]), "coverage": round(float(keep.mean()), 4),
                            "accuracy_on_kept": round(float(correct[keep].mean()), 4) if keep.any() else None}
        qhat = calib["conformal"]["qhat"]
        sets = (probs >= 1 - qhat) & m
        y = t.argmax(-1)
        res["conformal"] = {"alpha": calib["conformal"]["alpha"], "coverage": round(float(sets[np.arange(n), y].mean()), 4),
                            "avg_set_size": round(float(sets.sum(-1).mean()), 3)}
    return res


def _ablate(rows, drop_state=False, drop_question=False):
    out = []
    for r in rows:
        r = dict(r)
        if drop_state:
            r["state"] = ""
        if drop_question:
            r["questions"] = [dict(q, text="") for q in r["questions"]]
        out.append(r)
    return out


def invariance(engine, rows, n=150, seed=0) -> dict[str, float]:
    rng = random.Random(seed)
    rows = [r for r in rows if r["questions"][0]["type"] != "score" and len(r["questions"][0]["options"]) >= 3]
    rows = rng.sample(rows, min(n, len(rows)))
    tv_perm, tv_q = [], []
    extra = {"type": "choice", "text": "Unrelated: which color is the sky?", "options": ["blue", "green", "red"]}
    for r in rows:
        q = r["questions"][0]
        base = engine.decide_multi(r["state"], [q])[0]["probabilities"]
        rev = engine.decide_multi(r["state"], [dict(q, options=list(reversed(q["options"])))])[0]["probabilities"]
        pre = engine.decide_multi(r["state"], [extra, q])[1]["probabilities"]
        tv_perm.append(0.5 * sum(abs(base[o] - rev[o]) for o in base))
        tv_q.append(0.5 * sum(abs(base[o] - pre[o]) for o in base))
    return {"n": len(rows), "option_reversal_tv_mean": float(np.mean(tv_perm)), "option_reversal_tv_max": float(np.max(tv_perm)),
            "prepend_question_tv_mean": float(np.mean(tv_q)), "prepend_question_tv_max": float(np.max(tv_q))}


def speed(engine, rows) -> dict[str, Any]:
    reqs = [{"state": r["state"], "questions": [{k: q[k] for k in ("type", "text", "options")} for q in r["questions"][:1]]}
            for r in rows[:2048]]
    if getattr(engine, "fast", False):
        engine.warmup()
        for r in reqs[:400]:  # compile the shape buckets real traffic hits
            engine.decide_many([r])
    dev = engine.device
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for _ in range(5):
        engine.decide_many(reqs[:1])
    sync()
    lat = []
    for r in reqs[:300]:
        sync()
        t = time.perf_counter()
        engine.decide_many([r])
        sync()
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    single_ms = lat[len(lat) // 2]
    engine.decide_many(reqs)
    sync()
    t = time.perf_counter()
    reps = 3
    for _ in range(reps):
        engine.decide_many(reqs)
    sync()
    thr = reps * len(reqs) / (time.perf_counter() - t)
    # cached state: one document, 50 questions asked one at a time
    h = engine.encode_state(reqs[0]["state"])
    q = reqs[0]["questions"]
    engine.decide_on_state(h, q)
    sync()
    t = time.perf_counter()
    for _ in range(50):
        engine.decide_on_state(h, q)
    sync()
    cached_ms = (time.perf_counter() - t) / 50 * 1000
    A = ASSUMPTIONS
    usd_per_m_local = A["local_gpu_usd_per_hour"] / (thr * 3600) * 1e6
    usd_per_m_cloud = A["cloud_gpu_usd_per_hour"] / (thr * 3600) * 1e6
    usd_per_m_frontier = (A["frontier_tokens_per_decision_in"] * A["frontier_input_usd_per_mtok"]
                          + A["frontier_tokens_per_decision_out"] * A["frontier_output_usd_per_mtok"])
    return {
        "single_request_latency_ms": round(single_ms, 2),
        "single_request_latency_p90_ms": round(lat[int(len(lat) * 0.9)], 2),
        "cached_state_question_latency_ms": round(cached_ms, 2),
        "batched_decisions_per_s": round(thr, 1),
        "usd_per_million_decisions": {"local_gpu": round(usd_per_m_local, 3), "cloud_gpu": round(usd_per_m_cloud, 3),
                                      "frontier_api": round(usd_per_m_frontier, 2)},
        "cost_ratio_vs_frontier": {"local_gpu": round(usd_per_m_frontier / usd_per_m_local, 1),
                                   "cloud_gpu": round(usd_per_m_frontier / usd_per_m_cloud, 1)},
        "latency_ratio_vs_frontier_single": round(A["frontier_latency_s"] * 1000 / single_ms, 1),
        "latency_ratio_range_0.5s_to_2s": [round(500 / single_ms, 1), round(2000 / single_ms, 1)],
        "assumptions": A,
    }


def run_suite(checkpoint: str, cfg, device, out_dir: str, max_per_split: int = 3000) -> dict[str, Any]:
    from verqen.inference.engine import VerqenInferenceEngine
    from verqen.rlcd.trainer import EnvRunner, evaluate_envs
    from verqen.tokenizer import VerqenTokenizer

    os.makedirs(out_dir, exist_ok=True)
    model = load_model(checkpoint, device)
    calib = load_calibration(checkpoint)
    tok_path = os.path.join(checkpoint, "tokenizer.json")
    report: dict[str, Any] = {"checkpoint": checkpoint, "parameters": model.count_parameters(), "splits": {}}
    eval_files = sorted(glob.glob(os.path.join(cfg.data.decision_dir, "eval", "*.jsonl")))
    for f in eval_files:
        name = os.path.basename(f)[:-6]
        rows = read_jsonl(f)
        random.Random(0).shuffle(rows)
        rows = rows[:max_per_split]
        store = build_store(rows, tok_path, cfg.data.max_seq_len, workers=1)
        out = collect_outputs(model, store, device, cfg.data.max_seq_len)
        report["splits"][name] = split_metrics(out, calib)
        if name.startswith("proc_") or name in ("boolq", "race", "sciq", "mnli", "hellaswag"):
            probes = {}
            for label, kw in (("no_state", {"drop_state": True}), ("options_only", {"drop_state": True, "drop_question": True})):
                st = build_store(_ablate(rows[:1500], **kw), tok_path, cfg.data.max_seq_len, workers=1)
                probes[label] = split_metrics(collect_outputs(model, st, device, cfg.data.max_seq_len), calib)["accuracy"]
            report["splits"][name]["shortcut_probes"] = probes
        print(f"[eval] {name}: {json.dumps(report['splits'][name])}", flush=True)

    engine = VerqenInferenceEngine(model, VerqenTokenizer.from_pretrained(tok_path), device, cfg.data.max_seq_len, calib)
    # T1 data names its procedural split proc_test; v3 data has proc_v3_test / proc_v3_unseen / proc_v2_test
    proc_name = next(n for n in ("proc_test", "proc_v3_test", "proc_v2_test")
                     if os.path.exists(os.path.join(cfg.data.decision_dir, "eval", f"{n}.jsonl")))
    proc = read_jsonl(os.path.join(cfg.data.decision_dir, "eval", f"{proc_name}.jsonl"))
    report["invariance"] = invariance(engine, proc)
    report["environments"] = evaluate_envs(model, EnvRunner(model, engine.tokenizer, device, cfg.data.max_seq_len), episodes=60)
    fast_engine = VerqenInferenceEngine(load_model(checkpoint, device), engine.tokenizer, device, cfg.data.max_seq_len,
                                        calib, fast=device.type == "cuda")
    report["speed"] = speed(fast_engine, proc)
    real = [v["accuracy"] for k, v in report["splits"].items() if not k.startswith("proc_")]
    report["summary"] = {"mean_accuracy_real_benchmarks": round(float(np.mean(real)), 4) if real else None,
                         "calibration_report": (calib or {}).get("calibration_report")}
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("invariance", "environments", "speed", "summary")}, indent=2))
    return os.path.join(out_dir, "report.json")
