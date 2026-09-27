"""Run the Jev-parity suite against one system and score it per capability and domain.

    python evals/jev_parity/run.py --checkpoint runs/verqen-t2/release --out results/verqen.json
    python evals/jev_parity/run.py --adapter jev --out results/jev.json          # needs JEV_* env (see adapters.py)
    python evals/jev_parity/run.py --adapter anthropic --model claude-sonnet-5 --out results/sonnet.json

Metrics per capability (and overall): accuracy with a Wilson 95% CI, top-label ECE of the
system's confidence, AURC (area under the risk-coverage curve; lower is better; it measures
whether confidence is worth routing on), p50/p95 latency and the cost per 1M decisions at
the adapter's declared price. `compare.py` turns two result files into a paired comparison.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import make_adapter  # noqa: E402


GPU_PRICE_H = 0.80  # $/h for one inference GPU (L4-class cloud price), applied to every local model alike


def _gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        return "unknown"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def ece(conf, correct, bins=10):
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            e += len(idx) / len(conf) * abs(sum(conf[i] for i in idx) / len(idx) - sum(correct[i] for i in idx) / len(idx))
    return round(e, 4)


def aurc(conf, correct):
    order = sorted(range(len(conf)), key=lambda i: -conf[i])
    err, risks = 0, []
    for k, i in enumerate(order, 1):
        err += 1 - correct[i]
        risks.append(err / k)
    return round(sum(risks) / len(risks), 4) if risks else None


def summarize(rows):
    k = sum(r["correct"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    conf = [r["confidence"] for r in rows]
    corr = [r["correct"] for r in rows]
    return {"n": len(rows), "accuracy": round(k / max(1, len(rows)), 4), "ci95": wilson(k, len(rows)),
            "ece": ece(conf, corr), "aurc": aurc(conf, corr),
            "latency_p50_ms": round(statistics.median(lat), 2) if lat else None,
            "latency_p95_ms": round(sorted(lat)[int(0.95 * (len(lat) - 1))], 2) if lat else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(Path(__file__).parent / "cases.jsonl"))
    ap.add_argument("--adapter", default="verqen")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if not Path(a.cases).exists():
        import subprocess
        subprocess.run([sys.executable, str(Path(__file__).parent / "build.py"), "--out", a.cases], check=True)
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")][: a.limit]
    ad = make_adapter(a.adapter, checkpoint=a.checkpoint, model=a.model)
    rows = []
    for c in cases:
        t = time.perf_counter()
        sel, conf, meta = ad.decide(c["state"], c["question"], c["options"])
        ms = (time.perf_counter() - t) * 1000
        rows.append({"id": c["id"], "capability": c["capability"], "domain": c["domain"], "selected": sel,
                     "confidence": float(conf), "correct": int(sel == c["answer"]), "latency_ms": ms,
                     "input_chars": len(c["state"]) + len(c["question"]) + sum(map(len, c["options"])), **meta})
    by_cap = collections.defaultdict(list)
    by_dom = collections.defaultdict(list)
    for r in rows:
        by_cap[r["capability"]].append(r)
        by_dom[r["domain"]].append(r)
    chars = sum(r["input_chars"] for r in rows) / max(1, len(rows))
    # Local models (VerQen, Laya) are priced on the SAME basis: one GPU at GPU_PRICE_H $/h, one
    # decision at a time, at the measured mean latency. API models keep their published prices.
    if getattr(ad, "local_gpu", False):
        mean_s = statistics.fmean(r["latency_ms"] for r in rows) / 1000 if rows else 0.0
        cost = round(GPU_PRICE_H / 3600 * mean_s * 1e6, 2)
    else:
        cost = ad.cost_per_million(chars)
    res = {"system": ad.name, "n": len(rows), "overall": summarize(rows),
           "by_capability": {k: summarize(v) for k, v in sorted(by_cap.items())},
           "by_domain": {k: summarize(v) for k, v in sorted(by_dom.items())},
           "cost_per_million_decisions_usd": cost, "gpu": _gpu_name(), "invalid": sum(1 for r in rows if r.get("invalid")),
           "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "by_domain")}, indent=1))


if __name__ == "__main__":
    main()
