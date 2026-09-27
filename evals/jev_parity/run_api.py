"""Run a suite against an API adapter with parallel requests; same output format as run.py.

    python evals/jev_parity/run_api.py --adapter jev_openrouter --cases evals/laya_bench/cases.jsonl \
        --out evals/results/launch/jev_laya_bench.json --workers 16

run.py calls one case at a time, which is right for local models (the latency IS the model's) but
takes hours over an API. Here each case is still one independent request and its latency is that
request's own round trip (network included); only the scheduling is parallel. Scoring reuses
run.py's summarize(), so accuracy / Wilson CI / ECE / AURC are computed identically for every
system. Cost per 1M decisions is the mean BILLED cost reported by the API, times 1e6.
Resumable: rows already in <out>.partial.jsonl are not re-requested.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import make_adapter  # noqa: E402
from run import summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--adapter", default="jev_openrouter")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(a.cases, encoding="utf-8")][: a.limit]
    ad = make_adapter(a.adapter, model=a.model)
    part = Path(a.out + ".partial.jsonl")
    done = {}
    if part.exists():
        for l in open(part, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("invalid"):
                done[r["id"]] = r
    todo = [c for c in cases if c["id"] not in done]
    print(f"[run_api] {ad.name}: {len(cases)} cases, {len(done)} cached, {len(todo)} to request", flush=True)
    lock = threading.Lock()
    fout = open(part, "a", encoding="utf-8")

    def one(c):
        t = time.perf_counter()
        sel, conf, meta = ad.decide(c["state"], c["question"], c["options"])
        ms = (time.perf_counter() - t) * 1000
        return {"id": c["id"], "capability": c["capability"], "domain": c["domain"], "selected": sel,
                "confidence": float(conf), "correct": int(sel == c["answer"]), "latency_ms": ms,
                "input_chars": len(c["state"]) + len(c["question"]) + sum(map(len, c["options"])), **meta}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, c) for c in todo]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            with lock:
                fout.write(json.dumps(r) + "\n")
                fout.flush()
                done[r["id"]] = r
            if i % 500 == 0:
                print(f"[run_api] {i}/{len(todo)} in {time.time() - t0:.0f}s", flush=True)
    fout.close()
    rows = [done[c["id"]] for c in cases if c["id"] in done]
    by_cap, by_dom = collections.defaultdict(list), collections.defaultdict(list)
    for r in rows:
        by_cap[r["capability"]].append(r)
        by_dom[r["domain"]].append(r)
    costs = [r["cost_usd"] for r in rows if r.get("cost_usd") is not None]
    cost = round(statistics.fmean(costs) * 1e6, 2) if costs else ad.cost_per_million(
        sum(r["input_chars"] for r in rows) / max(1, len(rows)))
    served = collections.Counter(r.get("served_model") for r in rows)
    res = {"system": ad.name, "n": len(rows), "overall": summarize(rows),
           "by_capability": {k: summarize(v) for k, v in sorted(by_cap.items())},
           "by_domain": {k: summarize(v) for k, v in sorted(by_dom.items())},
           "cost_per_million_decisions_usd": cost, "cost_basis": "billed (mean per-call cost x 1e6)" if costs else "estimated",
           "total_billed_usd": round(sum(costs), 4), "served_models": dict(served), "gpu": "api",
           "invalid": sum(1 for r in rows if r.get("invalid")), "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "by_domain")}, indent=1))


if __name__ == "__main__":
    main()
