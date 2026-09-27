"""Paired head-to-head of two result files from evals/jev_parity/run.py on the same case set.

    python evals/compare.py evals/results/verqen_t2_laya_bench.json evals/results/laya_laya_bench.json

Per capability (and overall): both accuracies, the difference A - B with a 95% bootstrap CI
(cases resampled jointly, so the pairing is kept), an exact McNemar p-value on the discordant
cases, ECE and AURC side by side, and latency. A capability counts as a WIN or LOSS only when the
CI excludes 0; otherwise it is a TIE. Cases missing from either file are dropped and counted.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def boot_ci(diffs: list[int], reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = json.load(open(a_path)), json.load(open(b_path))
    ra = {r["id"]: r for r in A["rows"]}
    rb = {r["id"]: r for r in B["rows"]}
    ids = sorted(set(ra) & set(rb))
    groups = collections.defaultdict(list)
    for i in ids:
        groups[ra[i]["capability"]].append(i)
        groups["ALL"].append(i)
    print(f"A = {A['system']}   B = {B['system']}   paired cases: {len(ids)} "
          f"(dropped {len(set(ra) ^ set(rb))}); GPU A={A.get('gpu')} B={B.get('gpu')}")
    print(f"{'capability':16s} {'n':>5s} {'A':>6s} {'B':>6s} {'A-B':>7s} {'95% CI':>17s} {'McNemar p':>10s}  verdict")
    out = {}
    for cap in sorted(groups, key=lambda k: (k == "ALL", k)):
        g = groups[cap]
        ca = [ra[i]["correct"] for i in g]
        cb = [rb[i]["correct"] for i in g]
        d = [x - y for x, y in zip(ca, cb)]
        lo, hi = boot_ci(d)
        b = sum(1 for x, y in zip(ca, cb) if x and not y)
        c = sum(1 for x, y in zip(ca, cb) if y and not x)
        p = mcnemar_exact(b, c)
        verdict = "WIN A" if lo > 0 else "WIN B" if hi < 0 else "tie"
        acc_a, acc_b = sum(ca) / len(g), sum(cb) / len(g)
        out[cap] = {"n": len(g), "a": round(acc_a, 4), "b": round(acc_b, 4), "diff": round(acc_a - acc_b, 4),
                    "ci95": [round(lo, 4), round(hi, 4)], "mcnemar_p": p, "verdict": verdict}
        print(f"{cap:16s} {len(g):5d} {acc_a:6.3f} {acc_b:6.3f} {acc_a - acc_b:+7.3f} [{lo:+.3f}, {hi:+.3f}] {p:10.2g}  {verdict}")
    sa, sb = A["overall"], B["overall"]
    print(f"\nECE      A {sa['ece']}   B {sb['ece']}        (lower is better)")
    print(f"AURC     A {sa['aurc']}   B {sb['aurc']}       (lower is better)")
    print(f"p50 ms   A {sa['latency_p50_ms']}   B {sb['latency_p50_ms']};  p95 ms A {sa['latency_p95_ms']}   B {sb['latency_p95_ms']}")
    print(f"$ per 1M decisions   A {A['cost_per_million_decisions_usd']}   B {B['cost_per_million_decisions_usd']}")
    print(f"invalid (could not be encoded)   A {A.get('invalid', 0)}   B {B.get('invalid', 0)}")
    json.dump({"a": A["system"], "b": B["system"], "by_capability": out,
               "ece": [sa["ece"], sb["ece"]], "aurc": [sa["aurc"], sb["aurc"]],
               "latency_p50_ms": [sa["latency_p50_ms"], sb["latency_p50_ms"]],
               "cost_per_million": [A["cost_per_million_decisions_usd"], B["cost_per_million_decisions_usd"]]},
              open(a_path.replace(".json", "_vs.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
