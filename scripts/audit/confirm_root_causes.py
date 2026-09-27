"""Confirmation experiments for the root-cause hypotheses (diagnostics only, no model changes).

A  numeric comparison inside a *training* template (refund_policy): accuracy vs distance to the
   threshold. Distinguishes "template overfit" from "never learned comparison".
B  distractor bait: how often the chosen wrong option is the one the distractor text mentions.
C  selective scoring on the benchmark predictions: AUROC of p_correct vs max-probability.
D  SLA counterfactual pairs: which direction the model moves.
"""

from __future__ import annotations

import collections
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
os.chdir(ROOT)

import probes as P  # noqa: E402
from verqen.inference.engine import VerqenInferenceEngine  # noqa: E402


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def refund_state(days_limit, amount_limit, days, amount, vip, order):
    policy = (f"Refund policy: refunds are accepted within {days_limit} days of delivery (VIP members: no time limit). "
              f"Refunds above ${amount_limit} need manager approval. Opened digital products are never refundable.")
    case = f"Case: Maria requests a ${amount} refund for a desk lamp delivered {days} days ago. VIP member: {'yes' if vip else 'no'}."
    return f"{policy}\n{case}" if order else f"{case}\n{policy}"


def main():
    eng = VerqenInferenceEngine.from_checkpoint("runs/verqen-t1-v2/release", "auto")
    rep = {}
    rng = random.Random(7)

    # ---- A: in-template numeric comparison --------------------------------------------------
    by_d = collections.defaultdict(lambda: [0, 0])
    by_d_noul = collections.defaultdict(lambda: [0, 0])
    for _ in range(40):
        dl, al = rng.choice([14, 30, 45, 60]), rng.choice([100, 200, 500, 1000])
        amount = rng.randint(10, al - 5)  # below the manager limit, so only the time rule matters
        order = rng.random() < 0.5
        for d in (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20):
            days = dl + d
            if days < 1:
                continue
            st = refund_state(dl, al, days, amount, False, order)
            gold = "approve" if days <= dl else "deny"
            r = eng.decide_multi(st, [{"type": "choice", "text": "What is the correct decision under the policy?",
                                       "options": ["approve", "deny", "escalate to manager"]},
                                      {"type": "noul", "text": "Is the request inside the time window?"}])
            by_d[d][0] += r[0]["selected"] == gold; by_d[d][1] += 1
            by_d_noul[d][0] += r[1]["value"] == (days <= dl); by_d_noul[d][1] += 1
    rep["A_refund_in_template"] = {"choice_acc_by_delta": {d: round(a / n, 3) for d, (a, n) in sorted(by_d.items())},
                                   "noul_acc_by_delta": {d: round(a / n, 3) for d, (a, n) in sorted(by_d_noul.items())}}

    # ---- B: distractor bait pick rate ------------------------------------------------------
    prose = P.fineweb_sentences(3000)
    dp = P.distractor_probes(prose)
    bait = collections.defaultdict(lambda: {"wrong": 0, "wrong_is_mentioned": 0, "n": 0})
    for pr in dp:
        if "bait=named" not in pr["cell"]:
            continue
        r = eng.decide_multi(pr["state"], [{"type": "choice", "text": pr["question"], "options": pr["options"]}])[0]
        b = bait[pr["cell"].split("|pos")[0]]
        b["n"] += 1
        if r["selected"] != pr["answer"]:
            b["wrong"] += 1
            words = r["selected"].split("_")
            b["wrong_is_mentioned"] += all(w.lower() in pr["state"].lower() for w in words if len(w) > 3)
    # chance that a random wrong option is a mentioned one: k mentions of 3 wrong options
    rep["B_bait"] = {k: {**v, "share_of_errors_on_mentioned": round(v["wrong_is_mentioned"] / max(1, v["wrong"]), 3),
                         "chance_share": round(min(3, int(k.split('=')[1])) / 3, 3)} for k, v in bait.items()}

    # ---- C: selective ranking on benchmark predictions --------------------------------------
    preds = json.load(open("runs/audit/baseline/predictions.json"))
    ys = [int(p["is_correct"]) for p in preds if p.get("p_correct") is not None]
    rep["C_benchmark_selective"] = {
        "n": len(ys),
        "auroc_p_correct": auroc([p["p_correct"] for p in preds if p.get("p_correct") is not None], ys),
        "auroc_max_prob": auroc([max(p["probabilities"].values()) for p in preds if p.get("p_correct") is not None], ys),
        "auroc_margin": auroc([(lambda v: v[-1] - v[-2])(sorted(p["probabilities"].values())) for p in preds if p.get("p_correct") is not None], ys),
    }

    # ---- D: SLA counterfactual direction ----------------------------------------------------
    dirs = collections.Counter()
    for pr in P.counterfactual_probes():
        if not pr["pair"].startswith("sla") or pr["kind"] != "relevant":
            continue
        ra = eng.decide_multi(pr["a"][0], [{"type": "choice", "text": pr["question"], "options": pr["options"]}])[0]
        rb = eng.decide_multi(pr["b"][0], [{"type": "choice", "text": pr["question"], "options": pr["options"]}])[0]
        dirs[f"{ra['selected']} -> {rb['selected']}"] += 1
    rep["D_sla_relevant_transitions (gold: within_target -> target_breached)"] = dict(dirs)
    Path("runs/audit/probes_baseline").mkdir(parents=True, exist_ok=True)
    json.dump(rep, open("runs/audit/probes_baseline/confirm.json", "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
