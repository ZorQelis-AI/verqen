"""Corrected analysis of Artifact-G predictions, reported *next to* the original harness metrics.

    python scripts/audit/corrected_eval.py --tag baseline [--tag bench_AB ...]

It never changes labels, cases or the harness. It reads predictions written by
repro_benchmark.py (which calls the unchanged harness) and adds:

  1. template-clustered accuracy: cases from one template are not independent, so the CI
     comes from a cluster bootstrap, and the effective n is the number of templates;
  2. valid baselines only: chance and first-option are computed here. The two oracle
     "baselines" that read the ground truth are flagged invalid, and ECE is reported only
     for systems that output probabilities;
  3. proper pairing: every pair_id with two members, split into counterfactual pairs
     (labels differ) and invariance pairs (labels equal). One-sided pair ids are counted;
  4. high-stakes: wrong-and-not-escalated rate among cases with error_cost >= 20, per family;
  5. selective prediction from p_correct: AUROC and AURC, plus the risk at the operating
     point the model actually used. This is descriptive only; no threshold is fitted here;
  6. dev/test template overlap, which explains why a template-memorizing classifier can
     beat a general model.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np

BENCH = "verqen_benchmark/dataset/benchmark_cases.jsonl"


def template_key(case):
    q = case["input"]["questions"][0]
    masked_q = re.sub(r"\d+(\.\d+)?", "#", q["text"].lower())
    return (case["case_family"][:5], masked_q, tuple(sorted(q.get("options") or [])))


def cluster_bootstrap(correct_by_cluster: dict, reps=4000, seed=0):
    keys = list(correct_by_cluster)
    rng = np.random.default_rng(seed)
    sums = np.array([sum(correct_by_cluster[k]) for k in keys], float)
    ns = np.array([len(correct_by_cluster[k]) for k in keys], float)
    stats = []
    for _ in range(reps):
        idx = rng.integers(0, len(keys), len(keys))
        stats.append(sums[idx].sum() / ns[idx].sum())
    return [round(float(np.percentile(stats, 2.5)), 4), round(float(np.percentile(stats, 97.5)), 4)]


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1
    return round(float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def aurc(conf, y):
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    err = 1 - np.asarray(y, float)[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    return round(float(risks.mean()), 4)


def top_label_ece(conf, y, bins=10):
    conf, y = np.asarray(conf, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - y[m].mean())
    return round(float(e), 4)


def analyze(preds, cases):
    by_id = {c["case_id"]: c for c in cases}
    rows = [(p, by_id[p["case_id"]]) for p in preds]
    y = np.array([int(p["is_correct"]) for p, _ in rows])
    rep = {"n_cases": len(rows), "accuracy": round(float(y.mean()), 4)}
    clusters = collections.defaultdict(list)
    fam_clusters = collections.defaultdict(lambda: collections.defaultdict(list))
    for (p, c), yi in zip(rows, y):
        k = template_key(c)
        clusters[k].append(int(yi))
        fam_clusters[c["case_family"][:5]][k].append(int(yi))
    rep["template_clusters"] = len(clusters)
    rep["accuracy_cluster_ci95"] = cluster_bootstrap(clusters)
    rep["accuracy_template_macro"] = round(float(np.mean([np.mean(v) for v in clusters.values()])), 4)
    rep["families"] = {}
    for fam, cl in sorted(fam_clusters.items()):
        vals = [x for v in cl.values() for x in v]
        rep["families"][fam] = {"n": len(vals), "templates": len(cl), "accuracy": round(float(np.mean(vals)), 4),
                                "cluster_ci95": cluster_bootstrap(cl, reps=2000)}
    # valid simple baselines
    n_opts = [len(c["input"]["questions"][0].get("options") or ["no", "yes"]) for _, c in rows]
    first = [int((c["input"]["questions"][0].get("options") or ["no"])[0] == c["ground_truth"]) for _, c in rows]
    rep["baselines_valid"] = {"chance_expected": round(float(np.mean([1 / n for n in n_opts])), 4),
                              "first_option": round(float(np.mean(first)), 4)}
    # calibration (only meaningful because VerQen emits probabilities)
    conf = np.array([max(p["probabilities"].values()) for p, _ in rows])
    pc = np.array([p["p_correct"] for p, _ in rows])
    rep["calibration"] = {"top_label_ece_probs": top_label_ece(conf, y), "top_label_ece_p_correct": top_label_ece(pc, y),
                          "auroc_p_correct": auroc(pc, y), "auroc_max_prob": auroc(conf, y),
                          "aurc_p_correct": aurc(pc, y), "aurc_max_prob": aurc(conf, y),
                          "aurc_oracle": aurc(y.astype(float), y)}
    esc = np.array([bool(p["should_escalate"]) for p, _ in rows])
    kept = ~esc
    rep["operating_point"] = {"coverage": round(float(kept.mean()), 4),
                              "risk_on_kept": round(float(1 - y[kept].mean()), 4) if kept.any() else None,
                              "note": "the escalation threshold is the one fitted on the model's calibration split"}
    # pairs
    groups = collections.defaultdict(list)
    for (p, c), yi in zip(rows, y):
        if c.get("pair_id"):
            groups[c["pair_id"]].append((p, c))
    cf, inv, one_sided = [], [], 0
    for pid, g in groups.items():
        if len(g) != 2:
            one_sided += 1
            continue
        (pa, ca), (pb, cb) = g
        if ca["ground_truth"] != cb["ground_truth"]:
            cf.append({"pair": pid, "both_correct": pa["is_correct"] and pb["is_correct"], "flipped": pa["selected"] != pb["selected"]})
        else:
            inv.append({"pair": pid, "consistent": pa["selected"] == pb["selected"], "both_correct": pa["is_correct"] and pb["is_correct"]})
    rep["pairs"] = {"counterfactual_pairs": len(cf), "cf_both_correct": sum(x["both_correct"] for x in cf),
                    "cf_decision_flipped": sum(x["flipped"] for x in cf), "invariance_pairs": len(inv),
                    "inv_consistent": sum(x["consistent"] for x in inv), "inv_both_correct": sum(x["both_correct"] for x in inv),
                    "one_sided_pair_ids": one_sided}
    # high stakes
    hs = collections.defaultdict(lambda: [0, 0, 0])  # n, wrong, wrong_and_not_escalated
    for (p, c), yi in zip(rows, y):
        if c["decision_cost"]["error_cost"] >= 20:
            h = hs[c["case_family"][:5]]
            h[0] += 1; h[1] += int(not yi); h[2] += int((not yi) and not p["should_escalate"])
    tot = np.array(list(hs.values())).sum(0) if hs else np.zeros(3)
    rep["high_stakes"] = {"n": int(tot[0]), "wrong": int(tot[1]), "silent_failures": int(tot[2]),
                          "silent_failure_rate": round(float(tot[2] / max(1, tot[0])), 4),
                          "by_family": {k: {"n": v[0], "wrong": v[1], "silent_failures": v[2]} for k, v in sorted(hs.items())}}
    return rep


def dev_test_overlap(cases):
    dev = {template_key(c) for c in cases if c["split"] in ("development", "validation")}
    test = [c for c in cases if c["split"] == "locked_test"]
    return {"locked_test_cases": len(test), "share_template_with_dev_or_val": round(sum(template_key(c) in dev for c in test) / len(test), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=[])
    a = ap.parse_args()
    cases = [json.loads(l) for l in open(BENCH, encoding="utf-8") if l.strip()]
    out = {"dev_test_template_overlap": dev_test_overlap(cases),
           "invalid_baselines": {"baseline_general_purpose_ai": "returns case['ground_truth'] with hash() noise (oracle)",
                                 "human_expert": "returns case['ground_truth'] with hash() noise (oracle)",
                                 "human_novice": "returns case['ground_truth'] with hash() noise (oracle)",
                                 "ece_of_non_probabilistic_systems": "harness defaults ECE to 0.0; not a measurement"}}
    for tag in a.tag or ["baseline"]:
        d = Path("runs/audit") / tag
        preds = json.load(open(d / "predictions.json"))
        rep = analyze(preds, cases)
        locked = [p for p in preds if p.get("split") == "locked_test"]
        rep["locked_test_only"] = analyze(locked, cases) if locked else None
        harness = json.load(open(d / "harness_metrics.json"))
        rep["original_harness_metrics"] = {k: harness["full"][k] for k in ("overall_accuracy", "ci95", "economic_loss")}
        rep["original_harness_metrics"]["ece"] = harness["full"]["calibration"]["ece"]
        out[tag] = rep
        json.dump(rep, open(d / "corrected.json", "w"), indent=2)
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
