"""Before/after comparison of the ablation arms, with the pre-registered selection rule applied.

    python scripts/audit/compare_arms.py

Pre-registered in docs/audit/ROOT_CAUSES.md before any arm finished:
  select the arm with the best mean of (held-out probe aggregate, real-benchmark mean accuracy),
  subject to: procedural test and real-benchmark mean within 1 pt of BASE, and invariance
  intact (max total-variation change under option reversal / question prepending <= 0.02;
  exact in fp32 by construction and pytest, the bound only admits bf16 noise, which is ~0.01
  for the release). Artifact-G is reported for every arm and never used for selection.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
ARMS = ["base", "A", "B", "AB"]


def load(p):
    return json.load(open(p)) if Path(p).exists() else None


def probe_summary(tag):
    pr, cm = load(f"runs/audit/{tag}/probes.json"), load(f"runs/audit/{tag}/cf_missing.json")
    if not pr or not cm:
        return None
    d = pr["distractor"]
    bait = [v["accuracy"] for k, v in d.items() if "bait=named" in k]
    nobait = [v["accuracy"] for k, v in d.items() if "bait=none" in k]
    distractor = float(np.mean([v["accuracy"] for v in d.values()]))
    boundary = float(np.mean(list(pr["boundary"]["accuracy_by_delta"].values())))
    miss_v1 = [pr["missing"][c]["accuracy"] for c in ("complete", "missing", "unknown", "contradictory")]
    mv2 = cm["missing_v2"]
    miss_v2 = [mv2[c]["correct"] / mv2[c]["n"] for c in ("complete", "missing", "unknown", "contradictory")]
    missing = float(np.mean(miss_v1 + miss_v2))
    cf = cm["counterfactual"]
    cf_rel = float(np.mean([cf[k]["both_correct"] / cf[k]["pairs"] for k in ("role|relevant", "sla|relevant")]))
    cf_irr = float(np.mean([1 - cf[k]["flip"] / cf[k]["pairs"] for k in ("role|irrelevant", "sla|irrelevant")]))
    parts = {"distractor": distractor, "distractor_bait": float(np.mean(bait)), "distractor_nobait": float(np.mean(nobait)),
             "boundary": boundary, "missing_info": missing,
             "missing_info_request_rate_when_incomplete": float(np.mean(
                 [pr["missing"][c]["chose_request_info"] for c in ("missing", "unknown", "contradictory")] +
                 [mv2[c]["chose_info"] / mv2[c]["n"] for c in ("missing", "unknown", "contradictory")])),
             "missing_info_request_rate_when_complete": float(np.mean([pr["missing"]["complete"]["chose_request_info"],
                                                                        mv2["complete"]["chose_info"] / mv2["complete"]["n"]])),
             "cf_relevant_both_correct": cf_rel, "cf_irrelevant_stable": cf_irr}
    parts["aggregate"] = float(np.mean([distractor, boundary, missing, cf_rel, cf_irr]))
    return {k: round(v, 4) for k, v in parts.items()}


def suite_summary(path):
    r = load(path)
    if not r:
        return None
    real = {k: v["accuracy"] for k, v in r["splits"].items() if not k.startswith("proc_")}
    sel_viol = {k: v["selective"]["accuracy_on_kept"] for k, v in r["splits"].items() if not k.startswith("proc_")}
    return {"real_mean": round(float(np.mean(list(real.values()))), 4), "real": real,
            "proc_test": r["splits"]["proc_test"]["accuracy"], "proc_test_ood": r["splits"]["proc_test_ood"]["accuracy"],
            "proc_test_adv": r["splits"]["proc_test_adv"]["accuracy"],
            "invariance": r.get("invariance"), "environments": r.get("environments"),
            "speed": {k: r["speed"].get(k) for k in ("single_request_latency_ms", "batched_decisions_per_s") if k in r.get("speed", {})},
            "risk0.1_accuracy_on_kept_min_real": round(min(sel_viol.values()), 4),
            "risk0.1_sources_violating": sum(v < 0.9 for v in sel_viol.values())}


def bench_summary(tag):
    h, c = load(f"runs/audit/{tag}/harness_metrics.json"), load(f"runs/audit/{tag}/corrected.json")
    if not h:
        return None
    f = h["full"]
    out = {"accuracy": f["overall_accuracy"], "ci95_harness": f["ci95"], "ece": f["calibration"]["ece"],
           "brier": f["calibration"]["brier_score"], "coverage": f["selective_abstention"]["coverage"],
           "accuracy_on_kept": f["selective_abstention"]["accuracy_on_kept"], "economic_loss": f["economic_loss"],
           "cf_sensitivity_harness": f["behavioral_invariants"]["counterfactual_sensitivity_rate"],
           "invariance_harness": f["behavioral_invariants"]["invariance_consistency_rate"],
           "median_latency_ms": f["latency"]["median_ms"],
           "families": {k[:5]: v["accuracy"] for k, v in f["family_breakdown"].items()},
           "no_state_accuracy": h["no_state"]["overall_accuracy"]}
    if c:
        out.update({"cluster_ci95": c["accuracy_cluster_ci95"], "template_macro": c["accuracy_template_macro"],
                    "locked_test_accuracy": c["locked_test_only"]["accuracy"] if c.get("locked_test_only") else None,
                    "auroc_p_correct": c["calibration"]["auroc_p_correct"], "aurc_p_correct": c["calibration"]["aurc_p_correct"],
                    "high_stakes_wrong": c["high_stakes"]["wrong"], "high_stakes_silent": c["high_stakes"]["silent_failures"],
                    "risk_on_kept": c["operating_point"]["risk_on_kept"]})
    return out


def mcnemar(tag_a, tag_b):
    pa, pb = load(f"runs/audit/{tag_a}/predictions.json"), load(f"runs/audit/{tag_b}/predictions.json")
    if not pa or not pb:
        return None
    a = {p["case_id"]: p["is_correct"] for p in pa}
    b = {p["case_id"]: p["is_correct"] for p in pb}
    n01 = sum(1 for k in a if not a[k] and b[k])
    n10 = sum(1 for k in a if a[k] and not b[k])
    n = n01 + n10
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return {"b_fixes": n01, "b_breaks": n10, "exact_p": round(p, 4)}


def main():
    for tag in ["baseline"] + [f"bench_{a}" for a in ARMS]:
        if Path(f"runs/audit/{tag}/predictions.json").exists() and not Path(f"runs/audit/{tag}/corrected.json").exists():
            subprocess.run([sys.executable, "scripts/audit/corrected_eval.py", "--tag", tag], check=True, stdout=subprocess.DEVNULL)
    table = {"release (before)": {"suite": suite_summary("runs/verqen-t1-v2/eval/report.json"),
                                  "probes": probe_summary("probes_baseline"), "artifact_g": bench_summary("baseline")}}
    for a in ARMS:
        table[a] = {"suite": suite_summary(f"runs/fix/{a}/eval/report.json"), "probes": probe_summary(f"probes_{a}"),
                    "artifact_g": bench_summary(f"bench_{a}"), "fix_families": load(f"runs/fix/{a}/eval/fix_families.json"),
                    "mcnemar_vs_base": mcnemar("bench_base", f"bench_{a}") if a != "base" else None,
                    "run_info": load(f"runs/fix/{a}/run_info.json")}
    base = table["base"]
    eligible, scores = [], {}
    for a in ARMS:
        s, p = table[a]["suite"], table[a]["probes"]
        if not s or not p:
            continue
        ok = (s["proc_test"] >= base["suite"]["proc_test"] - 0.01 and s["real_mean"] >= base["suite"]["real_mean"] - 0.01
              and max((s["invariance"] or {}).get("option_reversal_tv_max", 1.0),
                      (s["invariance"] or {}).get("prepend_question_tv_max", 1.0)) <= 0.02)
        scores[a] = round((p["aggregate"] + s["real_mean"]) / 2, 4)
        if ok:
            eligible.append(a)
    selected = max(eligible, key=scores.get) if eligible else None
    table["selection"] = {"rule": "argmax mean(probe aggregate, real mean) s.t. proc_test & real_mean >= BASE-1pt, invariance TV max <= 0.02",
                          "scores": scores, "eligible": eligible, "selected": selected}
    json.dump(table, open("runs/fix/comparison.json", "w"), indent=2)
    print(json.dumps(table["selection"], indent=1))
    for a in ["release (before)"] + ARMS:
        t = table[a]
        if t["suite"] and t["probes"]:
            print(f"{a:18s} real={t['suite']['real_mean']:.4f} proc={t['suite']['proc_test']:.4f} "
                  f"probe={t['probes']['aggregate']:.4f} bench={(t['artifact_g'] or {}).get('accuracy')}")


if __name__ == "__main__":
    main()
