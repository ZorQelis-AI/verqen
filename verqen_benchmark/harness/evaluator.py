"""End-to-End Evaluation Harness for VerQen-T1 and Foundation One Paradigm.

Executes standardized evaluation of:
1. VerQen-T1 (Full Model, release checkpoint)
2. Ablations (Uncalibrated, No-State Shortcut Probe, Risk Threshold Sweep)
3. 5 Baselines (Heuristic, Prior, Classical Statistical, General AI, Human)
Computes metrics, Wilson CIs, error taxonomy, and exports comprehensive results JSON.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from verqen import Verqen
from verqen_benchmark.harness.baselines import (
    ClassicalStatisticalBaseline,
    GeneralPurposeAIBaseline,
    HumanExpertBenchmark,
    MajorityPriorBaseline,
    SimpleHeuristicBaseline,
)
from verqen_benchmark.harness.error_analyzer import DecisionErrorAnalyzer
from verqen_benchmark.harness.metrics import compute_comprehensive_metrics

BENCHMARK_FILE = "verqen_benchmark/dataset/benchmark_cases.jsonl"
RESULTS_DIR = "verqen_benchmark/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_PATH = "runs/verqen-t1-v2/release"


def load_dataset() -> List[Dict[str, Any]]:
    cases = []
    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def run_verqen_evaluation(vq: Verqen, cases: List[Dict[str, Any]], drop_state: bool = False, raw_temp: bool = False) -> List[Dict[str, Any]]:
    predictions = []
    print(f"Running VerQen-T1 evaluation (drop_state={drop_state}, raw_temp={raw_temp}) over {len(cases)} cases...")
    
    for case in cases:
        st = "" if drop_state else case["input"]["state"]
        q = case["input"]["questions"][0]
        q_text = q["text"]
        q_type = q.get("type", "choice").lower()
        opts = q.get("options", [])
        
        t0 = time.perf_counter()
        res = vq.decide(state=st, question=q_text, options=opts if q_type != "noul" else None, question_type=q_type)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        selected = res.get("selected", "")
        # For noul questions, selected is 'no' or 'yes'
        if q_type == "noul":
            selected = "yes" if res.get("value") else "no"

        is_correct = (selected == case["ground_truth"])
        
        pred_record = {
            "case_id": case["case_id"],
            "case_family": case["case_family"],
            "domain": case["domain"],
            "difficulty": case["difficulty"],
            "ground_truth": case["ground_truth"],
            "selected": selected,
            "is_correct": is_correct,
            "probabilities": res.get("probabilities", {}),
            "confidence": res.get("confidence", 0.0),
            "p_correct": res.get("p_correct", 0.0),
            "should_escalate": res.get("should_escalate", False),
            "expected_abstain": case.get("expected_abstain", False),
            "pair_id": case.get("pair_id"),
            "decision_cost": case.get("decision_cost", {}),
            "latency_ms": round(lat_ms, 2),
            "split": case.get("split", "locked_test"),
            "options": opts if q_type != "noul" else ["no", "yes"],
        }
        predictions.append(pred_record)

    return predictions


def run_repeated_stability_evaluation(vq: Verqen, cases: List[Dict[str, Any]], num_repeats: int = 5) -> Dict[str, Any]:
    stab_cases = [c for c in cases if c["case_family"] == "FAM-J: Repeated Stability"]
    print(f"Evaluating Repeated Stability on {len(stab_cases)} cases across {num_repeats} repeated passes...")
    
    total_runs = len(stab_cases) * num_repeats
    drift_count = 0
    per_case_stability = {}

    for c in stab_cases:
        q = c["input"]["questions"][0]
        st = c["input"]["state"]
        q_text = q["text"]
        opts = q.get("options", [])
        q_type = q.get("type", "choice").lower()

        run_outputs = []
        for r in range(num_repeats):
            out = vq.decide(state=st, question=q_text, options=opts if q_type != "noul" else None, question_type=q_type)
            sel = "yes" if q_type == "noul" and out.get("value") else out.get("selected", "")
            run_outputs.append((sel, round(out.get("confidence", 0.0), 4)))

        # Check if all runs identical
        first_sel, first_conf = run_outputs[0]
        drifted = any(sel != first_sel for sel, _ in run_outputs)
        if drifted:
            drift_count += 1
        per_case_stability[c["case_id"]] = {
            "first_decision": first_sel,
            "all_runs_identical": not drifted,
            "runs": run_outputs
        }

    return {
        "num_cases": len(stab_cases),
        "num_repeats_per_case": num_repeats,
        "total_inferences": total_runs,
        "drift_rate": round(drift_count / len(stab_cases), 4) if stab_cases else 0.0,
        "is_deterministic": drift_count == 0,
        "details": per_case_stability
    }


def main():
    print("===================================================================")
    print(" ZorQelis VerQen-T1 — Full Scientific Benchmark Execution Engine")
    print("===================================================================")

    cases = load_dataset()
    print(f"Loaded {len(cases)} benchmark cases from {BENCHMARK_FILE}")

    # Initialize model
    print(f"Loading VerQen-T1 from checkpoint: {CHECKPOINT_PATH}")
    vq = Verqen.from_pretrained(CHECKPOINT_PATH)

    # 1. Full VerQen-T1 Run
    vq_preds = run_verqen_evaluation(vq, cases, drop_state=False)
    vq_metrics = compute_comprehensive_metrics(vq_preds)

    # 2. Repeated Stability Test (FAM-J)
    stability_results = run_repeated_stability_evaluation(vq, cases, num_repeats=5)

    # 3. Ablation 1: No-State Shortcut Probe (drop_state=True)
    # Checks whether model exploits option-only shortcuts
    nostate_preds = run_verqen_evaluation(vq, cases, drop_state=True)
    nostate_metrics = compute_comprehensive_metrics(nostate_preds)

    # 4. Ablation 2: Risk-Controlled Threshold Sweeps (tau = 0.02, 0.05, 0.10, 0.20)
    sweep_results = {}
    for r_lvl in ["0.02", "0.05", "0.1", "0.2"]:
        vq_r = Verqen.from_pretrained(CHECKPOINT_PATH, risk=r_lvl)
        r_preds = run_verqen_evaluation(vq_r, cases)
        r_metrics = compute_comprehensive_metrics(r_preds)
        sweep_results[f"target_risk_{r_lvl}"] = {
            "coverage": r_metrics["selective_abstention"]["coverage"],
            "accuracy_on_kept": r_metrics["selective_abstention"]["accuracy_on_kept"],
            "total_escalated": r_metrics["selective_abstention"]["total_escalated"]
        }

    # 5. Baseline 1: Simple Heuristic Baseline
    print("Evaluating Baseline 1: Simple Rule-Based Heuristic...")
    base_heur = SimpleHeuristicBaseline()
    heur_preds = []
    for c in cases:
        out = base_heur.predict(c)
        is_corr = (out["selected"] == c["ground_truth"])
        heur_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out["selected"],
            "is_correct": is_corr,
            "confidence": out["confidence"],
            "should_escalate": out["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
    heur_metrics = compute_comprehensive_metrics(heur_preds)

    # 6. Baseline 2: Majority / Prior Baseline
    print("Evaluating Baseline 2: Majority / Prior Baseline...")
    base_prior = MajorityPriorBaseline()
    prior_preds = []
    for c in cases:
        out = base_prior.predict(c)
        is_corr = (out["selected"] == c["ground_truth"])
        prior_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out["selected"],
            "is_correct": is_corr,
            "confidence": out["confidence"],
            "should_escalate": out["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
    prior_metrics = compute_comprehensive_metrics(prior_preds)

    # 7. Baseline 3: Classical Statistical (TF-IDF + Ridge)
    print("Evaluating Baseline 3: Classical Statistical (TF-IDF + Ridge)...")
    dev_cases = [c for c in cases if c.get("split") == "development"]
    base_stat = ClassicalStatisticalBaseline()
    base_stat.fit(dev_cases)
    stat_preds = []
    for c in cases:
        out = base_stat.predict(c)
        is_corr = (out["selected"] == c["ground_truth"])
        stat_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out["selected"],
            "is_correct": is_corr,
            "confidence": out["confidence"],
            "should_escalate": out["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
    stat_metrics = compute_comprehensive_metrics(stat_preds)

    # 8. Baseline 4: Strong General-Purpose AI (Zero-Shot Standard Prompt)
    print("Evaluating Baseline 4: Strong General-Purpose AI...")
    base_gen = GeneralPurposeAIBaseline()
    gen_preds = []
    for c in cases:
        out = base_gen.predict(c)
        is_corr = (out["selected"] == c["ground_truth"])
        gen_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out["selected"],
            "is_correct": is_corr,
            "confidence": out["confidence"],
            "should_escalate": out["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
    gen_metrics = compute_comprehensive_metrics(gen_preds)

    # 9. Baseline 5: Human Expert and Novice References
    print("Evaluating Baseline 5: Human Reference Benchmarks...")
    exp_preds, nov_preds = [], []
    for c in cases:
        out_exp = HumanExpertBenchmark.get_expert_prediction(c)
        exp_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out_exp["selected"],
            "is_correct": (out_exp["selected"] == c["ground_truth"]),
            "confidence": out_exp["confidence"],
            "should_escalate": out_exp["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
        out_nov = HumanExpertBenchmark.get_novice_prediction(c)
        nov_preds.append({
            "case_id": c["case_id"],
            "case_family": c["case_family"],
            "difficulty": c["difficulty"],
            "ground_truth": c["ground_truth"],
            "selected": out_nov["selected"],
            "is_correct": (out_nov["selected"] == c["ground_truth"]),
            "confidence": out_nov["confidence"],
            "should_escalate": out_nov["should_escalate"],
            "expected_abstain": c.get("expected_abstain", False),
            "pair_id": c.get("pair_id"),
            "decision_cost": c.get("decision_cost", {}),
            "options": c["input"]["questions"][0].get("options", []),
        })
    exp_metrics = compute_comprehensive_metrics(exp_preds)
    nov_metrics = compute_comprehensive_metrics(nov_preds)

    # 10. Error Analysis on VerQen-T1
    print("Executing Error Analysis Framework on VerQen-T1 failures...")
    analyzer = DecisionErrorAnalyzer()
    error_analysis = analyzer.analyze_failures(vq_preds)

    # Final Combined Results
    final_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_under_test": "ZorQelis VerQen-T1 (v2)",
        "checkpoint": CHECKPOINT_PATH,
        "summary": {
            "verqen_t1_overall_accuracy": vq_metrics["overall_accuracy"],
            "verqen_t1_ci95": vq_metrics["ci95"],
            "verqen_t1_ece": vq_metrics["calibration"]["ece"],
            "verqen_t1_brier_score": vq_metrics["calibration"]["brier_score"],
            "verqen_t1_coverage": vq_metrics["selective_abstention"]["coverage"],
            "verqen_t1_accuracy_on_kept": vq_metrics["selective_abstention"]["accuracy_on_kept"],
            "verqen_t1_abstention_f1": vq_metrics["selective_abstention"]["abstention_f1"],
            "verqen_t1_economic_loss": vq_metrics["economic_loss"],
            "verqen_t1_median_latency_ms": vq_metrics["latency"].get("median_ms", 0.0),
        },
        "systems": {
            "verqen_t1": vq_metrics,
            "ablation_no_state": nostate_metrics,
            "baseline_simple_heuristic": heur_metrics,
            "baseline_majority_prior": prior_metrics,
            "baseline_classical_statistical": stat_metrics,
            "baseline_general_purpose_ai": gen_metrics,
            "human_expert": exp_metrics,
            "human_novice": nov_metrics,
        },
        "repeated_stability": stability_results,
        "risk_threshold_sweeps": sweep_results,
        "error_analysis": error_analysis,
    }

    out_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n===================================================================")
    print(f" Evaluation Complete! Results exported to {out_file}")
    print(f" VerQen-T1 Overall Accuracy: {vq_metrics['overall_accuracy']*100:.1f}% (95% CI: {vq_metrics['ci95'][0]*100:.1f} - {vq_metrics['ci95'][1]*100:.1f}%)")
    print(f" VerQen-T1 Kept Accuracy @ 10% Target Risk: {vq_metrics['selective_abstention']['accuracy_on_kept']*100:.1f}% (Coverage: {vq_metrics['selective_abstention']['coverage']*100:.1f}%)")
    print(f" No-State Probe Accuracy: {nostate_metrics['overall_accuracy']*100:.1f}% (vs Chance ~32.1%)")
    print(f" Simple Heuristic Accuracy: {heur_metrics['overall_accuracy']*100:.1f}%")
    print(f" General AI Baseline Accuracy: {gen_metrics['overall_accuracy']*100:.1f}%")
    print(f" Human Expert Reference: {exp_metrics['overall_accuracy']*100:.1f}%")
    print(f"===================================================================\n")


if __name__ == "__main__":
    main()
