"""Metrics Calculation and Scoring System for VerQen-T1 Evaluation.

Implements exact mathematical formulations for accuracy, calibration (ECE, Brier),
selective risk-coverage, counterfactual sensitivity, invariance delta, and economic loss.
"""

from typing import Any, Dict, List, Tuple
import math
import numpy as np


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes the Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    z = 1.959963984540054  # 95% confidence
    p = k / n
    denom = 1 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    spread = (z * math.sqrt((p * (1 - p) / n) + ((z ** 2) / (4 * (n ** 2))))) / denom
    return max(0.0, round(center - spread, 4)), min(1.0, round(center + spread, 4))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) with equal-width confidence bins."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(labels)
    if n == 0:
        return 0.0
        
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n) * abs(bin_acc - bin_conf)
            
    return round(float(ece), 4)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes multiclass Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2."""
    n, k = probs.shape
    if n == 0:
        return 0.0
    one_hot = np.zeros((n, k))
    for i, y in enumerate(labels):
        if 0 <= y < k:
            one_hot[i, y] = 1.0
    score = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return round(float(score), 4)


def brier_skill_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Brier Skill Score relative to a uniform reference baseline: 1 - (BS / BS_ref)."""
    bs = brier_score(probs, labels)
    n, k = probs.shape
    if n == 0 or k == 0:
        return 0.0
    ref_probs = np.full_like(probs, 1.0 / k)
    bs_ref = brier_score(ref_probs, labels)
    if bs_ref == 0:
        return 0.0
    return round(float(1.0 - (bs / bs_ref)), 4)


def compute_comprehensive_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes the full metric suite across all evaluation predictions."""
    total = len(predictions)
    if total == 0:
        return {}

    correct_count = sum(1 for p in predictions if p["is_correct"])
    acc = correct_count / total
    lo, hi = wilson_score_interval(correct_count, total)

    # Metrics by Family
    families = sorted(list(set(p["case_family"] for p in predictions)))
    family_metrics = {}
    for fam in families:
        fam_preds = [p for p in predictions if p["case_family"] == fam]
        n_fam = len(fam_preds)
        k_fam = sum(1 for p in fam_preds if p["is_correct"])
        f_acc = k_fam / n_fam if n_fam > 0 else 0.0
        f_lo, f_hi = wilson_score_interval(k_fam, n_fam)
        family_metrics[fam] = {
            "n": n_fam,
            "accuracy": round(f_acc, 4),
            "ci95": [f_lo, f_hi],
            "mean_confidence": round(float(np.mean([p.get("confidence", 0.0) for p in fam_preds])), 4),
            "escalation_rate": round(float(np.mean([1.0 if p.get("should_escalate") else 0.0 for p in fam_preds])), 4),
        }

    # Metrics by Difficulty (1 to 5)
    diff_metrics = {}
    for d in range(1, 6):
        d_preds = [p for p in predictions if p.get("difficulty") == d]
        n_d = len(d_preds)
        if n_d > 0:
            k_d = sum(1 for p in d_preds if p["is_correct"])
            d_lo, d_hi = wilson_score_interval(k_d, n_d)
            diff_metrics[f"difficulty_{d}"] = {
                "n": n_d,
                "accuracy": round(k_d / n_d, 4),
                "ci95": [d_lo, d_hi]
            }

    # Calibration & Brier
    # Construct aligned probability and label matrices for cases with choice outputs
    prob_list, label_list = [], []
    for p in predictions:
        opts = p.get("options", [])
        gt = p.get("ground_truth")
        probs_dict = p.get("probabilities", {})
        if opts and gt in opts and probs_dict:
            max_opts = len(opts)
            prob_vec = [probs_dict.get(o, 0.0) for o in opts]
            s = sum(prob_vec)
            if s > 0:
                prob_vec = [v / s for v in prob_vec]
            prob_list.append(prob_vec)
            label_list.append(opts.index(gt))

    ece = 0.0
    bs = 0.0
    bss = 0.0
    if prob_list:
        # Pad to max options for matrix operations
        max_k = max(len(pv) for pv in prob_list)
        padded_probs = np.zeros((len(prob_list), max_k))
        for i, pv in enumerate(prob_list):
            padded_probs[i, :len(pv)] = pv
        labels_arr = np.array(label_list)
        ece = expected_calibration_error(padded_probs, labels_arr)
        bs = brier_score(padded_probs, labels_arr)
        bss = brier_skill_score(padded_probs, labels_arr)

    # Abstention & Escalation Quality
    # Expected abstain vs empirical should_escalate
    tp_abs = sum(1 for p in predictions if p.get("expected_abstain") and p.get("should_escalate"))
    fp_abs = sum(1 for p in predictions if not p.get("expected_abstain") and p.get("should_escalate"))
    fn_abs = sum(1 for p in predictions if p.get("expected_abstain") and not p.get("should_escalate"))
    tn_abs = sum(1 for p in predictions if not p.get("expected_abstain") and not p.get("should_escalate"))

    prec_abs = tp_abs / (tp_abs + fp_abs) if (tp_abs + fp_abs) > 0 else 0.0
    rec_abs = tp_abs / (tp_abs + fn_abs) if (tp_abs + fn_abs) > 0 else 0.0
    f1_abs = (2 * prec_abs * rec_abs) / (prec_abs + rec_abs) if (prec_abs + rec_abs) > 0 else 0.0

    # Selective Accuracy on Kept (Non-Escalated) Cases
    kept_preds = [p for p in predictions if not p.get("should_escalate")]
    n_kept = len(kept_preds)
    coverage = n_kept / total
    acc_kept = (sum(1 for p in kept_preds if p["is_correct"]) / n_kept) if n_kept > 0 else None

    # Counterfactual Sensitivity Rate
    # Group by pair_id for FAM-H
    cf_pairs: Dict[str, List[Dict[str, Any]]] = {}
    for p in predictions:
        pid = p.get("pair_id")
        if pid and "pair_cf" in pid:
            cf_pairs.setdefault(pid, []).append(p)
    
    cf_valid_count = 0
    cf_total_pairs = len(cf_pairs)
    for pid, pair_list in cf_pairs.items():
        if len(pair_list) == 2:
            p_a, p_b = pair_list[0], pair_list[1]
            # Sensitive if decisions are both correct and different
            if p_a["is_correct"] and p_b["is_correct"] and (p_a["selected"] != p_b["selected"]):
                cf_valid_count += 1
    cf_sensitivity_rate = round(cf_valid_count / cf_total_pairs, 4) if cf_total_pairs > 0 else 0.0

    # Invariance Consistency Rate
    inv_pairs: Dict[str, List[Dict[str, Any]]] = {}
    for p in predictions:
        pid = p.get("pair_id")
        if pid and "pair_inv" in pid:
            inv_pairs.setdefault(pid, []).append(p)
    
    inv_consistent_count = 0
    inv_total_pairs = len(inv_pairs)
    for pid, pair_list in inv_pairs.items():
        if len(pair_list) == 2:
            p_a, p_b = pair_list[0], pair_list[1]
            if p_a["selected"] == p_b["selected"]:
                inv_consistent_count += 1
    inv_consistency_rate = round(inv_consistent_count / inv_total_pairs, 4) if inv_total_pairs > 0 else 0.0

    # Total Decision Cost / Expected Economic Loss
    total_loss = 0.0
    for p in predictions:
        dcost = p.get("decision_cost", {})
        is_corr = p.get("is_correct")
        esc = p.get("should_escalate")
        if esc:
            total_loss += dcost.get("escalation_cost", 0.2)
        elif not is_corr:
            total_loss += dcost.get("error_cost", 1.0)
        else:
            total_loss -= dcost.get("correct_reward", 1.0)

    # Latencies
    latencies = [p.get("latency_ms", 0.0) for p in predictions if "latency_ms" in p]
    lat_stats = {}
    if latencies:
        lat_stats = {
            "mean_ms": round(float(np.mean(latencies)), 2),
            "median_ms": round(float(np.median(latencies)), 2),
            "p90_ms": round(float(np.percentile(latencies, 90)), 2),
            "p99_ms": round(float(np.percentile(latencies, 99)), 2),
        }

    return {
        "total_cases": total,
        "overall_accuracy": round(acc, 4),
        "ci95": [lo, hi],
        "calibration": {
            "ece": ece,
            "brier_score": bs,
            "brier_skill_score": bss
        },
        "selective_abstention": {
            "coverage": round(coverage, 4),
            "accuracy_on_kept": round(acc_kept, 4) if acc_kept is not None else None,
            "abstention_f1": round(f1_abs, 4),
            "abstention_precision": round(prec_abs, 4),
            "abstention_recall": round(rec_abs, 4),
            "total_escalated": total - n_kept,
        },
        "behavioral_invariants": {
            "counterfactual_sensitivity_rate": cf_sensitivity_rate,
            "invariance_consistency_rate": inv_consistency_rate,
        },
        "economic_loss": round(total_loss, 2),
        "latency": lat_stats,
        "family_breakdown": family_metrics,
        "difficulty_breakdown": diff_metrics,
    }
