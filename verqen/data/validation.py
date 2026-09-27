"""Dataset validation module for Verqen-T1.

Audits:
- Exact and normalized duplicate states (zero-duplicate guarantee)
- Cross-split state contamination (train vs test leakage)
- Duplicate options within a single question
- Target probability normalization (sums to ~1.0)
- NaNs, infinities, and out-of-bound probabilities
- Empty or whitespace-only states or questions
- Label and type validity
- Task family / template distribution
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def validate_dataset(
    train_path: str,
    test_path: str | None = None,
    output_report_path: str | None = "dataset_report.json",
) -> dict[str, Any]:
    """Validate dataset quality, integrity, and uniqueness.

    Returns a comprehensive validation report dictionary.
    """
    report: dict[str, Any] = {
        "status": "PASSED",
        "train_examples": 0,
        "test_examples": 0,
        "empty_states": 0,
        "empty_questions": 0,
        "duplicate_states": 0,
        "cross_split_state_leakage": 0,
        "duplicate_options": 0,
        "invalid_prob_sums": 0,
        "nan_or_inf_detected": 0,
        "invalid_types": 0,
        "type_distribution": {"choice": 0, "noul": 0, "score": 0},
        "difficulty_distribution": {},
        "domain_distribution": {},
    }

    seen_train_hashes: set[str] = set()

    # 1. Validate Train Set
    with open(train_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            report["train_examples"] += 1
            ex = json.loads(line)

            # Check state
            raw_state = ex.get("state_text", ex.get("state", "")).strip()
            if not raw_state:
                report["empty_states"] += 1

            norm_state = " ".join(raw_state.lower().split())
            state_hash = hashlib.sha256(norm_state.encode("utf-8")).hexdigest()
            if state_hash in seen_train_hashes:
                report["duplicate_states"] += 1
            seen_train_hashes.add(state_hash)

            # Check questions
            q_type = ex.get("question_type", ex.get("type", "choice")).lower()
            if q_type not in ("choice", "noul", "score"):
                report["invalid_types"] += 1
            else:
                report["type_distribution"][q_type] = report["type_distribution"].get(q_type, 0) + 1

            diff = ex.get("difficulty", "unknown")
            report["difficulty_distribution"][diff] = report["difficulty_distribution"].get(diff, 0) + 1

            dom = ex.get("domain", "unknown")
            report["domain_distribution"][dom] = report["domain_distribution"].get(dom, 0) + 1

            q_text = ex.get("question_text", ex.get("question", "")).strip()
            if not q_text:
                report["empty_questions"] += 1

            # Check options
            opts = ex.get("options", [])
            if len(opts) != len(set(opts)):
                report["duplicate_options"] += 1

            # Check target distribution
            raw_target = ex.get("target_distribution", ex.get("target", {}).get("probabilities", []))
            if isinstance(raw_target, dict):
                probs = list(raw_target.values())
            elif isinstance(raw_target, (list, tuple)):
                probs = [float(x) for x in raw_target]
            else:
                probs = []

            for p in probs:
                if math.isnan(p) or math.isinf(p) or p < 0.0 or p > 1.0:
                    report["nan_or_inf_detected"] += 1

            if q_type in ("choice", "score") and len(probs) > 1:
                total_p = sum(probs)
                if not math.isclose(total_p, 1.0, abs_tol=1e-3):
                    report["invalid_prob_sums"] += 1

    # 2. Check Test Set for cross-split leakage
    if test_path and Path(test_path).exists():
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                report["test_examples"] += 1
                ex = json.loads(line)
                raw_state = ex.get("state_text", ex.get("state", "")).strip()
                norm_state = " ".join(raw_state.lower().split())
                test_hash = hashlib.sha256(norm_state.encode("utf-8")).hexdigest()
                if test_hash in seen_train_hashes:
                    report["cross_split_state_leakage"] += 1

    # Status check
    critical_errors = (
        report["empty_states"]
        + report["empty_questions"]
        + report["nan_or_inf_detected"]
        + report["invalid_types"]
        + report["duplicate_states"]
        + report["cross_split_state_leakage"]
    )
    if critical_errors > 0:
        report["status"] = "FAILED"
    elif report["invalid_prob_sums"] > 0:
        report["status"] = "WARNING"
    else:
        report["status"] = "PASSED"

    if output_report_path:
        Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
