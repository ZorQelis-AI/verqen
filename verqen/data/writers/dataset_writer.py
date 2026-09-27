"""Writes synthetic examples to JSONL with dataset manifests and quality reports."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

from verqen.data.schema import SyntheticExample


class DatasetWriter:
    """Handles JSONL dataset serialization, manifests, and quality audit reports."""

    @staticmethod
    def write_jsonl(examples: List[SyntheticExample], output_path: str) -> int:
        """Write examples to JSONL file. Returns count written."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def generate_manifest(
        dataset_name: str,
        version: str,
        examples: List[SyntheticExample],
        output_path: str,
        seed: int = 42,
        generator_version: str = "0.3.0",
    ) -> Dict[str, Any]:
        """Generate dataset_manifest.json as per Section 23."""
        domain_counts = Counter(ex.domain for ex in examples)
        task_counts = Counter(ex.task_family for ex in examples)
        type_counts = Counter(ex.question_type.value for ex in examples)
        diff_counts = Counter(ex.difficulty.value for ex in examples)
        verifier_counts = Counter(ex.ground_truth_method.value for ex in examples)

        manifest = {
            "dataset_name": dataset_name,
            "version": version,
            "num_examples": len(examples),
            "generated_at": datetime.now().isoformat(),
            "generator_version": generator_version,
            "seed": seed,
            "domains": dict(sorted(domain_counts.items())),
            "task_types": dict(sorted(type_counts.items())),
            "difficulty": dict(sorted(diff_counts.items())),
            "task_families": dict(sorted(task_counts.items())),
            "verifier_distribution": dict(sorted(verifier_counts.items())),
            "adversarial_count": sum(1 for ex in examples if ex.is_adversarial),
            "counterfactual_count": sum(1 for ex in examples if ex.counterfactual_meta is not None),
            "trajectory_count": sum(1 for ex in examples if ex.trajectory is not None),
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    @staticmethod
    def generate_report(
        train_examples: List[SyntheticExample],
        val_examples: List[SyntheticExample],
        test_examples: List[SyntheticExample],
        output_path: str,
    ) -> Dict[str, Any]:
        """Generate dataset_report.json with quality audit and cross-split leakage metrics."""
        train_hashes = set(ex.state for ex in train_examples)
        val_hashes = set(ex.state for ex in val_examples)
        test_hashes = set(ex.state for ex in test_examples)

        train_val_leakage = len(train_hashes.intersection(val_hashes))
        train_test_leakage = len(train_hashes.intersection(test_hashes))
        val_test_leakage = len(val_hashes.intersection(test_hashes))

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "PASSED" if (train_val_leakage == 0 and train_test_leakage == 0) else "FAILED",
            "splits": {
                "train_count": len(train_examples),
                "val_count": len(val_examples),
                "test_count": len(test_examples),
                "total_count": len(train_examples) + len(val_examples) + len(test_examples),
            },
            "leakage_audit": {
                "train_val_leakage": train_val_leakage,
                "train_test_leakage": train_test_leakage,
                "val_test_leakage": val_test_leakage,
                "zero_leakage_verified": (train_val_leakage == 0 and train_test_leakage == 0),
            },
            "uniqueness": {
                "train_unique_states": len(train_hashes),
                "val_unique_states": len(val_hashes),
                "test_unique_states": len(test_hashes),
            },
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
