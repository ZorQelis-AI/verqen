"""Splits dataset according to IID, Domain OOD, Difficulty OOD, and Calibration holdout policies."""

from __future__ import annotations

import random
from enum import Enum
from typing import Dict, List, Optional, Set

from verqen.data.schema import SyntheticExample


class SplitStrategy(str, Enum):
    """Splitting strategy as per Section 21 & 38."""
    IID = "iid"
    DOMAIN_OOD = "domain_ood"
    DIFFICULTY_OOD = "difficulty_ood"
    ADVERSARIAL_OOD = "adversarial_ood"


class DatasetSplitter:
    """Partitions examples into train, validation, calibration, and test splits."""

    def __init__(
        self,
        strategy: SplitStrategy = SplitStrategy.IID,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
        holdout_domains: Optional[List[str]] = None,
    ) -> None:
        self.strategy = strategy
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.rng = random.Random(seed)
        self.holdout_domains = set(holdout_domains or [])

    def split(
        self,
        examples: List[SyntheticExample],
        calibration_size: int = 0,
    ) -> Dict[str, List[SyntheticExample]]:
        """Split examples according to configured strategy."""
        splits: Dict[str, List[SyntheticExample]] = {
            "train": [],
            "val": [],
            "calibration": [],
            "test": [],
        }

        if self.strategy == SplitStrategy.DOMAIN_OOD and self.holdout_domains:
            # Training only on non-holdout domains
            for ex in examples:
                if ex.domain in self.holdout_domains:
                    splits["test"].append(ex)
                else:
                    if self.rng.random() < self.train_ratio:
                        splits["train"].append(ex)
                    else:
                        splits["val"].append(ex)

        elif self.strategy == SplitStrategy.DIFFICULTY_OOD:
            # Hard difficulties (d4+) go into test
            for ex in examples:
                diff = ex.difficulty.value
                if diff in ("d4_constraint", "d5_counterfactual", "d6_planning", "d7_multi_domain"):
                    splits["test"].append(ex)
                else:
                    if self.rng.random() < 0.85:
                        splits["train"].append(ex)
                    else:
                        splits["val"].append(ex)

        elif self.strategy == SplitStrategy.ADVERSARIAL_OOD:
            # All adversarial cases held out for test
            for ex in examples:
                if ex.is_adversarial:
                    splits["test"].append(ex)
                else:
                    if self.rng.random() < self.train_ratio:
                        splits["train"].append(ex)
                    else:
                        splits["val"].append(ex)

        else:
            # Standard IID Split with group-level integrity (parent and variants stay together)
            grouped: Dict[str, List[SyntheticExample]] = {}
            for ex in examples:
                base_id = ex.id.split('_adv')[0].split('_cf')[0]
                grouped.setdefault(base_id, []).append(ex)

            keys = list(grouped.keys())
            self.rng.shuffle(keys)
            n_keys = len(keys)
            n_val = int(n_keys * self.val_ratio)
            n_test = int(n_keys * self.test_ratio)
            n_train = n_keys - n_val - n_test

            for k in keys[:n_train]:
                splits["train"].extend(grouped[k])
            for k in keys[n_train : n_train + n_val]:
                splits["val"].extend(grouped[k])
            for k in keys[n_train + n_val :]:
                splits["test"].extend(grouped[k])

        # Extract calibration holdout if requested (Section 38)
        if calibration_size > 0 and len(splits["train"]) > calibration_size:
            splits["calibration"] = splits["train"][:calibration_size]
            splits["train"] = splits["train"][calibration_size:]

        return splits
