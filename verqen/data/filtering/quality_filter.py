"""Quality Filter verifying schema compliance, target probabilities, and no label leakage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from verqen.data.schema import QuestionType, SyntheticExample


@dataclass
class QualityReport:
    """Report produced by QualityFilter."""
    total_evaluated: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted / self.total_evaluated) if self.total_evaluated > 0 else 0.0


class QualityFilter:
    """Rigorous gate preventing flawed or leaked examples from entering datasets."""

    def __init__(
        self,
        min_options: int = 2,
        max_options: int = 64,
        allow_unresolved: bool = False,
    ) -> None:
        self.min_options = min_options
        self.max_options = max_options
        self.allow_unresolved = allow_unresolved
        self.report = QualityReport()

    def evaluate(self, example: SyntheticExample) -> Tuple[bool, Optional[str]]:
        """Validate an example. Returns (passed, reason_if_failed)."""
        self.report.total_evaluated += 1

        # 1. State and question presence
        if not example.state or len(example.state.strip()) < 5:
            return self._reject("state_too_short")
        if not example.question or len(example.question.strip()) < 3:
            return self._reject("question_too_short")

        # 2. Options validation
        if example.question_type in (QuestionType.CHOICE, QuestionType.SCORE, QuestionType.RANK):
            if not example.options:
                return self._reject("missing_options")
            if len(example.options) < self.min_options:
                return self._reject("too_few_options")
            if len(example.options) > self.max_options:
                return self._reject("too_many_options")
            if len(set(example.options)) != len(example.options):
                return self._reject("duplicate_options")

        # 3. Probability distribution validation
        probs = example.target.probabilities
        if not probs:
            return self._reject("missing_target_probabilities")

        prob_sum = sum(probs.values())
        if abs(prob_sum - 1.0) > 1e-3:
            return self._reject("probabilities_do_not_sum_to_1")

        for k, p in probs.items():
            if p < 0.0 or p > 1.0:
                return self._reject("probability_out_of_bounds")

        # 4. Label leakage check (Section 92)
        if example.target.label:
            lbl = str(example.target.label).strip()
            if len(lbl) > 3:
                leak_phrases = [
                    f"the answer is {lbl.lower()}",
                    f"correct answer is {lbl.lower()}",
                    f"the label is {lbl.lower()}",
                    f"choose option '{lbl.lower()}'",
                ]
                state_lower = example.state.lower()
                for phrase in leak_phrases:
                    if phrase in state_lower:
                        return self._reject("label_leakage_detected")

        # 5. Verification status check
        if not self.allow_unresolved and example.verification_status.value == "failed":
            return self._reject("ground_truth_verification_failed")

        self.report.accepted += 1
        return True, None

    def _reject(self, reason: str) -> Tuple[bool, Optional[str]]:
        self.report.rejected += 1
        self.report.rejection_reasons[reason] = self.report.rejection_reasons.get(reason, 0) + 1
        return False, reason
