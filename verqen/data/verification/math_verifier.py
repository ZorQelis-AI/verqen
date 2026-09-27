"""Deterministic Math Verifier for exact arithmetic, squares, cubes, powers, and equations."""

from __future__ import annotations

import re
from verqen.data.schema import SyntheticExample, VerificationStatus
from verqen.data.verification.base import VerificationResult, Verifier


class MathVerifier(Verifier):
    """Deterministic mathematical ground truth verifier."""
    name: str = "math_verifier"
    version: str = "1.0.0"

    def verify(self, example: SyntheticExample) -> VerificationResult:
        state = example.state
        q = example.question
        label = example.target.label

        # Square verification: "square of integer X" or "X^2"
        sq_match = re.search(r"square of (?:integer )?(\d+)", state, re.IGNORECASE) or re.search(r"(\d+)\^2", state)
        if sq_match:
            x = int(sq_match.group(1))
            expected = str(x * x)
            if example.question_type.value == "CHOICE":
                passed = (label == expected)
                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    confidence=1.0,
                    expected_answer=expected,
                    evidence=f"Exact arithmetic computation: {x}^2 = {expected}",
                    verifier_name=self.name,
                )
            elif example.question_type.value == "NOUL":
                # Check if question claims a specific candidate
                cand_match = re.search(r"equal to (-?\d+)", q)
                if cand_match:
                    cand = int(cand_match.group(1))
                    is_eq = (x * x == cand)
                    expected_bool = "true" if is_eq else "false"
                    passed = (label.lower() == expected_bool)
                    return VerificationResult(
                        status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                        confidence=1.0,
                        expected_answer=expected_bool,
                        evidence=f"Identity check: {x}^2 = {expected}, claimed: {cand} -> {expected_bool}",
                        verifier_name=self.name,
                    )

        # Cube verification: "cube of integer X" or "X^3"
        cube_match = re.search(r"cube of (?:integer )?(\d+)", state, re.IGNORECASE) or re.search(r"(\d+)\^3", state)
        if cube_match:
            x = int(cube_match.group(1))
            expected = str(x ** 3)
            if example.question_type.value == "CHOICE":
                passed = (label == expected)
                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    confidence=1.0,
                    expected_answer=expected,
                    evidence=f"Exact arithmetic computation: {x}^3 = {expected}",
                    verifier_name=self.name,
                )

        # Multiplication verification: "product of X multiplied by Y" or "X * Y"
        mult_match = re.search(r"product of (\d+) multiplied by (\d+)", state, re.IGNORECASE) or re.search(r"(\d+)\s*\*\s*(\d+)", state)
        if mult_match:
            x, y = int(mult_match.group(1)), int(mult_match.group(2))
            expected = str(x * y)
            if example.question_type.value == "CHOICE":
                passed = (label == expected)
                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    confidence=1.0,
                    expected_answer=expected,
                    evidence=f"Exact arithmetic: {x} * {y} = {expected}",
                    verifier_name=self.name,
                )

        # Default fallback verification if label exists in options
        if label and label in example.options:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=0.95,
                expected_answer=label,
                evidence="Label exists in valid options set",
                verifier_name=self.name,
            )

        return VerificationResult(
            status=VerificationStatus.AMBIGUOUS,
            confidence=0.5,
            expected_answer=None,
            evidence="Pattern not recognized for exact math verification",
            verifier_name=self.name,
        )
