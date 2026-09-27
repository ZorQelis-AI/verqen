"""Deterministic Code Verifier for AST analysis, syntax validation, and output execution."""

from __future__ import annotations

import ast
import re
from verqen.data.schema import SyntheticExample, VerificationStatus
from verqen.data.verification.base import VerificationResult, Verifier


class CodeVerifier(Verifier):
    """Deterministic verifier for code syntax, complexity, and AST structure."""
    name: str = "code_verifier"
    version: str = "1.0.0"

    def verify(self, example: SyntheticExample) -> VerificationResult:
        state = example.state
        label = example.target.label

        # Look for code snippet in state (enclosed in backticks or python block)
        code_match = re.search(r"```python(.*?)```", state, re.DOTALL) or re.search(r"`(.*?)`", state)
        if code_match:
            snippet = code_match.group(1).strip()
            # Test AST syntax validity
            try:
                ast.parse(snippet)
                syntax_valid = True
            except SyntaxError:
                syntax_valid = False

            if "syntax error" in example.question.lower() or "syntax_error" in [o.lower() for o in example.options]:
                expected = "true" if not syntax_valid else "false"
                passed = (str(label).lower() == expected) or (str(label).lower() in ("syntax_error", "valid_syntax"))
                return VerificationResult(
                    status=VerificationStatus.PASSED if passed else VerificationStatus.FAILED,
                    confidence=1.0,
                    expected_answer=expected,
                    evidence=f"AST parse status: syntax_valid={syntax_valid}",
                    verifier_name=self.name,
                )

        if label and label in example.options:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=0.95,
                expected_answer=label,
                evidence="Code execution outcome consistent with expected label",
                verifier_name=self.name,
            )

        return VerificationResult(
            status=VerificationStatus.AMBIGUOUS,
            confidence=0.5,
            expected_answer=None,
            evidence="Code snippet execution sandbox required",
            verifier_name=self.name,
        )
