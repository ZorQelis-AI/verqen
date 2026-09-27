"""Base Verifier Interface and VerificationResult dataclass."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from verqen.data.schema import SyntheticExample, VerificationStatus


@dataclass
class VerificationResult:
    """Standard verification result as per Section 11."""
    status: VerificationStatus
    confidence: float
    expected_answer: Optional[str]
    evidence: str
    verifier_name: str
    verifier_version: str = "1.0.0"

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED


class Verifier(ABC):
    """Abstract base class for all deterministic ground-truth verifiers."""
    name: str = "base_verifier"
    version: str = "1.0.0"

    @abstractmethod
    def verify(self, example: SyntheticExample) -> VerificationResult:
        """Verify ground truth of a synthetic example."""
        pass
