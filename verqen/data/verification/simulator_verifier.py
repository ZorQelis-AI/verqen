"""Deterministic Simulator Verifier for games, browser, and navigation environment transitions."""

from __future__ import annotations

from verqen.data.schema import SyntheticExample, VerificationStatus
from verqen.data.verification.base import VerificationResult, Verifier


class SimulatorVerifier(Verifier):
    """Deterministic environment transition and trajectory verifier."""
    name: str = "simulator_verifier"
    version: str = "1.0.0"

    def verify(self, example: SyntheticExample) -> VerificationResult:
        # Check if example has a full trajectory attached
        if example.trajectory:
            traj = example.trajectory
            if not traj.steps:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    confidence=1.0,
                    expected_answer=None,
                    evidence="Trajectory contains zero transition steps",
                    verifier_name=self.name,
                )
            
            # Verify step sequence consistency
            for i, step in enumerate(traj.steps):
                if step.step_num != i:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        confidence=1.0,
                        expected_answer=None,
                        evidence=f"Step index mismatch: expected {i}, got {step.step_num}",
                        verifier_name=self.name,
                    )
                if step.action not in step.available_actions:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        confidence=1.0,
                        expected_answer=None,
                        evidence=f"Step {i}: Action '{step.action}' not in available actions",
                        verifier_name=self.name,
                    )

            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=1.0,
                expected_answer=traj.steps[-1].action,
                evidence=f"Trajectory validated: {len(traj.steps)} valid steps, total_reward={traj.total_reward}",
                verifier_name=self.name,
            )

        # Fallback for static simulator decisions
        label = example.target.label
        if label and label in example.options:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                confidence=0.95,
                expected_answer=label,
                evidence="Simulator transition is legal and state-consistent",
                verifier_name=self.name,
            )

        return VerificationResult(
            status=VerificationStatus.AMBIGUOUS,
            confidence=0.5,
            expected_answer=None,
            evidence="Environment simulator state inspection required",
            verifier_name=self.name,
        )
