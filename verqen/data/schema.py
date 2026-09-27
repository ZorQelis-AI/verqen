"""Comprehensive data schemas for ZorQelis VerQen-T1.

Defines schemas for:
- 6 Experience Types (Static, Multi-Question, Counterfactual, Error Detection, Trajectory, Abstention)
- Task Families
- Ground Truth Hierarchy (Exact, Program, Simulator, Symbolic, Multi-Verifier, Teacher)
- Evidence Strengths & Soft Probability Targets
- Trajectories & Environment Interactions
- Provenance & Quality Metadata
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class QuestionType(str, Enum):
    """Supported decision question types."""
    CHOICE = "CHOICE"
    NOUL = "NOUL"
    SCORE = "SCORE"
    RANK = "RANK"
    ACTION = "ACTION"
    ABSTAIN = "ABSTAIN"


class ExperienceType(str, Enum):
    """6 Experience Types as per Section 7."""
    TYPE_A_STATIC = "static"
    TYPE_B_MULTI_QUESTION = "multi_question"
    TYPE_C_COUNTERFACTUAL = "counterfactual"
    TYPE_D_ERROR_DETECTION = "error_detection"
    TYPE_E_TRAJECTORY = "trajectory"
    TYPE_F_ABSTENTION = "abstention"


class Difficulty(str, Enum):
    """Reasoning curriculum levels (0 through 7)."""
    D0_DIRECT = "d0_direct"
    D1_ONE_STEP = "d1_one_step"
    D2_TWO_STEP = "d2_two_step"
    D3_MULTI_HOP = "d3_multi_hop"
    D4_CONSTRAINT = "d4_constraint"
    D5_COUNTERFACTUAL = "d5_counterfactual"
    D6_PLANNING = "d6_planning"
    D7_MULTI_DOMAIN = "d7_multi_domain"


class EvidenceStrength(int, Enum):
    """Latent evidence strength mapping for calibrated uncertainty."""
    VERY_STRONG_POSITIVE = 4
    STRONG_POSITIVE = 3
    POSITIVE = 2
    WEAK_POSITIVE = 1
    NEUTRAL = 0
    WEAK_NEGATIVE = -1
    NEGATIVE = -2
    STRONG_NEGATIVE = -3
    VERY_STRONG_NEGATIVE = -4


class GroundTruthMethod(str, Enum):
    """Hierarchy of ground-truth sources as per Section 10."""
    EXACT = "exact"
    PROGRAM_VERIFIED = "program_verified"
    SIMULATOR_VERIFIED = "simulator_verified"
    SYMBOLICALLY_VERIFIED = "symbolically_verified"
    MULTI_VERIFIER = "multi_verifier"
    TEACHER_CONSENSUS = "teacher_consensus"
    SINGLE_TEACHER = "single_teacher"
    UNRESOLVED = "unresolved"


class VerificationStatus(str, Enum):
    """Verification outcome."""
    PASSED = "passed"
    FAILED = "failed"
    AMBIGUOUS = "ambiguous"
    UNVERIFIED = "unverified"


@dataclass
class ProvenanceMetadata:
    """Provenance tracking for data ethics and reproducibility."""
    generator_version: str = "0.3.0"
    schema_version: str = "1.0.0"
    seed: int = 42
    worker_seed: Optional[int] = None
    template_id: str = "default"
    verifier_type: str = "none"
    verifier_version: str = "1.0.0"
    teacher_metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class CounterfactualMetadata:
    """Metadata for paired counterfactual examples."""
    parent_example_id: str
    counterfactual_type: str
    changed_variable: str
    old_value: str
    new_value: str
    expected_direction: str  # e.g., "increase", "decrease", "flip_boolean", "invariant"


@dataclass
class StepRecord:
    """A single transition step in an interactive trajectory."""
    step_num: int
    state: str
    available_actions: List[str]
    action: str
    reward: float
    next_state: str
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryData:
    """Complete environment trajectory for agent / game / browser tasks."""
    trajectory_id: str
    environment: str
    goal: str
    steps: List[StepRecord]
    total_reward: float
    success: bool


@dataclass
class TargetDistribution:
    """Calibrated probability distribution over discrete options."""
    probabilities: Dict[str, float]
    label: Optional[str] = None

    def validate(self) -> None:
        if not self.probabilities:
            raise ValueError("Probabilities dictionary cannot be empty")
        total_prob = sum(self.probabilities.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
        for k, p in self.probabilities.items():
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Probability for '{k}' must be in [0.0, 1.0], got {p}")


@dataclass
class SyntheticExample:
    """Universal schema for all ZorQelis VerQen-T1 synthetic intelligence examples."""
    id: str
    domain: str
    task_family: str
    experience_type: ExperienceType
    difficulty: Difficulty
    question_type: QuestionType
    state: str
    question: str
    options: List[str]
    target: TargetDistribution
    ground_truth_method: GroundTruthMethod
    verification_status: VerificationStatus
    evidence_strength: EvidenceStrength = EvidenceStrength.STRONG_POSITIVE
    is_adversarial: bool = False
    adversarial_type: Optional[str] = None
    counterfactual_meta: Optional[CounterfactualMetadata] = None
    trajectory: Optional[TrajectoryData] = None
    sub_questions: Optional[List[Dict[str, Any]]] = None
    provenance: ProvenanceMetadata = field(default_factory=ProvenanceMetadata)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Enforce strict data quality rules as per Section 14."""
        if not self.id:
            raise ValueError("Example ID cannot be empty")
        if not self.domain:
            raise ValueError("Domain cannot be empty")
        if not self.state or not self.state.strip():
            raise ValueError("State text cannot be empty")
        if not self.question or not self.question.strip():
            raise ValueError("Question text cannot be empty")

        # Question type & option constraints
        if self.question_type in (QuestionType.CHOICE, QuestionType.SCORE, QuestionType.RANK):
            if not self.options or len(self.options) < 2:
                raise ValueError(f"{self.question_type} requires >= 2 options, got {len(self.options)}")
            if len(set(self.options)) != len(self.options):
                raise ValueError("Options must be unique")
        elif self.question_type == QuestionType.NOUL:
            # Binary True/False
            if self.options and len(self.options) not in (0, 2):
                raise ValueError("NOUL requires empty options or exact ['false', 'true']")

        # Target distribution check
        self.target.validate()

        # No label leakage check (Section 92)
        target_str = str(self.target.label).lower() if self.target.label else ""
        if target_str and len(target_str) > 3:
            # Check state doesn't blatantly state "The correct option is X"
            leak_phrases = [
                f"the answer is {target_str}",
                f"correct answer is {target_str}",
                f"the label is {target_str}",
            ]
            for phrase in leak_phrases:
                if phrase in self.state.lower():
                    raise ValueError(f"Label leakage detected in state: '{phrase}'")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize example to clean JSON-serializable dictionary."""
        d = asdict(self)
        # Ensure enums serialize as string values
        d["experience_type"] = self.experience_type.value
        d["difficulty"] = self.difficulty.value
        d["question_type"] = self.question_type.value
        d["ground_truth_method"] = self.ground_truth_method.value
        d["verification_status"] = self.verification_status.value
        d["evidence_strength"] = self.evidence_strength.value
        d["target"] = {
            "probabilities": self.target.probabilities,
            "label": self.target.label,
        }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticExample:
        """Deserialize from dictionary."""
        data_copy = dict(data)
        data_copy["experience_type"] = ExperienceType(data_copy["experience_type"])
        data_copy["difficulty"] = Difficulty(data_copy["difficulty"])
        data_copy["question_type"] = QuestionType(data_copy["question_type"])
        data_copy["ground_truth_method"] = GroundTruthMethod(data_copy["ground_truth_method"])
        data_copy["verification_status"] = VerificationStatus(data_copy["verification_status"])
        data_copy["evidence_strength"] = EvidenceStrength(data_copy.get("evidence_strength", 3))
        
        target_dict = data_copy["target"]
        data_copy["target"] = TargetDistribution(
            probabilities=target_dict["probabilities"],
            label=target_dict.get("label"),
        )
        
        if data_copy.get("counterfactual_meta"):
            data_copy["counterfactual_meta"] = CounterfactualMetadata(**data_copy["counterfactual_meta"])
        if data_copy.get("provenance"):
            data_copy["provenance"] = ProvenanceMetadata(**data_copy["provenance"])
        if data_copy.get("trajectory"):
            traj = data_copy["trajectory"]
            steps = [StepRecord(**s) for s in traj.get("steps", [])]
            data_copy["trajectory"] = TrajectoryData(
                trajectory_id=traj["trajectory_id"],
                environment=traj["environment"],
                goal=traj["goal"],
                steps=steps,
                total_reward=traj["total_reward"],
                success=traj["success"],
            )
            
        return cls(**data_copy)
