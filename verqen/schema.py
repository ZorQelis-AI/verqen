from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import math

class QuestionType(Enum):
    CHOICE = "CHOICE"
    NOUL = "NOUL"
    SCORE = "SCORE"

@dataclass
class Question:
    id: str
    type: QuestionType
    text: str
    options: Optional[List[str]] = None

@dataclass
class DecisionRequest:
    id: str
    state: str
    questions: List[Question]

    def validate(self) -> None:
        if not self.state:
            raise ValueError("State cannot be empty")
        if not self.questions:
            raise ValueError("Questions cannot be empty")
        
        for q in self.questions:
            if not isinstance(q.type, QuestionType):
                try:
                    q.type = QuestionType(q.type)
                except ValueError:
                    raise ValueError(f"Invalid question type: {q.type}")
            
            if q.type == QuestionType.CHOICE or q.type == QuestionType.SCORE:
                if not q.options or len(q.options) < 2 or len(q.options) > 64:
                    raise ValueError(f"{q.type.value} requires 2 to 64 options, got {len(q.options) if q.options else 0}")
                if len(set(q.options)) != len(q.options):
                    raise ValueError("Options must be unique")
            elif q.type == QuestionType.NOUL:
                if q.options is not None and len(q.options) > 0:
                    raise ValueError("NOUL questions should not have options")

@dataclass
class TargetDistribution:
    probabilities: Dict[str, float]
    label: Optional[str] = None

    def validate(self) -> None:
        if not self.probabilities:
            raise ValueError("Probabilities dictionary cannot be empty")
        total_prob = sum(self.probabilities.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-5):
            raise ValueError(f"Probabilities must sum to 1.0, got {total_prob}")
        for p in self.probabilities.values():
            if p < 0.0 or p > 1.0:
                raise ValueError("Probabilities must be between 0.0 and 1.0")

@dataclass
class DecisionExample:
    id: str
    domain: str
    difficulty: str
    type: QuestionType
    state: str
    question: str
    options: Optional[List[str]]
    target: TargetDistribution
    metadata: Dict[str, Any] = field(default_factory=dict)
