"""Curriculum Scheduler controlling difficulty pacing, horizon scaling, and domain weighting."""

from __future__ import annotations

from typing import Dict, List, Optional
from verqen.data.schema import Difficulty, SyntheticExample


class CurriculumScheduler:
    """Controls progression of reasoning complexity across training epochs."""

    def __init__(
        self,
        curriculum_schedule: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> None:
        # Default curriculum: gradual ramp from direct (d0) to multi-domain reasoning (d7)
        self.schedule = curriculum_schedule or {
            1: {"d0_direct": 0.50, "d1_one_step": 0.35, "d2_two_step": 0.15},
            2: {"d0_direct": 0.25, "d1_one_step": 0.35, "d2_two_step": 0.25, "d3_multi_hop": 0.15},
            3: {"d1_one_step": 0.20, "d2_two_step": 0.30, "d3_multi_hop": 0.30, "d4_constraint": 0.20},
            4: {"d2_two_step": 0.15, "d3_multi_hop": 0.25, "d4_constraint": 0.30, "d5_counterfactual": 0.30},
            5: {"d3_multi_hop": 0.15, "d4_constraint": 0.25, "d5_counterfactual": 0.25, "d6_planning": 0.20, "d7_multi_domain": 0.15},
        }
        self.current_epoch: int = 1

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch

    def get_difficulty_weights(self, epoch: Optional[int] = None) -> Dict[str, float]:
        ep = epoch or self.current_epoch
        # Fallback to nearest defined epoch
        available_epochs = sorted(self.schedule.keys())
        if ep in self.schedule:
            return self.schedule[ep]
        elif ep > max(available_epochs):
            return self.schedule[max(available_epochs)]
        return self.schedule[min(available_epochs)]

    def filter_for_epoch(
        self,
        examples: List[SyntheticExample],
        epoch: Optional[int] = None,
    ) -> List[SyntheticExample]:
        """Filter/reweight examples for the current curriculum stage."""
        weights = self.get_difficulty_weights(epoch)
        allowed_difficulties = set(weights.keys())
        return [ex for ex in examples if ex.difficulty.value in allowed_difficulties]
