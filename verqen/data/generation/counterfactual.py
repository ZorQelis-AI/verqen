"""Counterfactual Generator creating paired examples with causal intervention (Section 29, 30)."""

from __future__ import annotations

import copy
from typing import Optional, Tuple
from verqen.data.schema import CounterfactualMetadata, ExperienceType, SyntheticExample


class CounterfactualGenerator:
    """Generates paired counterfactuals with controlled single-variable interventions."""

    @staticmethod
    def generate_pair(
        example: SyntheticExample,
        counterfactual_id: str,
    ) -> Optional[SyntheticExample]:
        """Produce a counterfactual variant of an existing example."""
        # Mutate based on domain or question type
        cf = copy.deepcopy(example)
        cf.id = counterfactual_id
        cf.experience_type = ExperienceType.TYPE_C_COUNTERFACTUAL

        state = cf.state
        if "charge" in state.lower() or "amount" in state.lower():
            # Flip high amount to low amount or vice-versa
            mod_state = state.replace("twice", "once").replace("unauthorized", "pre-authorized")
            if mod_state == state:
                mod_state = f"{state} [Intervention: authorization status set to pre-authorized]"
            cf.state = mod_state
            old_label = str(cf.target.label)
            new_label = "allow" if "deny" in old_label else ("waive" if "fee" in old_label else "no_action")
            cf.target.label = new_label
            # Rebalance probabilities to favor new label
            if new_label in cf.options:
                cf.target.probabilities = {
                    opt: (0.85 if opt == new_label else round(0.15 / max(1, len(cf.options) - 1), 4))
                    for opt in cf.options
                }
            cf.counterfactual_meta = CounterfactualMetadata(
                parent_example_id=example.id,
                counterfactual_type="authorization_reversal",
                changed_variable="authorization_status",
                old_value="unauthorized",
                new_value="pre-authorized",
                expected_direction="decrease_risk",
            )
            return cf

        elif "delay" in state.lower() or "days" in state.lower():
            mod_state = state.replace("delayed", "on_schedule")
            if mod_state == state:
                mod_state = f"{state} [Intervention: shipment status set to on_schedule]"
            cf.state = mod_state
            cf.counterfactual_meta = CounterfactualMetadata(
                parent_example_id=example.id,
                counterfactual_type="schedule_reversal",
                changed_variable="shipment_status",
                old_value="delayed",
                new_value="on_schedule",
                expected_direction="decrease_escalation",
            )
            return cf

        return None
