"""Synthetic Tool Orchestration Environment as per Section 24 and 82."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from verqen.environments.base import Environment


class ToolEnv(Environment):
    """Synthetic environment for tool dispatch, parameter validation, and error recovery."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.goal = "Analyze quarterly revenue trends from ERP database and format chart"
        self.available_toolset = ["sql_query", "python_calculator", "chart_generator", "web_search", "email_client"]
        self.workflow_stage = "start"
        self.done = False
        self.total_r = 0.0

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.rng = random.Random(seed)
        self.workflow_stage = "start"
        self.done = False
        self.total_r = 0.0
        return self.observe()

    def observe(self) -> str:
        return (
            f"Agent Tool Orchestrator: Goal='{self.goal}'. "
            f"Stage='{self.workflow_stage}'. Available tools: {self.available_toolset}."
        )

    def valid_actions(self) -> List[str]:
        if self.workflow_stage == "start":
            return ["call_sql_query", "call_web_search", "call_email_client"]
        elif self.workflow_stage == "data_retrieved":
            return ["call_python_calculator", "call_chart_generator", "retry_sql"]
        elif self.workflow_stage == "chart_ready":
            return ["verify_results", "call_email_client", "abort_workflow"]
        return ["finish"]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        reward = 0.0

        if self.workflow_stage == "start":
            if action == "call_sql_query":
                self.workflow_stage = "data_retrieved"
                reward = 1.0
            else:
                reward = -0.5

        elif self.workflow_stage == "data_retrieved":
            if action == "call_chart_generator":
                self.workflow_stage = "chart_ready"
                reward = 1.5
            elif action == "call_python_calculator":
                reward = 0.5
            else:
                reward = -0.2

        elif self.workflow_stage == "chart_ready":
            if action == "verify_results":
                reward = 2.0
                self.done = True
            else:
                reward = 0.0
                self.done = True

        self.total_r += reward
        return self.observe(), reward, self.done, {"stage": self.workflow_stage}

    def is_terminal(self) -> bool:
        return self.done

    def reward(self) -> float:
        return self.total_r
