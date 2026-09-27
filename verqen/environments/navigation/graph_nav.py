"""Spatial Graph Navigation Environment as per Section 13 and 88."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from verqen.environments.base import Environment


class GraphNavEnv(Environment):
    """Synthetic topological graph navigation environment with weighted paths."""

    def __init__(self, n_nodes: int = 6, seed: int = 42) -> None:
        self.n_nodes = n_nodes
        self.rng = random.Random(seed)
        self.current_node = 0
        self.target_node = n_nodes - 1
        # Linear/mesh graph with connections
        self.adj: Dict[int, List[int]] = {
            i: [(i + 1) % n_nodes, (i + 2) % n_nodes] for i in range(n_nodes)
        }
        self.done = False
        self.total_cost = 0.0

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_node = 0
        self.done = False
        self.total_cost = 0.0
        return self.observe()

    def observe(self) -> str:
        neighbors = self.adj.get(self.current_node, [])
        return (
            f"GraphNav: CurrentNode={self.current_node}, TargetNode={self.target_node}. "
            f"Connected accessible neighbors: {neighbors}."
        )

    def valid_actions(self) -> List[str]:
        neighbors = self.adj.get(self.current_node, [])
        return [f"goto_node_{n}" for n in neighbors]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        target_n = int(action.split("_")[-1])
        if target_n in self.adj.get(self.current_node, []):
            self.current_node = target_n
            reward = 1.0 if self.current_node == self.target_node else -0.1
        else:
            reward = -1.0

        if self.current_node == self.target_node:
            self.done = True
            reward += 3.0

        self.total_cost += reward
        return self.observe(), reward, self.done, {"node": self.current_node}

    def is_terminal(self) -> bool:
        return self.done

    def reward(self) -> float:
        return self.total_cost
