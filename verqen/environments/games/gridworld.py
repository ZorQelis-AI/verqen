"""Deterministic GridWorld Environment for Game & Spatial Trajectories."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from verqen.environments.base import Environment


class GridWorldEnv(Environment):
    """Grid navigation world with deterministic transitions and shortest-path rewards."""

    def __init__(self, size: int = 5, n_obstacles: int = 3, seed: int = 42) -> None:
        self.size = size
        self.n_obstacles = n_obstacles
        self.seed = seed
        self.rng = random.Random(seed)
        self.agent_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.obstacles: set[Tuple[int, int]] = set()
        self.steps_taken = 0
        self.max_steps = size * 4
        self.done = False
        self.current_reward = 0.0
        self.reset(seed)

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.rng = random.Random(seed)
        self.agent_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.obstacles.clear()
        
        # Place random obstacles (avoiding start and goal)
        while len(self.obstacles) < self.n_obstacles:
            ox = self.rng.randint(0, self.size - 1)
            oy = self.rng.randint(0, self.size - 1)
            if (ox, oy) != (0, 0) and (ox, oy) != self.goal_pos:
                self.obstacles.add((ox, oy))
                
        self.steps_taken = 0
        self.done = False
        self.current_reward = 0.0
        return self.observe()

    def observe(self) -> str:
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        obs_desc = f"GridWorld [{self.size}x{self.size}]: Agent at ({ax}, {ay}), Goal at ({gx}, {gy}). "
        obs_desc += f"Obstacles at: {list(self.obstacles)}. Steps remaining: {self.max_steps - self.steps_taken}."
        return obs_desc

    def valid_actions(self) -> List[str]:
        actions = []
        ax, ay = self.agent_pos
        if ax > 0 and (ax - 1, ay) not in self.obstacles:
            actions.append("move_up")
        if ax < self.size - 1 and (ax + 1, ay) not in self.obstacles:
            actions.append("move_down")
        if ay > 0 and (ax, ay - 1) not in self.obstacles:
            actions.append("move_left")
        if ay < self.size - 1 and (ax, ay + 1) not in self.obstacles:
            actions.append("move_right")
        return actions if actions else ["wait"]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        if self.done:
            return self.observe(), 0.0, True, {"reason": "already_done"}

        self.steps_taken += 1
        ax, ay = self.agent_pos
        old_dist = abs(ax - self.goal_pos[0]) + abs(ay - self.goal_pos[1])

        if action == "move_up" and ax > 0:
            next_pos = (ax - 1, ay)
        elif action == "move_down" and ax < self.size - 1:
            next_pos = (ax + 1, ay)
        elif action == "move_left" and ay > 0:
            next_pos = (ax, ay - 1)
        elif action == "move_right" and ay < self.size - 1:
            next_pos = (ax, ay + 1)
        else:
            next_pos = self.agent_pos

        if next_pos in self.obstacles:
            reward = -0.5
        else:
            self.agent_pos = next_pos
            new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            # Reward shaping based on distance reduction
            reward = 0.2 if new_dist < old_dist else -0.1

        if self.agent_pos == self.goal_pos:
            reward += 5.0
            self.done = True
        elif self.steps_taken >= self.max_steps:
            reward -= 1.0
            self.done = True

        self.current_reward += reward
        return self.observe(), reward, self.done, {"position": self.agent_pos}

    def is_terminal(self) -> bool:
        return self.done

    def reward(self) -> float:
        return self.current_reward
