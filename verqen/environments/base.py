"""Abstract Base Environment as per Section 88."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class Environment(ABC):
    """Universal composable environment interface for synthetic trajectories."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> str:
        """Reset environment to initial state and return observation."""
        pass

    @abstractmethod
    def observe(self) -> str:
        """Return current textual observation of the state."""
        pass

    @abstractmethod
    def valid_actions(self) -> List[str]:
        """Return list of valid actions in current state."""
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Take an action. Returns (next_state_observation, reward, is_done, info)."""
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if episode has terminated."""
        pass

    @abstractmethod
    def reward(self) -> float:
        """Return cumulative or step reward."""
        pass
