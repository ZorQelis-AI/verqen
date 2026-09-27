"""Interactive Synthetic Environments for ZorQelis VerQen-T1.

Provides composable environment interfaces for games, browser navigation,
spatial navigation, and tool orchestration trajectories.
"""

from verqen.environments.base import Environment
from verqen.environments.games.gridworld import GridWorldEnv
from verqen.environments.browser.browser_env import BrowserEnv
from verqen.environments.tools.tool_env import ToolEnv
from verqen.environments.navigation.graph_nav import GraphNavEnv

__all__ = [
    "Environment",
    "GridWorldEnv",
    "BrowserEnv",
    "ToolEnv",
    "GraphNavEnv",
]
