"""Synthetic Browser Environment as per Section 22 and 85."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from verqen.environments.base import Environment


class BrowserEnv(Environment):
    """Synthetic browser automation environment with DOM elements, forms, and navigation."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.current_url = "https://portal.enterprise.io/login"
        self.page_state = "login_page"
        self.elements = ["#input-username", "#input-password", "#btn-submit", "#link-forgot-password"]
        self.done = False
        self.accumulated_reward = 0.0
        self.step_count = 0

    def reset(self, seed: Optional[int] = None) -> str:
        if seed is not None:
            self.rng = random.Random(seed)
        self.current_url = "https://portal.enterprise.io/login"
        self.page_state = "login_page"
        self.elements = ["#input-username", "#input-password", "#btn-submit", "#link-forgot-password"]
        self.done = False
        self.accumulated_reward = 0.0
        self.step_count = 0
        return self.observe()

    def observe(self) -> str:
        return (
            f"Browser [URL: {self.current_url}]: State='{self.page_state}'. "
            f"Interactive elements visible: {self.elements}."
        )

    def valid_actions(self) -> List[str]:
        if self.page_state == "login_page":
            return ["fill_credentials", "click_submit", "click_forgot_password", "reload_page"]
        elif self.page_state == "dashboard":
            return ["click_settings", "click_reports", "click_logout", "export_csv"]
        elif self.page_state == "modal_dialog":
            return ["confirm_modal", "cancel_modal", "close_modal"]
        return ["navigate_home", "back"]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0

        if self.page_state == "login_page":
            if action == "fill_credentials":
                reward = 0.5
            elif action == "click_submit":
                self.page_state = "dashboard"
                self.current_url = "https://portal.enterprise.io/dashboard"
                self.elements = ["#btn-settings", "#btn-reports", "#btn-logout", "#btn-export"]
                reward = 1.0
            else:
                reward = -0.2

        elif self.page_state == "dashboard":
            if action in ("click_reports", "export_csv"):
                reward = 2.0
                self.done = True
            elif action == "click_logout":
                self.page_state = "login_page"
                reward = 0.1
            else:
                reward = -0.1

        if self.step_count >= 10:
            self.done = True

        self.accumulated_reward += reward
        return self.observe(), reward, self.done, {"state": self.page_state, "url": self.current_url}

    def is_terminal(self) -> bool:
        return self.done

    def reward(self) -> float:
        return self.accumulated_reward
