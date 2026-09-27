"""Top-level Python API.

    from verqen import Verqen
    vq = Verqen.from_pretrained("runs/verqen-t1-v2/release")
    vq.decide(state="...", question="...", options=[...])
"""

from __future__ import annotations

from typing import Any

from verqen.inference.engine import VerqenInferenceEngine


class Verqen:
    def __init__(self, engine: VerqenInferenceEngine) -> None:
        self.engine = engine

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: str = "auto", risk: str | None = None,
                        risk_mode: str = "marginal", fast: bool = False, risk_domain: str | None = None) -> "Verqen":
        """risk_mode="worst_group" certifies the risk level on every calibration source (high stakes);
        fast=True compiles the GPU path (CUDA graphs over shape buckets)."""
        return cls(VerqenInferenceEngine.from_checkpoint(checkpoint_dir, device, risk, risk_mode=risk_mode, fast=fast,
                                                        risk_domain=risk_domain))

    def decide(self, state: str, question: str, options: list[str] | None = None, question_type: str = "choice",
               cost: dict[str, float] | None = None) -> dict:
        return self.engine.decide(state, question, options, question_type, cost=cost)

    def decide_multi(self, state: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.engine.decide_multi(state, questions)

    def decide_many(self, requests: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        return self.engine.decide_many(requests)

    def decide_request(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.engine.decide_request(request)

    def encode_state(self, state: str):
        return self.engine.encode_state(state)

    def decide_on_state(self, handle, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.engine.decide_on_state(handle, questions)
