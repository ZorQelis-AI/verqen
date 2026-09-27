"""Adapters: one interface, `decide(state, question, options) -> (selected, confidence, meta)`.

* verqen    local checkpoint through the compiled fast engine (production path).
* jev       TypeSafe's Jev over HTTP. Its request schema is not public beyond "choice / score /
            noul with calibrated probabilities", so the endpoint, auth header and field names are
            read from the environment (JEV_URL, JEV_API_KEY, JEV_*_FIELD) instead of being
            guessed. Fill them in from the early-access docs; nothing here assumes a schema silently.
* anthropic a frontier LLM asked for a JSON decision + confidence (requires `anthropic` and
            ANTHROPIC_API_KEY); used as the "frontier" reference point.
"""

from __future__ import annotations

import json
import os
import re
import time


class VerqenAdapter:
    local_gpu = True

    def __init__(self, checkpoint: str):
        from verqen.inference.engine import VerqenInferenceEngine
        self.engine = VerqenInferenceEngine.from_checkpoint(checkpoint, "auto", fast=True)
        self.name = f"verqen:{checkpoint}"
        self.engine.decide("warmup state", "warmup?", ["a", "b"])

    def decide(self, state, question, options):
        r = self.engine.decide(state, question, options)
        return r["selected"], max(r["probabilities"].values()), {"p_correct": r["p_correct"], "escalate": r["should_escalate"]}

    def cost_per_million(self, avg_chars: float) -> float:
        # measured-price basis documented in verqen/evaluation/suite.py ASSUMPTIONS (local GPU $/h over measured throughput)
        return round(0.80 / 3600 / 1000 * 1e6, 2)  # cloud L4-class $0.80/h at ~1,000 decisions/s


class LayaAdapter:
    """Convai Innovations' Laya (open weights, Apache 2.0), through its own `laya` package.

    Runs in the isolated venv Desktop/Language_Models/laya_env (the package is not installed in the
    VerQen environment). Every case is posed as a `choice` question with the options as criteria,
    exactly as VerQen receives it. Laya caps question + options at `head_max_len` (192 tokens by
    default); wide option sets (Banking77, MASSIVE) get the smallest cap that fits, up to half the
    context, and a case that still cannot be encoded is scored wrong and flagged `invalid`.
    Confidence = Laya's `answer_confidence` (the calibrated quantity its docs say ECE measures).
    """

    local_gpu = True

    def __init__(self, model: str | None = None):
        import laya
        repo, _, sub = (model or "convaiinnovations/laya").partition(":")
        self.agent = laya.load(repo, device="cuda", subfolder=sub or None)
        self.max_len = int(self.agent.cfg.get("max_len", 512))
        self.name = f"laya:{repo}{':' + sub if sub else ''}"
        self.agent.predict("warmup state", {"q": {"type": "choice", "instructions": "warmup?", "criteria": ["a", "b"]}})

    def decide(self, state, question, options):
        q = {"q": {"type": "choice", "instructions": question, "criteria": list(options)}}
        last = None
        for head in (None, 256, 384, self.max_len // 2):
            try:
                a = self.agent.predict(state, q, head_max_len=head)["answers"]["q"]
                return a["choice"], float(a["answer_confidence"]), {"head_max_len": head}
            except ValueError as e:  # options exceed head_max_len: retry with a larger budget
                last = e
        return None, 0.0, {"invalid": True, "error": str(last)[:120]}


class JevAdapter:
    def __init__(self):
        self.url = os.environ.get("JEV_URL")
        self.key = os.environ.get("JEV_API_KEY")
        if not self.url or not self.key:
            raise SystemExit("jev adapter: set JEV_URL and JEV_API_KEY (and JEV_*_FIELD if the schema differs)")
        self.f_state = os.environ.get("JEV_STATE_FIELD", "state")
        self.f_question = os.environ.get("JEV_QUESTION_FIELD", "question")
        self.f_options = os.environ.get("JEV_OPTIONS_FIELD", "options")
        self.f_selected = os.environ.get("JEV_SELECTED_FIELD", "selected")
        self.f_conf = os.environ.get("JEV_CONFIDENCE_FIELD", "confidence")
        self.name = "jev"

    def decide(self, state, question, options):
        import urllib.request
        body = json.dumps({self.f_state: state, self.f_question: question, self.f_options: options, "type": "choice"}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                                                   "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            out = json.loads(resp.read())
        return out[self.f_selected], float(out.get(self.f_conf, 0.0)), {}

    def cost_per_million(self, avg_chars: float) -> float:
        return round(avg_chars / 4 * 0.042, 2)  # $0.042 per M input tokens (published), output free


class JevOpenRouterAdapter:
    """The real Jev (TypeSafe System One), through OpenRouter's Decisions endpoint.

    `typesafe/jev-1.13` refuses chat/completions ("a decisions model"); it is served at
    POST /api/alpha/decisions with {model, state, questions}, the same typed schema Laya mirrors:
    a choice question carries `criteria` {option: description}. The response names the served
    snapshot (typesafe/jev-1.13-20260917), per-option probabilities and the billed cost, which
    is recorded per call so the cost column is measured, not estimated. Key: OPENROUTER_API_KEY.
    `typesafe/jev-router` is NOT used: it routes the prompt to other LLMs (a test call was
    answered by stealth/space-bunny-alpha), so its answers are not Jev's.
    """
    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model: str | None = None):
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise SystemExit("jev_openrouter adapter: set OPENROUTER_API_KEY")
        self.model = model or "typesafe/jev-1.13"
        self.name = f"openrouter:{self.model}"

    def decide(self, state, question, options):
        import urllib.error
        import urllib.request
        body = json.dumps({"model": self.model, "state": state,
                           "questions": {"q": {"type": "choice", "instructions": question,
                                               "criteria": {o: o for o in options}}}}).encode()
        last = None
        for attempt in range(6):
            req = urllib.request.Request(self.URL, data=body, headers={"Content-Type": "application/json",
                                                                       "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    out = json.loads(resp.read())
                a = out["answers"]["q"]
                return a.get("choice"), float(a.get("confidence", 0.0)), {
                    "served_model": out.get("model"), "cost_usd": (out.get("usage") or {}).get("cost"),
                    "input_tokens": (out.get("usage") or {}).get("input_tokens")}
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:160]!r}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
            except Exception as e:  # network hiccup
                last = f"{type(e).__name__}: {str(e)[:160]}"
            time.sleep(min(30, 2 ** attempt))
        return None, 0.0, {"invalid": True, "error": last}

    def cost_per_million(self, avg_chars: float) -> float:
        return round(avg_chars / 4 * 0.042, 2)  # fallback only; run_api.py uses the billed cost


class AnthropicAdapter:
    PROMPT = ("Decide the case. Return only JSON: {{\"selected\": <one option exactly>, \"confidence\": <0..1>}}.\n\n"
              "STATE:\n{state}\n\nQUESTION: {question}\nOPTIONS: {options}")

    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def decide(self, state, question, options):
        msg = self.client.messages.create(model=self.model, max_tokens=100, messages=[{"role": "user", "content":
                                          self.PROMPT.format(state=state, question=question, options=json.dumps(options))}])
        text = msg.content[0].text
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        sel = d.get("selected") if d.get("selected") in options else None
        return sel, float(d.get("confidence", 0.0) or 0.0), {"raw": text[:200], "invalid": sel is None,
                                                               "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens}

    def cost_per_million(self, avg_chars: float) -> float:
        return round((avg_chars / 4 * 3.0 + 20 * 15.0), 2)  # $3 / $15 per M tokens (assumed; edit per model)


def make_adapter(kind: str, checkpoint: str | None = None, model: str | None = None):
    if kind == "verqen":
        if not checkpoint:
            raise SystemExit("--checkpoint is required for the verqen adapter")
        return VerqenAdapter(checkpoint)
    if kind == "jev":
        return JevAdapter()
    if kind == "jev_openrouter":
        return JevOpenRouterAdapter(model)
    if kind == "laya":
        return LayaAdapter(model)
    if kind == "anthropic":
        return AnthropicAdapter(model or "claude-sonnet-5")
    raise SystemExit(f"unknown adapter {kind}")
