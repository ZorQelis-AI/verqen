"""Shared helpers for procedural decision families.

Design rules every family follows (these are what the old generator violated):

1. The answer is *computed* from the generated state (search, simulation, rule
   evaluation, program execution), never looked up from a fixed string.
2. Option pools are shared: every option string appears as a wrong answer about
   as often as a right one, so the option set alone carries no label information.
3. Label-first generation where possible: sample the answer uniformly, then build
   a state that implies it.
4. Ties are real: when several options are equally optimal the target spreads
   probability over all of them (genuine soft labels).
5. Each family has several phrasing variants. Variant 0 is reserved for the
   held-out "unseen phrasing" test split.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

FIRST_NAMES = [
    "Aarav", "Maya", "Liam", "Sofia", "Kenji", "Amara", "Noah", "Priya", "Lucas", "Zara", "Omar", "Elena",
    "Mateo", "Aisha", "Hugo", "Mei", "Ravi", "Chloe", "Tariq", "Ingrid", "Diego", "Yuki", "Kofi", "Hana",
    "Felix", "Nadia", "Arjun", "Lena", "Samir", "Ivy", "Jonas", "Leila", "Marco", "Rosa", "Theo", "Anika",
]
CITIES = [
    "Avalon", "Brisk", "Cedar", "Dunmore", "Eastvale", "Fairport", "Glenrock", "Harbor", "Ironwood", "Juniper",
    "Kestrel", "Lakeside", "Marlow", "Northgate", "Oakridge", "Pinecrest", "Quarry", "Riverton", "Stonebrook",
    "Thornfield", "Upland", "Valemont", "Westmoor", "Yarrow", "Zenith",
]
ITEMS = [
    "lamp", "kettle", "backpack", "monitor", "headphones", "chair", "notebook", "blender", "jacket", "router",
    "camera", "tent", "bicycle", "keyboard", "printer", "umbrella", "watch", "speaker", "desk", "mug",
]
COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "black", "white", "silver", "teal"]


def pick(rng: random.Random, seq: Sequence, k: int | None = None):
    return rng.choice(seq) if k is None else rng.sample(list(seq), k)


def choice_q(text: str, options: list[str], target: list[float] | int, rng: random.Random, qtype: str = "choice",
             shuffle: bool = True) -> dict[str, Any]:
    """Build a question; ``target`` is an index or a distribution aligned with ``options``."""
    if isinstance(target, int):
        t = [0.0] * len(options)
        t[target] = 1.0
    else:
        s = sum(target)
        t = [x / s for x in target]
    idx = list(range(len(options)))
    if shuffle and qtype != "score":
        rng.shuffle(idx)
    return {"type": qtype, "text": text, "options": [options[i] for i in idx], "target": [t[i] for i in idx]}


def noul_q(text: str, value: bool) -> dict[str, Any]:
    return {"type": "noul", "text": text, "options": ["no", "yes"], "target": [0.0, 1.0] if value else [1.0, 0.0]}


def uniform_over(options: list[str], good: set[str]) -> list[float]:
    return [1.0 if o in good else 0.0 for o in options]


def numeric_options(correct: int | float, rng: random.Random, k: int = 4, spread: int | None = None,
                    fmt=lambda v: str(v), extra: Sequence | None = None) -> tuple[list[str], int]:
    """Correct value plus distractors that share its granularity, with the correct value's
    *rank* among the options uniform (so "pick the middle/largest value" is not a shortcut).
    ``extra`` holds plausible error-model values (e.g. forgot the discount) used first."""
    if isinstance(correct, float) and correct == round(correct):
        step = 1.0
    elif isinstance(correct, float) and (correct * 2) == round(correct * 2):
        step = 0.5
    elif isinstance(correct, float):
        step = 0.01
    else:
        step = 1
    spread = spread or max(3, int(abs(correct) * 0.3) + 2)
    n_below = rng.randrange(k)
    vals = [correct]
    for e in (extra or []):
        if len(vals) >= k:
            break
        e = round(e / step) * step if step != 1 else int(round(e))
        if e != correct and e not in vals:
            below = sum(v < correct for v in vals)
            if (e < correct and below < n_below) or (e > correct and len(vals) - 1 - below < k - 1 - n_below):
                vals.append(e)
    guard = 0
    while len(vals) < k and guard < 500:
        guard += 1
        below = sum(v < correct for v in vals)
        sign = -1 if below < n_below else 1
        delta = rng.randint(1, max(1, int(spread / step))) * step
        cand = correct + sign * delta
        cand = round(cand, 2) if step != 1 else int(cand)
        if cand not in vals:
            vals.append(cand)
    rng.shuffle(vals)
    return [fmt(v) for v in vals], vals.index(correct)


def example(family: str, state: str, questions: list[dict], variant: int) -> dict[str, Any]:
    return {"source": "procedural", "family": family, "variant": variant, "state": state, "questions": questions}
