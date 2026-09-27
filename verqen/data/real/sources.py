"""Real-world decision datasets converted to the VerQen v2 schema.

Rules:
* Training draws ONLY from official train splits.
* Evaluation uses official validation/test splits that never enter training.
* Every loader returns ``{"id", "source", "family", "state", "questions": [...]}``.

MMLU is evaluation-only (its ``auxiliary_train`` duplicates ARC/OBQA/RACE).
"""

from __future__ import annotations

import random
import re
import zlib
from typing import Callable, Iterator

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


def _ex(source, idx, state, question, options, answer_idx, qtype="choice", family=None):
    options = [str(o).strip() for o in options]
    if qtype == "noul":
        target = [0.0, 1.0] if answer_idx else [1.0, 0.0]
        q = {"type": "noul", "text": question, "options": ["no", "yes"], "target": target}
    else:
        if not (0 <= answer_idx < len(options)) or len(set(options)) != len(options) or len(options) < 2:
            return None
        target = [0.0] * len(options)
        target[answer_idx] = 1.0
        q = {"type": qtype, "text": question, "options": options, "target": target}
    return {"id": f"{source}_{idx}", "source": source, "family": family or source, "state": state.strip(), "questions": [q]}


def _letter_idx(key, labels=None):
    key = str(key).strip()
    if labels and key in labels:
        return labels.index(key)
    if key.isdigit():
        return int(key) - 1
    return LETTERS.index(key.upper()) if key.upper() in LETTERS else -1


def race(split):
    for i, r in enumerate(load_dataset("ehovy/race", "all", split=split)):
        yield _ex("race", i, r["article"], r["question"], r["options"], _letter_idx(r["answer"]), family="reading")


def mnli(split):
    labels = ["entailment", "neutral", "contradiction"]
    for i, r in enumerate(load_dataset("nyu-mll/glue", "mnli", split=split)):
        if r["label"] < 0:
            continue
        yield _ex("mnli", i, f"Text: {r['premise']}\nHow does the hypothesis below relate to the text?",
                  f"Hypothesis: {r['hypothesis']}", labels, r["label"], family="inference")


def boolq(split):
    for i, r in enumerate(load_dataset("google/boolq", split=split)):
        q = r["question"].strip()
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        yield _ex("boolq", i, r["passage"], q, [], int(r["answer"]), qtype="noul", family="reading")


def hellaswag(split):
    for i, r in enumerate(load_dataset("Rowan/hellaswag", split=split)):
        if r["label"] in ("", None):
            continue
        yield _ex("hellaswag", i, f"{r['activity_label']}: {r['ctx']}", "What most plausibly happens next?",
                  r["endings"], int(r["label"]), family="commonsense")


def winogrande(split):
    for i, r in enumerate(load_dataset("allenai/winogrande", "winogrande_xl", split=split)):
        if r["answer"] not in ("1", "2"):
            continue
        yield _ex("winogrande", i, r["sentence"], "Which option correctly fills the blank (_)?",
                  [r["option1"], r["option2"]], int(r["answer"]) - 1, family="commonsense")


def csqa(split):
    for i, r in enumerate(load_dataset("tau/commonsense_qa", split=split)):
        if not r["answerKey"]:
            continue
        labels = r["choices"]["label"]
        yield _ex("csqa", i, "", r["question"], r["choices"]["text"], _letter_idx(r["answerKey"], labels), family="commonsense")


def piqa(split):
    for i, r in enumerate(load_dataset("baber/piqa", split=split)):
        if r["label"] not in (0, 1):
            continue
        yield _ex("piqa", i, f"Goal: {r['goal']}", "Which solution achieves the goal?", [r["sol1"], r["sol2"]],
                  int(r["label"]), family="physical")


def siqa(split):
    for i, r in enumerate(load_dataset("lighteval/siqa", split=split)):
        yield _ex("siqa", i, r["context"], r["question"], [r["answerA"], r["answerB"], r["answerC"]],
                  int(r["label"]) - 1, family="social")


def sciq(split):
    rng = random.Random(7)
    for i, r in enumerate(load_dataset("allenai/sciq", split=split)):
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        order = list(range(4))
        rng.shuffle(order)
        yield _ex("sciq", i, r["support"] or "", r["question"], [opts[j] for j in order], order.index(0), family="science")


def arc(split, subset):
    name = "arc_easy" if subset == "ARC-Easy" else "arc_challenge"
    for i, r in enumerate(load_dataset("allenai/ai2_arc", subset, split=split)):
        labels = r["choices"]["label"]
        yield _ex(name, i, "", r["question"], r["choices"]["text"], _letter_idx(r["answerKey"], labels), family="science")


def obqa(split):
    for i, r in enumerate(load_dataset("allenai/openbookqa", "main", split=split)):
        labels = r["choices"]["label"]
        yield _ex("obqa", i, "", r["question_stem"], r["choices"]["text"], _letter_idx(r["answerKey"], labels), family="science")


def qasc(split):
    for i, r in enumerate(load_dataset("allenai/qasc", split=split)):
        if not r["answerKey"]:
            continue
        labels = r["choices"]["label"]
        yield _ex("qasc", i, f"{r['fact1']} {r['fact2']}", r["question"], r["choices"]["text"],
                  _letter_idx(r["answerKey"], labels), family="science")


def quartz(split):
    for i, r in enumerate(load_dataset("allenai/quartz", split=split)):
        labels = r["choices"]["label"]
        yield _ex("quartz", i, r["para"], r["question"], r["choices"]["text"], _letter_idx(r["answerKey"], labels), family="science")


def medmcqa(split):
    for i, r in enumerate(load_dataset("openlifescienceai/medmcqa", split=split)):
        if r.get("choice_type") != "single" or r["cop"] is None or r["cop"] < 0:
            continue
        yield _ex("medmcqa", i, f"Medicine ({r['subject_name']})", r["question"],
                  [r["opa"], r["opb"], r["opc"], r["opd"]], int(r["cop"]), family="medical")


def aqua(split):
    for i, r in enumerate(load_dataset("deepmind/aqua_rat", "raw", split=split)):
        opts = [re.sub(r"^[A-E]\)\s*", "", o) for o in r["options"]]
        yield _ex("aqua", i, "", r["question"], opts, _letter_idx(r["correct"]), family="math")


def mmlu(split):
    for i, r in enumerate(load_dataset("cais/mmlu", "all", split=split)):
        yield _ex("mmlu", i, f"Subject: {r['subject'].replace('_', ' ')}", r["question"], r["choices"], int(r["answer"]),
                  family="knowledge")


# name -> (train loader, train cap, eval loader, eval cap)
SOURCES: dict[str, tuple[Callable, int | None, Callable | None, int | None]] = {
    "race":          (lambda: race("train"), None, lambda: race("test"), 2000),
    "mnli":          (lambda: mnli("train"), 150_000, lambda: mnli("validation_matched"), 2000),
    "boolq":         (lambda: boolq("train"), None, lambda: boolq("validation"), None),
    "hellaswag":     (lambda: hellaswag("train"), None, lambda: hellaswag("validation"), 2000),
    "winogrande":    (lambda: winogrande("train"), None, lambda: winogrande("validation"), None),
    "csqa":          (lambda: csqa("train"), None, lambda: csqa("validation"), None),
    "piqa":          (lambda: piqa("train"), None, lambda: piqa("validation"), None),
    "siqa":          (lambda: siqa("train"), None, lambda: siqa("validation"), None),
    "sciq":          (lambda: sciq("train"), None, lambda: sciq("test"), None),
    "arc_easy":      (lambda: arc("train", "ARC-Easy"), None, lambda: arc("test", "ARC-Easy"), None),
    "arc_challenge": (lambda: arc("train", "ARC-Challenge"), None, lambda: arc("test", "ARC-Challenge"), None),
    "obqa":          (lambda: obqa("train"), None, lambda: obqa("test"), None),
    "qasc":          (lambda: qasc("train"), None, None, None),
    "quartz":        (lambda: quartz("train"), None, lambda: quartz("test"), None),
    "medmcqa":       (lambda: medmcqa("train"), 40_000, lambda: medmcqa("validation"), 2000),
    "aqua":          (lambda: aqua("train"), 20_000, lambda: aqua("test"), None),
    "mmlu":          (None, None, lambda: mmlu("test"), None),
}


def _take(it: Iterator, cap: int | None, seed: int) -> list[dict]:
    rows = [r for r in it if r is not None]
    if cap is not None and len(rows) > cap:
        rng = random.Random(seed)
        rows = rng.sample(rows, cap)
    return rows


def load_split(which: str, names: list[str] | None = None, log: Callable = print) -> dict[str, list[dict]]:
    """Load ``which`` in {"train", "eval"} for the given sources (default: all)."""
    out = {}
    for name, (tr, tr_cap, ev, ev_cap) in SOURCES.items():
        if names and name not in names:
            continue
        loader, cap = (tr, tr_cap) if which == "train" else (ev, ev_cap)
        if loader is None:
            continue
        try:
            out[name] = _take(loader(), cap, seed=zlib.crc32(name.encode()))
            log(f"[real] {which:5s} {name:14s} {len(out[name]):>7,}")
        except Exception as e:  # keep going if one hub dataset is unavailable
            log(f"[real] {which:5s} {name:14s} FAILED: {type(e).__name__}: {str(e)[:120]}")
    return out


def iter_all_train_text(max_per_source: int = 20000) -> Iterator[str]:
    for name, rows in load_split("train", log=lambda *_: None).items():
        for r in rows[:max_per_source]:
            yield r["state"]
            for q in r["questions"]:
                yield q["text"] + " " + " ".join(q["options"])
