"""Phase-0 data audits: duplicates, cross-split contamination, and model-free shortcut baselines.

Every check is designed to catch a failure the v1 dataset actually had:

* ``exact_duplicates``: counts duplicates *after* stripping ID stamps (v1 stamped
  ``#vq_..`` into each state, which made 1.1% unique content look 100% unique).
* ``contamination``: exact state/question overlap and 13-gram overlap of
  question+options between training rows and evaluation rows.
* ``option_set_baseline``: predicts the label from the option set alone (v1: 96.2%).
* ``prior_baseline``: majority label per (family, question text).

``audit(train, evals)`` returns a report and ``passed`` flag with thresholds.
"""

from __future__ import annotations

import collections
import hashlib
import re
from typing import Any, Iterable

ID_PATTERNS = [re.compile(r"#vq_\d+_\d+"), re.compile(r"\bproc_[a-z_]+_\d+_\d+\b")]
WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    for p in ID_PATTERNS:
        text = p.sub("", text)
    return " ".join(text.lower().split())


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def content_key(ex: dict[str, Any]) -> str:
    parts = [normalize(ex.get("state", ""))]
    for q in ex["questions"]:
        parts.append(normalize(q["text"]))
        parts.append("|".join(sorted(normalize(o) for o in q["options"])))
    return _h("\x1f".join(parts))


def question_key(ex: dict[str, Any]) -> str | None:
    """Key for contamination: question + sorted options (ignores state phrasing)."""
    q = ex["questions"][0]
    text = normalize(q["text"])
    if len(text) < 20:  # generic prompts ("What happens next?") are not identifying
        text = normalize(ex.get("state", ""))[:300] + "|" + text
    return _h(text + "|" + "|".join(sorted(normalize(o) for o in q["options"])))


def ngrams(text: str, n: int = 13) -> set[int]:
    w = WORD.findall(text.lower())
    # a STABLE 64-bit hash: built-in hash() is salted per process, so a blocklist built in one
    # process never matched in a spawn/forkserver worker (the pretraining filter dropped 0 docs)
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + n]).encode(), digest_size=8).digest(), "little")
            for i in range(len(w) - n + 1)}


def field_grams(ex: dict[str, Any], n: int = 13) -> set[int]:
    """13-grams inside individual fields only (question text, each option), so shared
    templates plus a few content words never count as overlap."""
    q = ex["questions"][0]
    g = ngrams(q["text"], n)
    for o in q["options"]:
        g |= ngrams(o, n)
    return g


def exact_duplicates(rows: Iterable[dict]) -> dict[str, Any]:
    c = collections.Counter(content_key(r) for r in rows)
    total = sum(c.values())
    return {"rows": total, "unique": len(c), "duplicate_rows": total - len(c),
            "unique_ratio": len(c) / max(1, total), "max_repeat": max(c.values()) if c else 0}


def contamination(train: list[dict], evals: list[dict], n: int = 13) -> dict[str, Any]:
    eval_keys = {question_key(e) for e in evals}
    eval_states = {_h(normalize(e["state"])) for e in evals if len(e.get("state", "")) > 40}
    eval_grams: set[int] = set()
    for e in evals:
        eval_grams |= field_grams(e, n)
    hit_q = hit_s = hit_g = 0
    for t in train:
        if question_key(t) in eval_keys:
            hit_q += 1
        if len(t.get("state", "")) > 40 and _h(normalize(t["state"])) in eval_states:
            hit_s += 1
        if eval_grams and field_grams(t, n) & eval_grams:
            hit_g += 1
    return {"train_rows": len(train), "eval_rows": len(evals), "question_overlap": hit_q,
            "state_overlap": hit_s, f"{n}gram_overlap": hit_g}


def filter_contaminated(train: list[dict], evals: list[dict], n: int = 13) -> tuple[list[dict], int]:
    eval_keys = {question_key(e) for e in evals}
    eval_states = {_h(normalize(e["state"])) for e in evals if len(e.get("state", "")) > 40}
    eval_grams: set[int] = set()
    for e in evals:
        eval_grams |= field_grams(e, n)
    kept, removed = [], 0
    for t in train:
        bad = (question_key(t) in eval_keys
               or (len(t.get("state", "")) > 40 and _h(normalize(t["state"])) in eval_states)
               or bool(field_grams(t, n) & eval_grams))
        if bad:
            removed += 1
        else:
            kept.append(t)
    return kept, removed


def _label(q: dict) -> str:
    return q["options"][max(range(len(q["target"])), key=q["target"].__getitem__)]


def _correct(q: dict, pred: str) -> bool:
    best = max(q["target"])
    return any(o == pred and t >= best - 1e-9 for o, t in zip(q["options"], q["target"]))


def option_set_baseline(train: list[dict], test: list[dict]) -> float:
    """Accuracy of predicting the label from the (sorted) option set alone, learned on train."""
    table: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        for q in r["questions"]:
            table["|".join(sorted(q["options"]))][_label(q)] += 1
    hit = tot = 0
    for r in test:
        for q in r["questions"]:
            key = "|".join(sorted(q["options"]))
            if table.get(key):
                pred = table[key].most_common(1)[0][0]
            else:
                pred = q["options"][0]
            hit += _correct(q, pred)
            tot += 1
    return hit / max(1, tot)


def chance_accuracy(test: list[dict]) -> float:
    vals = [sum(1 for t in q["target"] if t >= max(q["target"]) - 1e-9) / len(q["options"])
            for r in test for q in r["questions"]]
    return sum(vals) / max(1, len(vals))


def prior_baseline(train: list[dict], test: list[dict]) -> float:
    """Majority label per (family, question text) — catches label-prior shortcuts."""
    table: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        for q in r["questions"]:
            table[(r.get("family"), normalize(q["text"]))][_label(q)] += 1
    hit = tot = 0
    for r in test:
        for q in r["questions"]:
            c = table.get((r.get("family"), normalize(q["text"])))
            pred = c.most_common(1)[0][0] if c else q["options"][0]
            hit += _correct(q, pred)
            tot += 1
    return hit / max(1, tot)


def audit(train: list[dict], tests: dict[str, list[dict]], max_shortcut_gain: float = 0.15) -> dict[str, Any]:
    """Run all checks. Fails when any shortcut beats chance by more than ``max_shortcut_gain``
    on a test split, when test rows repeat training content, or when duplicates dominate."""
    report: dict[str, Any] = {"train_duplicates": exact_duplicates(train)}
    train_keys = {content_key(r) for r in train}
    passed = report["train_duplicates"]["unique_ratio"] > 0.95
    for name, rows in tests.items():
        chance = chance_accuracy(rows)
        opt = option_set_baseline(train, rows)
        pri = prior_baseline(train, rows)
        leak = sum(content_key(r) in train_keys for r in rows)
        report[name] = {"rows": len(rows), "chance": round(chance, 4), "option_set_baseline": round(opt, 4),
                        "prior_baseline": round(pri, 4), "exact_train_overlap": leak}
        ok = leak <= 0.001 * max(1, len(rows)) and max(opt, pri) - chance <= max_shortcut_gain
        report[name]["passed"] = ok
        passed = passed and ok
    report["passed"] = passed
    return report
