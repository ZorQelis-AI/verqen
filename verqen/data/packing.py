"""Pack decision examples into token ids + per-token structural metadata.

Sequence layout of one example::

    <BOS> <STATE> state... </STATE>
      <Q> <TYPE> question... </Q> <OPT> option... </OPT> ... <DECIDE>     (per question)

Positions are *branch-local*: every question starts at the same position (right
after ``</STATE>``) and every option of a question starts at the same position
(right after ``</Q>``). Combined with the branch mask this makes a question's
answer independent of the other questions in the sequence and of option order.

Canonical example schema (v2)::

    {"id": str, "source": str, "family": str,
     "state": str,
     "questions": [{"type": "choice"|"score"|"noul", "text": str,
                    "options": [str, ...], "target": [float, ...]}]}

``noul`` questions always get the options ``["no", "yes"]`` and target
``[P(false), P(true)]``. The legacy flat schema (``question``/``options``/``target``
at top level) is also accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verqen.model.attention_mask import SEG_BOS, SEG_DECIDE, SEG_OPTION, SEG_QUESTION, SEG_STATE
from verqen.model.decision_head import QTYPE_IDS
from verqen.tokenizer import TYPE_TOKEN, VerqenTokenizer

NOUL_OPTIONS = ["no", "yes"]
MAX_QUESTION_TOKENS = 128
MAX_OPTION_TOKENS = 64
MIN_STATE_TOKENS = 16


@dataclass
class TargetInfo:
    question_type: str
    target_probs: list[float]
    options: list[str] = field(default_factory=list)
    label: str | None = None


@dataclass
class PackedQuestion:
    qtype: int
    n_opts: int
    decide_idx: int
    target: list[float]


@dataclass
class PackedExample:
    ids: list[int]
    seg: list[int]
    qid: list[int]
    oid: list[int]
    pos: list[int]
    questions: list[PackedQuestion]
    targets: list[TargetInfo]
    example_id: str = ""

    def __len__(self) -> int:
        return len(self.ids)


def normalize_example(example: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Return (state, questions) in the canonical schema from v2 or legacy input."""
    state = example.get("state", example.get("state_text", "")) or ""
    if isinstance(example.get("questions"), list):
        raw_qs = example["questions"]
    else:
        tgt = example.get("target")
        if isinstance(tgt, dict) and "probabilities" in tgt:
            target = tgt["probabilities"]
        else:
            target = example.get("target_distribution", example.get("target_probs", tgt))
        raw_qs = [{
            "type": example.get("question_type", example.get("type", "choice")),
            "text": example.get("question_text", example.get("question", "")),
            "options": example.get("options", []),
            "target": target,
        }]
    qs = []
    for q in raw_qs:
        qtype = str(q.get("type", q.get("question_type", "choice"))).lower()
        text = q.get("text", q.get("question_text", "")) or ""
        opts = [str(o) for o in (q.get("options") or [])]
        raw = q.get("target", q.get("target_probs", q.get("target_distribution")))
        if qtype == "noul":
            p_true = _noul_p_true(raw, opts)
            opts, target = list(NOUL_OPTIONS), ([1.0 - p_true, p_true] if p_true is not None else None)
        else:
            target = _align_target(raw, opts)
        qs.append({"type": qtype, "text": text, "options": opts, "target": target})
    return state, qs


def _noul_p_true(raw: Any, opts: list[str]) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, dict):
        for k in ("true", "yes", True, "True", "Yes"):
            if k in raw:
                return float(raw[k])
        if "false" in raw or "no" in raw:
            return 1.0 - float(raw.get("false", raw.get("no")))
        return None
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1:
            return float(raw[0])
        if len(raw) == 2:
            return float(raw[1])  # canonical [P(false), P(true)]
    return None


def _align_target(raw: Any, opts: list[str]) -> list[float] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        probs = [float(raw.get(o, 0.0)) for o in opts]
    elif isinstance(raw, (list, tuple)):
        probs = [float(x) for x in raw][: len(opts)]
        probs += [0.0] * (len(opts) - len(probs))
    else:
        return None
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else None


class SequencePacker:
    """Packs examples into token ids with branch-local positions and segment metadata."""

    def __init__(self, tokenizer: VerqenTokenizer, max_seq_len: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.sp = tokenizer.special_token_ids

    def pack_many(self, examples: list[dict[str, Any]], require_targets: bool = True) -> list[PackedExample]:
        """Pack many examples, tokenizing every distinct string in one batched (Rust, parallel) call."""
        norm = [normalize_example(e) for e in examples]
        texts = set()
        for state, qs in norm:
            texts.add(state)
            for q in qs:
                texts.add(q["text"])
                texts.update(q["options"])
        texts = list(texts)
        cache = dict(zip(texts, self.tokenizer.encode_batch(texts)))
        return [self.pack(e, require_targets, _normalized=n, _enc=cache.__getitem__) for e, n in zip(examples, norm)]

    def pack(self, example: dict[str, Any], require_targets: bool = True, _normalized=None, _enc=None) -> PackedExample:
        state, questions = _normalized or normalize_example(example)
        sp = self.sp
        enc = _enc or self.tokenizer.encode

        branches = []
        for q in questions:
            if not q["options"]:
                continue
            if require_targets and q["target"] is None:
                continue
            q_ids = enc(q["text"])[:MAX_QUESTION_TOKENS]
            opt_ids = [enc(o)[:MAX_OPTION_TOKENS] for o in q["options"]]
            cost = 3 + len(q_ids) + sum(2 + len(o) for o in opt_ids) + 1
            branches.append((q, q_ids, opt_ids, cost))

        # Keep as many questions as fit next to a minimal state.
        budget = self.max_seq_len - 3 - MIN_STATE_TOKENS
        kept, used = [], 0
        for b in branches:
            if used + b[3] > budget:
                continue
            kept.append(b)
            used += b[3]

        state_ids = enc(state)
        room = self.max_seq_len - 3 - used
        if len(state_ids) > room:
            tail = min(64, room // 4)
            state_ids = state_ids[: room - tail] + (state_ids[-tail:] if tail else [])

        ids, seg, qid, oid, pos = [], [], [], [], []

        def put(tok, s, q=-1, o=-1, p=0):
            ids.append(tok); seg.append(s); qid.append(q); oid.append(o); pos.append(p)

        put(sp["bos"], SEG_BOS, p=0)
        put(sp["state_start"], SEG_STATE, p=1)
        for i, t in enumerate(state_ids):
            put(t, SEG_STATE, p=2 + i)
        put(sp["state_end"], SEG_STATE, p=2 + len(state_ids))
        base = 3 + len(state_ids)

        packed_qs, targets = [], []
        for qi, (q, q_ids, opt_ids, _) in enumerate(kept):
            p = base
            put(sp["q_start"], SEG_QUESTION, qi, -1, p); p += 1
            put(sp[q["type"] if q["type"] in sp else "choice"], SEG_QUESTION, qi, -1, p); p += 1
            for t in q_ids:
                put(t, SEG_QUESTION, qi, -1, p); p += 1
            put(sp["q_end"], SEG_QUESTION, qi, -1, p); p += 1
            opt_base = p
            for oi, o_ids in enumerate(opt_ids):
                p = opt_base
                put(sp["opt_start"], SEG_OPTION, qi, oi, p); p += 1
                for t in o_ids:
                    put(t, SEG_OPTION, qi, oi, p); p += 1
                put(sp["opt_end"], SEG_OPTION, qi, oi, p)
            put(sp["decide"], SEG_DECIDE, qi, -1, opt_base)
            target = q["target"] if q["target"] is not None else [1.0 / len(opt_ids)] * len(opt_ids)
            packed_qs.append(PackedQuestion(QTYPE_IDS.get(q["type"], 0), len(opt_ids), len(ids) - 1, target))
            label = q["options"][max(range(len(target)), key=target.__getitem__)]
            targets.append(TargetInfo(q["type"], target, q["options"], label))

        return PackedExample(ids, seg, qid, oid, pos, packed_qs, targets, str(example.get("id", "")))


# Type tokens must exist in the tokenizer under the names used above.
assert set(TYPE_TOKEN) == {"choice", "score", "noul"}
