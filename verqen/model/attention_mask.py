"""Branch-aware attention mask for VerQen-T1, built from per-token segment metadata.

Every token carries four small integers:

    seg  segment type: PAD, BOS, STATE, QUESTION, OPTION, DECIDE
    doc  which example in the (packed) row the token belongs to, -1 for padding
    qid  question index within the example, -1 for BOS/STATE
    oid  option index within the question, -1 elsewhere

Rules (query -> key, always within the same ``doc``):

    BOS / STATE     -> BOS + whole state (bidirectional)
    QUESTION q      -> BOS + state + whole question q
    OPTION (q, o)   -> BOS + state + question q + whole option o
    DECIDE q        -> BOS + state + question q + all options of q + itself

So the state never sees any question (its representation is question-independent
and can be cached), questions never see each other, and options never see each
other inside the backbone. Options are compared later by the permutation-
equivariant set layers in the decision head.

Positions are branch-local (see ``verqen.data.packing``), so the answer to a
question does not depend on which other questions are packed next to it, nor on
the order its options are listed in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

SEG_PAD, SEG_BOS, SEG_STATE, SEG_QUESTION, SEG_OPTION, SEG_DECIDE = range(6)


def build_attention_mask(
    seg: torch.Tensor,
    doc: torch.Tensor,
    qid: torch.Tensor,
    oid: torch.Tensor,
    key_seg: torch.Tensor | None = None,
    key_doc: torch.Tensor | None = None,
    key_qid: torch.Tensor | None = None,
    key_oid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a boolean mask ``[B, 1, Lq, Lk]`` where ``True`` means *may attend*.

    Query metadata are ``[B, Lq]``. Key metadata default to the query metadata;
    pass them separately when attending over a cached prefix.
    """
    self_attention = key_seg is None
    if self_attention:
        key_seg, key_doc, key_qid, key_oid = seg, doc, qid, oid

    sq, sk = seg[:, :, None], key_seg[:, None, :]
    same_doc = (doc[:, :, None] == key_doc[:, None, :]) & (doc[:, :, None] >= 0)
    same_q = qid[:, :, None] == key_qid[:, None, :]
    same_o = oid[:, :, None] == key_oid[:, None, :]

    key_ctx = (sk == SEG_BOS) | (sk == SEG_STATE)
    query_branch = (sq == SEG_QUESTION) | (sq == SEG_OPTION) | (sq == SEG_DECIDE)

    branch = same_q & (
        ((sk == SEG_QUESTION) & query_branch)
        | ((sk == SEG_OPTION) & (((sq == SEG_OPTION) & same_o) | (sq == SEG_DECIDE)))
        | ((sk == SEG_DECIDE) & (sq == SEG_DECIDE))
    )
    allowed = same_doc & ((key_ctx & (sq != SEG_PAD)) | branch)

    if self_attention:
        # Padding queries attend to themselves so no softmax row is empty.
        L = seg.size(1)
        eye = torch.eye(L, dtype=torch.bool, device=seg.device)[None]
        allowed = allowed | (eye & (sq == SEG_PAD))
    return allowed[:, None]


# ── Legacy position metadata (kept for API compatibility) ────────────────────
@dataclass
class QuestionMetadata:
    q_start: int
    q_end: int
    options: list[tuple[int, int]]
    decide_pos: int


@dataclass
class SequenceMetadata:
    total_length: int
    bos_pos: int
    state_start: int
    state_end: int
    questions: list[QuestionMetadata] = field(default_factory=list)


def build_verqen_attention_mask(packed, device: torch.device | str = "cpu") -> torch.Tensor:
    """Mask ``[1, 1, L, L]`` for a single ``PackedExample``."""
    t = lambda xs: torch.tensor([xs], dtype=torch.long, device=device)
    doc = [0 if s != SEG_PAD else -1 for s in packed.seg]
    return build_attention_mask(t(packed.seg), t(doc), t(packed.qid), t(packed.oid))
