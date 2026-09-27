"""Row-packing collator: several decision examples share one row of ``max_seq_len`` tokens.

Examples in the same row are separated by the ``doc`` id, which the attention
mask uses, and each keeps its own branch-local positions. So packing removes
padding waste without changing any example's computation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from verqen.data.packing import PackedExample
from verqen.model.attention_mask import SEG_OPTION, SEG_PAD


def pack_rows(examples: Sequence[PackedExample], max_len: int, max_rows: int | None = None):
    """First-fit-decreasing bin packing. Returns (rows, dropped) as lists of example lists."""
    order = sorted(range(len(examples)), key=lambda i: -len(examples[i]))
    rows: list[list[int]] = []
    free: list[int] = []
    dropped = []
    for i in order:
        n = len(examples[i])
        for r, f in enumerate(free):
            if n <= f:
                rows[r].append(i)
                free[r] -= n
                break
        else:
            if max_rows is not None and len(rows) >= max_rows:
                dropped.append(i)
                continue
            rows.append([i])
            free.append(max_len - n)
    return [[examples[i] for i in row] for row in rows], [examples[i] for i in dropped]


def collate_rows(rows: list[list[PackedExample]], max_len: int, pad_id: int, n_rows: int | None = None) -> dict[str, Any]:
    """Vectorized: builds each field for the whole batch with numpy, then one tensor per field."""
    R = max(n_rows or 0, len(rows))
    L = max_len
    ids = np.full((R, L), pad_id, dtype=np.int64)
    seg = np.full((R, L), SEG_PAD, dtype=np.int64)
    pos = np.zeros((R, L), dtype=np.int64)
    doc = np.full((R, L), -1, dtype=np.int64)
    qid = np.full((R, L), -1, dtype=np.int64)
    oid = np.full((R, L), -1, dtype=np.int64)
    tok_gid = np.full((R * L,), -1, dtype=np.int64)

    dec_flat, bos_flat, qtypes, n_opts, targets, opt_first = [], [], [], [], [], []
    target_infos = []
    gid = 0
    for r, row in enumerate(rows):
        if not row:
            continue
        lens = [len(ex) for ex in row]
        n = sum(lens)
        ids[r, :n] = np.fromiter((t for ex in row for t in ex.ids), dtype=np.int64, count=n)
        seg[r, :n] = np.fromiter((t for ex in row for t in ex.seg), dtype=np.int64, count=n)
        pos[r, :n] = np.fromiter((t for ex in row for t in ex.pos), dtype=np.int64, count=n)
        q_row = np.fromiter((t for ex in row for t in ex.qid), dtype=np.int64, count=n)
        o_row = np.fromiter((t for ex in row for t in ex.oid), dtype=np.int64, count=n)
        qid[r, :n], oid[r, :n] = q_row, o_row
        doc[r, :n] = np.repeat(np.arange(len(row)), lens)
        # global option id of every OPTION token = first gid of its (example, question) + oid
        first_of_q = np.full(n, -1, dtype=np.int64)
        off = 0
        for ex, ln in zip(row, lens):
            base = []
            for q in ex.questions:
                base.append(gid)
                dec_flat.append(r * L + off + q.decide_idx)
                bos_flat.append(r * L + off)
                qtypes.append(q.qtype)
                n_opts.append(q.n_opts)
                targets.append(q.target)
                opt_first.append(gid)
                gid += q.n_opts
            if base:
                qq = q_row[off:off + ln]
                valid = qq >= 0
                first_of_q[off:off + ln][valid] = np.asarray(base)[qq[valid]]
            target_infos.append(ex.targets)
            off += ln
        is_opt = seg[r, :n] == SEG_OPTION
        tok_gid[r * L + np.nonzero(is_opt)[0]] = first_of_q[is_opt] + o_row[is_opt]

    N = len(dec_flat)
    O = max(n_opts) if n_opts else 1
    ar = np.arange(O)[None]
    nop = np.asarray(n_opts, dtype=np.int64)[:, None] if N else np.zeros((0, 1), np.int64)
    opt_table = np.where(ar < nop, np.asarray(opt_first, dtype=np.int64)[:, None] + ar, -1) if N else np.zeros((0, O), np.int64)
    target = np.zeros((N, O), dtype=np.float32)
    for i, tg in enumerate(targets):
        target[i, : len(tg)] = tg
    t = torch.from_numpy
    opt_table_t = t(opt_table)
    return {
        "input_ids": t(ids), "seg": t(seg), "pos": t(pos), "doc": t(doc), "qid": t(qid), "oid": t(oid),
        "tok_opt_gid": t(tok_gid), "n_opt_total": torch.tensor(gid),
        "dec_flat": torch.tensor(dec_flat, dtype=torch.long),
        "bos_flat": torch.tensor(bos_flat, dtype=torch.long),
        "qtype": torch.tensor(qtypes, dtype=torch.long),
        "opt_table": opt_table_t, "opt_mask": opt_table_t >= 0, "target": t(target),
        "targets": target_infos,
    }


class VerqenCollator:
    """DataLoader collate_fn: packs a list of dataset items into fixed-shape rows."""

    def __init__(self, pad_token_id: int, max_seq_len: int = 512, n_rows: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.n_rows = n_rows

    def __call__(self, batch: list[PackedExample]) -> dict[str, Any]:
        rows, dropped = pack_rows(batch, self.max_seq_len, self.n_rows)
        out = collate_rows(rows, self.max_seq_len, self.pad_token_id, self.n_rows)
        out["n_dropped"] = len(dropped)
        return out


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
