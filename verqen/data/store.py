"""Compact storage for packed decision examples (numpy ragged arrays, ~7 bytes/token).

Python lists for ~1M packed examples would need tens of GB; this store keeps one
flat array per field plus offsets, and materializes ``PackedExample`` objects only
for the examples in the current batch.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable

import numpy as np

from verqen.data.packing import PackedExample, PackedQuestion, SequencePacker, TargetInfo
from verqen.tokenizer import VerqenTokenizer

FIELDS = {"ids": np.uint16, "seg": np.int8, "qid": np.int8, "oid": np.int8, "pos": np.int16}


class PackedStore:
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.a = arrays
        self.n = len(arrays["tok_off"]) - 1

    def __len__(self) -> int:
        return self.n

    def length(self, i: int) -> int:
        return int(self.a["tok_off"][i + 1] - self.a["tok_off"][i])

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.a["tok_off"])

    def __getitem__(self, i: int) -> PackedExample:
        a = self.a
        s, e = a["tok_off"][i], a["tok_off"][i + 1]
        qs, qe = a["q_off"][i], a["q_off"][i + 1]
        questions, targets = [], []
        for j in range(qs, qe):
            ts, te = a["t_off"][j], a["t_off"][j + 1]
            t = a["target"][ts:te].tolist()
            questions.append(PackedQuestion(int(a["qtype"][j]), int(a["nopt"][j]), int(a["dec"][j]), t))
            targets.append(TargetInfo(["choice", "score", "noul"][int(a["qtype"][j])], t))
        return PackedExample(
            a["ids"][s:e].astype(np.int64).tolist(), a["seg"][s:e].tolist(), a["qid"][s:e].tolist(),
            a["oid"][s:e].tolist(), a["pos"][s:e].tolist(), questions, targets, "",
        )

    def meta(self, i: int) -> dict:
        return {"source": self.a["source"][i], "family": self.a["family"][i]}

    def save(self, path: str) -> None:
        np.savez(path, **{k: v for k, v in self.a.items()})

    @classmethod
    def load(cls, path: str) -> "PackedStore":
        z = np.load(path, allow_pickle=True)
        return cls({k: z[k] for k in z.files})


def _pack_shard(args) -> dict[str, np.ndarray]:
    rows, tokenizer_path, max_len = args
    tok = VerqenTokenizer.from_pretrained(tokenizer_path)
    packer = SequencePacker(tok, max_len)
    buf = {k: [] for k in FIELDS}
    tok_off, q_off, t_off = [0], [0], [0]
    qtype, nopt, dec, target, source, family = [], [], [], [], [], []
    for r in rows:
        ex = packer.pack(r)
        if not ex.questions:
            continue
        for k in FIELDS:
            buf[k].append(np.asarray(getattr(ex, k), dtype=FIELDS[k]))
        tok_off.append(tok_off[-1] + len(ex))
        for q in ex.questions:
            qtype.append(q.qtype)
            nopt.append(q.n_opts)
            dec.append(q.decide_idx)
            target.append(np.asarray(q.target, dtype=np.float32))
            t_off.append(t_off[-1] + len(q.target))
        q_off.append(q_off[-1] + len(ex.questions))
        source.append(r.get("source", ""))
        family.append(r.get("family", ""))
    out = {k: (np.concatenate(v) if v else np.zeros(0, FIELDS[k])) for k, v in buf.items()}
    out.update(tok_off=np.asarray(tok_off, np.int64), q_off=np.asarray(q_off, np.int64), t_off=np.asarray(t_off, np.int64),
               qtype=np.asarray(qtype, np.int8), nopt=np.asarray(nopt, np.int16), dec=np.asarray(dec, np.int32),
               target=np.concatenate(target) if target else np.zeros(0, np.float32),
               source=np.asarray(source, dtype=object), family=np.asarray(family, dtype=object))
    return out


def _merge(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for k in list(FIELDS) + ["qtype", "nopt", "dec", "target", "source", "family"]:
        out[k] = np.concatenate([p[k] for p in parts])
    for k in ("tok_off", "q_off", "t_off"):
        offs, base = [np.zeros(1, np.int64)], 0
        for p in parts:
            offs.append(p[k][1:] + base)
            base += p[k][-1]
        out[k] = np.concatenate(offs)
    return out


def read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def build_store(rows: list[dict], tokenizer_path: str, max_len: int, workers: int = 4, shard: int = 20000) -> PackedStore:
    shards = [(rows[i:i + shard], tokenizer_path, max_len) for i in range(0, len(rows), shard)]
    if workers <= 1 or len(shards) == 1:
        parts = [_pack_shard(s) for s in shards]
    else:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(_pack_shard, shards))
    return PackedStore(_merge(parts))


def iter_shards(path: str, shard: int, limit: int | None = None):
    rows, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            n += 1
            if len(rows) >= shard:
                yield rows
                rows = []
            if limit and n >= limit:
                break
    if rows:
        yield rows


def build_store_streaming(path: str, tokenizer_path: str, max_len: int, workers: int = 4, shard: int = 20000,
                          limit: int | None = None) -> PackedStore:
    """Pack a JSONL file shard by shard with at most ``2*workers`` shards in flight (bounded RAM)."""
    parts, pending = [], []
    if workers <= 1:
        return PackedStore(_merge([_pack_shard((rows, tokenizer_path, max_len)) for rows in iter_shards(path, shard, limit)]))
    with ProcessPoolExecutor(workers) as ex:
        for rows in iter_shards(path, shard, limit):
            pending.append(ex.submit(_pack_shard, (rows, tokenizer_path, max_len)))
            if len(pending) >= 2 * workers:
                parts.append(pending.pop(0).result())
        parts.extend(f.result() for f in pending)
    return PackedStore(_merge(parts))


def load_or_build(jsonl: str, tokenizer_path: str, max_len: int, workers: int = 4, limit: int | None = None) -> PackedStore:
    cache = jsonl.replace(".jsonl", f".packed{max_len}{'_' + str(limit) if limit else ''}.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(jsonl):
        return PackedStore.load(cache)
    store = build_store_streaming(jsonl, tokenizer_path, max_len, workers, limit=limit)
    store.save(cache)
    return store
