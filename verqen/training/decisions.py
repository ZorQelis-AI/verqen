"""Stage 2: decision fine-tuning on the hybrid (real + procedural) dataset.

* Token-budget batches: examples are packed several-per-row into fixed [rows, 512] tensors.
* Batches are built on a background thread so the GPU never waits for Python packing.
* Validation every ``eval_every`` steps; the checkpoint with the lowest validation loss
  (a proper scoring rule, so it rewards calibration as well as accuracy) is kept as ``best``.
* The randomly initialized decision head trains at ``head_lr_mult`` x the backbone LR.
"""

from __future__ import annotations

import collections
import os
import queue
import threading
import time
from typing import Iterator

import numpy as np
import torch

from verqen.config import VerqenConfig
from verqen.data.collator import collate_rows, pack_rows, to_device
from verqen.data.store import PackedStore
from verqen.losses.decision import decision_loss
from verqen.model.verqen import VerqenModel
from verqen.training.common import JsonlLogger, WSDSchedule, autocast, maybe_compile
from verqen.utils.checkpoint import save_checkpoint


def token_budget_batches(store: PackedStore, rows: int, max_len: int, seed: int, shuffle: bool = True,
                         fill: float = 0.95, epochs: int | None = None) -> Iterator[list[int]]:
    """Yield lists of example indices whose total length fits ``rows`` rows of ``max_len``."""
    lengths = store.lengths
    budget = int(rows * max_len * fill)
    rng = np.random.default_rng(seed)
    ep = 0
    while epochs is None or ep < epochs:
        order = rng.permutation(len(store)) if shuffle else np.arange(len(store))
        cur, used = [], 0
        for i in order:
            n = int(lengths[i])
            if used + n > budget and cur:
                yield cur
                cur, used = [], 0
            cur.append(int(i))
            used += n
        if cur:
            yield cur
        ep += 1


class BatchPrefetcher:
    def __init__(self, store: PackedStore, index_iter: Iterator[list[int]], rows: int, max_len: int, pad_id: int,
                 prefetch: int = 6):
        self.store, self.it, self.rows, self.max_len, self.pad = store, index_iter, rows, max_len, pad_id
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        for idx in self.it:
            exs = [self.store[i] for i in idx]
            packed, dropped = pack_rows(exs, self.max_len, self.rows)
            b = collate_rows(packed, self.max_len, self.pad, self.rows)
            b["example_idx"] = idx
            b["n_dropped"] = len(dropped)
            self.q.put(b)
        self.q.put(None)

    def __iter__(self):
        while True:
            b = self.q.get()
            if b is None:
                return
            yield b


def head_param_groups(model: VerqenModel, lr: float, weight_decay: float, head_mult: float):
    groups = {"bb_decay": [], "bb_nodecay": [], "head_decay": [], "head_nodecay": []}
    for name, p in model.named_parameters():
        if name.startswith("mlm_head"):
            p.requires_grad_(False)  # MLM head is not used after pretraining
            continue
        head = name.startswith("decision_head")
        nodecay = p.ndim < 2 or "embedding" in name or "norm" in name
        groups[("head" if head else "bb") + ("_nodecay" if nodecay else "_decay")].append(p)
    out = []
    for k, ps in groups.items():
        if ps:
            out.append({"params": ps, "weight_decay": 0.0 if "nodecay" in k else weight_decay,
                        "lr_mult": head_mult if k.startswith("head") else 1.0})
    return out


@torch.no_grad()
def evaluate_store(model: VerqenModel, store: PackedStore, cfg: VerqenConfig, device: torch.device,
                   rows: int = 32, max_examples: int | None = None, backbone=None) -> dict:
    """Accuracy / KL / proper-score loss overall and per source, plus raw outputs for calibration."""
    model.eval()
    backbone = backbone if backbone is not None else model.transformer
    n = len(store) if max_examples is None else min(max_examples, len(store))
    sub_order = iter(token_budget_batches(store, rows, cfg.data.max_seq_len, 0, shuffle=False, epochs=1))
    tot = collections.defaultdict(float)
    per_src = collections.defaultdict(lambda: [0.0, 0])
    seen = 0
    for idx in sub_order:
        idx = [i for i in idx if i < n]
        if not idx:
            break
        exs = [store[i] for i in idx]
        packed, dropped = pack_rows(exs, cfg.data.max_seq_len, None)
        b = to_device(collate_rows(packed, cfg.data.max_seq_len, 0), device)
        with autocast(device):
            from verqen.model.attention_mask import build_attention_mask
            mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
            h = backbone(b["input_ids"], b["seg"], b["pos"], mask)
            out = model.decision_readout(h, b)
        ld = decision_loss(out, b["target"], b["qtype"], cfg.loss)
        from verqen.losses.decision import correctness
        corr = correctness(out["probs"], b["target"], b["opt_mask"]).cpu().numpy()
        # map questions back to examples (packing reorders examples by length)
        ex_order = [e for row in packed for e in row]
        src_of_ex = {id(e): store.meta(i)["source"] for i, e in zip(idx, exs)}
        qi = 0
        for e in ex_order:
            s = src_of_ex[id(e)]
            for _ in e.questions:
                per_src[s][0] += corr[qi]
                per_src[s][1] += 1
                qi += 1
        k = len(b["qtype"])
        for key in ("loss", "kl", "brier", "acc"):
            tot[key] += float(ld[key]) * k
        tot["n"] += k
        seen += len(idx)
        if seen >= n:
            break
    model.train()
    res = {k: tot[k] / max(1, tot["n"]) for k in ("loss", "kl", "brier", "acc")}
    res["n_questions"] = int(tot["n"])
    res["by_source"] = {s: round(c / max(1, m), 4) for s, (c, m) in sorted(per_src.items())}
    return res


def run_decision_training(cfg: VerqenConfig, model: VerqenModel, train: PackedStore, val: PackedStore,
                          out_dir: str, budget_hours: float, device: torch.device, lr: float = 2e-4,
                          head_lr_mult: float = 5.0, epochs: int = 3, tokenizer_path: str | None = None,
                          max_steps: int | None = None) -> str:
    tc = cfg.training
    L = cfg.data.max_seq_len
    micro = tc.micro_rows
    accum = max(1, tc.batch_rows // micro)
    model.train()
    groups = head_param_groups(model, lr, tc.weight_decay, head_lr_mult)
    opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(tc.betas), fused=device.type == "cuda")
    tokens_per_epoch = int(train.lengths.sum())
    est_steps = max(100, int(epochs * tokens_per_epoch / (tc.batch_rows * L * 0.9)))
    if max_steps is not None:  # fixed-compute runs (ablations): same step count for every arm
        est_steps = max_steps
    sched = WSDSchedule(lr, min(500, est_steps // 20), est_steps, decay_ratio=0.3, min_ratio=0.05,
                        budget_s=budget_hours * 3600)
    backbone = maybe_compile(model.transformer, tc.compile)
    batches = BatchPrefetcher(train, token_budget_batches(train, micro, L, tc.seed), micro, L, 0)
    log = JsonlLogger(os.path.join(out_dir, "metrics.jsonl"))
    best, best_path = float("inf"), os.path.join(out_dir, "best")
    step, t0 = 0, time.time()
    run = collections.defaultdict(float)
    it = iter(batches)
    from verqen.model.attention_mask import build_attention_mask
    while not sched.done(step):
        cur = sched.lr(step)
        for g in opt.param_groups:
            g["lr"] = cur * g["lr_mult"]
        for _ in range(accum):
            b = to_device(next(it), device)
            with autocast(device):
                mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
                h = backbone(b["input_ids"], b["seg"], b["pos"], mask)
                out = model.decision_readout(h, b)
            ld = decision_loss(out, b["target"], b["qtype"], cfg.loss)
            (ld["loss"] / accum).backward()
            for k in ("loss", "kl", "acc", "brier"):
                run[k] += float(ld[k].detach()) / accum
            run["q"] += len(b["qtype"])
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.logging.log_every == 0:
            el = time.time() - t0
            log.log(stage="decisions", step=step, total=sched.total,
                    **{k: run[k] / cfg.logging.log_every for k in ("loss", "kl", "acc", "brier")},
                    lr=cur, q_per_s=int(run["q"] / el) if el else 0, hours=round(el / 3600, 3))
            run = collections.defaultdict(float, q=run["q"])
        if step % tc.eval_every == 0 or sched.done(step):
            ev = evaluate_store(model, val, cfg, device, backbone=backbone)
            log.log(stage="decisions_eval", step=step, val_loss=ev["loss"], val_kl=ev["kl"], val_acc=ev["acc"],
                    by_source=ev["by_source"])
            if ev["loss"] < best:
                best = ev["loss"]
                save_checkpoint(model, best_path, tokenizer_path or cfg.data.tokenizer_path, None, step,
                                extra={"val": ev})
    save_checkpoint(model, os.path.join(out_dir, "last"), tokenizer_path or cfg.data.tokenizer_path, None, step)
    print(f"[decisions] done: {step} steps, best val loss {best:.4f} -> {best_path}", flush=True)
    return best_path
