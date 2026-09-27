"""T2 decision training: DDP, wall-clock resumable, pre-registered checkpoint choice.

Fixes over T1's stage 2 (docs/production/T2_PLAN.md, P2/P3/P8):
* Resumable. Every ``save_minutes`` the model, optimizer, schedule plan and the data cursor
  (how many batches of the deterministic global batch stream were consumed) are saved
  atomically, so a crash or reboot loses minutes, not the stage. (T1 lost a whole run to a
  PC crash on 24 Sep 2026.)
* DDP. One deterministic global batch stream; rank r takes batches r, r+W, r+2W, ... so
  ranks never see the same data and every rank runs the same number of micro-steps.
  ``find_unused_parameters`` stays on: a micro-batch with no score questions leaves the
  ordinal head unused, which is legitimate.
* Checkpoint choice. ``final`` is the end of the schedule. Validation is logged (loss,
  accuracy, per source), and the *pre-registered* rule, "final unless its val accuracy
  is > 1 pt below the best seen", is applied and recorded rather than silently picking
  the lowest loss (T1: best-by-loss was the less accurate checkpoint).
"""

from __future__ import annotations

import json
import os
import time
from itertools import islice
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from verqen.config import VerqenConfig
from verqen.data.store import PackedStore, load_or_build
from verqen.losses.decision import decision_loss
from verqen.model.attention_mask import build_attention_mask
from verqen.model.verqen import VerqenModel
from verqen.training import distributed as D
from verqen.training.common import JsonlLogger, WSDSchedule, autocast, maybe_compile
from verqen.training.decisions import BatchPrefetcher, evaluate_store, head_param_groups, token_budget_batches
from verqen.training.pretrain_v3 import clean_state
from verqen.utils.checkpoint import load_model, save_checkpoint


class DecisionCore(nn.Module):
    def __init__(self, model: VerqenModel):
        super().__init__()
        self.model = model

    def forward(self, b):
        mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
        h = self.model.transformer(b["input_ids"], b["seg"], b["pos"], mask)
        return self.model.decision_readout(h, b)


def sharded(stream: Iterator[list[int]], rank: int, world: int, skip: int = 0) -> Iterator[list[int]]:
    """Rank r's share of a deterministic global batch stream, after skipping ``skip`` global batches."""
    for i, idx in enumerate(islice(stream, skip, None)):
        if (i + skip) % world == rank:
            yield idx


def run_decisions_v3(cfg: VerqenConfig, init_model: str, out_dir: str, budget_hours: float, info: D.DistInfo, *,
                     lr: float = 1e-4, head_lr_mult: float = 3.0, max_steps: int | None = None,
                     save_minutes: float | None = None) -> str:
    tc, device = cfg.training, info.device
    L = cfg.data.max_seq_len
    micro = tc.decision_micro_rows or tc.micro_rows  # measured at max_seq_len by production/preflight.py
    accum = max(1, tc.batch_rows // (micro * info.world))
    save_minutes = save_minutes if save_minutes is not None else cfg.pretrain.save_minutes
    d = cfg.data.decision_dir
    # rank 0 builds the packed caches first; the others wait, then load the cache (no double tokenization)
    if info.is_main:
        train = load_or_build(os.path.join(d, "train.jsonl"), cfg.data.tokenizer_path, L)
        val = load_or_build(os.path.join(d, "val.jsonl"), cfg.data.tokenizer_path, L)
    D.barrier(info)
    if not info.is_main:
        train = load_or_build(os.path.join(d, "train.jsonl"), cfg.data.tokenizer_path, L)
        val = None

    model = load_model(init_model, device)
    core = DecisionCore(model).to(device)
    groups = head_param_groups(model, lr, tc.weight_decay, head_lr_mult)
    opt = torch.optim.AdamW(groups, lr=lr, betas=tuple(tc.betas), fused=device.type == "cuda")
    tokens_per_step = tc.batch_rows * L * 0.9
    est = int(3 * int(train.lengths.sum()) / tokens_per_step)  # up to 3 epochs if the budget allows
    sched = WSDSchedule(lr, min(500, max(10, est // 20)), max_steps or est, decay_ratio=0.3, min_ratio=0.05,
                        budget_s=budget_hours * 3600)
    resume_dir = os.path.join(out_dir, "resume")
    step, consumed, t0 = 0, 0, time.time()
    history: list[dict] = []
    if os.path.exists(os.path.join(resume_dir, "resume.pt")):
        ck = torch.load(os.path.join(resume_dir, "resume.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        sched.total, sched._decay_start = ck["sched"]["total"], ck["sched"]["decay_start"]
        step, consumed, history = ck["step"], ck["consumed"], ck.get("history", [])
        t0 = time.time() - ck["elapsed_s"]
        if info.is_main:
            print(f"[decisions] resumed at step {step} (consumed {consumed} batches, {ck['elapsed_s'] / 3600:.2f} h)",
                  flush=True)
    sched.t0 = t0
    if tc.compile:
        model.transformer = maybe_compile(model.transformer, True)
    ddp = DDP(core, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True,
              broadcast_buffers=False) if info.enabled else core
    stream = token_budget_batches(train, micro, L, tc.seed, epochs=None)
    batches = iter(BatchPrefetcher(train, sharded(stream, info.rank, info.world, consumed), micro, L, 0))
    log = JsonlLogger(os.path.join(out_dir, "metrics.jsonl")) if info.is_main else None
    last_save, run = time.time(), {}

    def save_resume():
        def write(dd):
            torch.save({"model": clean_state(D.unwrap(model).state_dict()), "opt": opt.state_dict(),
                        "sched": {"total": sched.total, "decay_start": sched._decay_start}, "step": step,
                        "consumed": consumed, "elapsed_s": time.time() - t0, "history": history},
                       os.path.join(dd, "resume.pt"))
        D.atomic_dir_write(resume_dir, write)

    while True:
        cur, total = D.broadcast_floats([sched.lr(step), float(sched.total)], info)
        sched.total = int(total)
        if step >= sched.total:
            break
        for g in opt.param_groups:
            g["lr"] = cur * g["lr_mult"]
        for k in range(accum):
            b = {kk: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for kk, v in next(batches).items()}
            with D.maybe_no_sync(ddp, info.enabled and k < accum - 1):
                with autocast(device):
                    out = ddp(b)
                ld = decision_loss(out, b["target"], b["qtype"], cfg.loss)
                (ld["loss"] / accum).backward()
            for key in ("loss", "kl", "acc", "brier"):
                run[key] = run.get(key, 0.0) + float(ld[key].detach()) / accum
        consumed += accum * info.world
        torch.nn.utils.clip_grad_norm_(core.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if info.is_main and step % cfg.logging.log_every == 0:
            el = time.time() - t0
            log.log(stage="decisions", step=step, total=sched.total, lr=cur, hours=round(el / 3600, 3),
                    **{k: v / cfg.logging.log_every for k, v in run.items()})
            run = {}
        if info.is_main and (step % tc.eval_every == 0 or step >= sched.total):
            ev = evaluate_store(D.unwrap(model), val, cfg, device, max_examples=20000, backbone=model.transformer)
            history.append({"step": step, "val_loss": ev["loss"], "val_acc": ev["acc"]})
            log.log(stage="decisions_eval", step=step, val_loss=ev["loss"], val_acc=ev["acc"], val_kl=ev["kl"],
                    by_source=ev["by_source"])
        if info.is_main and (time.time() - last_save) / 60 >= save_minutes:
            save_resume()
            last_save = time.time()

    final = os.path.join(out_dir, "final")
    D.assert_in_sync(core, info)
    if info.is_main:
        save_resume()
        best = max(history, key=lambda h: h["val_acc"]) if history else None
        choice = {"rule": "final unless its val_acc is > 1 pt below the best eval", "history": history,
                  "final_val_acc": history[-1]["val_acc"] if history else None,
                  "best_val_acc": best["val_acc"] if best else None,
                  "flag_final_much_worse": bool(best and history[-1]["val_acc"] < best["val_acc"] - 0.01)}
        D.atomic_dir_write(final, lambda dd: save_checkpoint(D.unwrap(model), dd, cfg.data.tokenizer_path, None, step,
                                                              extra={"selection": choice}))
        with open(os.path.join(out_dir, "selection.json"), "w") as f:
            json.dump(choice, f, indent=2)
        print(f"[decisions] done: {step} steps -> {final}; selection {json.dumps({k: v for k, v in choice.items() if k != 'history'})}",
              flush=True)
    D.barrier(info)
    return final
