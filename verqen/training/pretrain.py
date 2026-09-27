"""Stage 1: masked-language-model pretraining on FineWeb-Edu (bidirectional encoder).

Each row is ``<BOS>`` + 511 contiguous tokens from the token file, treated as one
STATE segment with full bidirectional attention (no mask tensor -> fused attention
kernels). 30% of non-special tokens are selected; 80% of those become ``<MASK>``,
10% a random token, 10% unchanged.
"""

from __future__ import annotations

import os
import threading
import queue
import time

import numpy as np
import torch
import torch.nn.functional as F

from verqen.config import VerqenConfig
from verqen.model.attention_mask import SEG_BOS, SEG_STATE
from verqen.model.verqen import VerqenModel
from verqen.tokenizer import VerqenTokenizer
from verqen.training.common import JsonlLogger, WSDSchedule, autocast, build_optimizer, maybe_compile
from verqen.utils.checkpoint import save_checkpoint


class TokenStream:
    """Random fixed-length windows from a flat uint16 token file, prefetched on a thread."""

    def __init__(self, path: str, rows: int, seq_len: int, bos: int, seed: int, holdout: int = 2_000_000,
                 split: str = "train", prefetch: int = 4):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.rows, self.seq_len, self.bos = rows, seq_len, bos
        n = len(self.data)
        self.lo, self.hi = (0, n - holdout - seq_len) if split == "train" else (n - holdout, n - seq_len)
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        threading.Thread(target=self._fill, daemon=True).start()

    def _make(self) -> torch.Tensor:
        starts = self.rng.integers(self.lo, self.hi, size=self.rows)
        idx = starts[:, None] + np.arange(self.seq_len - 1)[None]
        body = torch.from_numpy(self.data[idx].astype(np.int64))
        out = torch.cat([torch.full((self.rows, 1), self.bos, dtype=torch.long), body], dim=1)
        return out.pin_memory() if torch.cuda.is_available() else out

    def _fill(self):
        while True:
            self.q.put(self._make())

    def next(self) -> torch.Tensor:
        return self.q.get()


def mask_tokens(ids: torch.Tensor, prob: float, mask_id: int, vocab: int, n_special: int, gen: torch.Generator):
    special = ids < n_special
    sel = (torch.rand(ids.shape, device=ids.device, generator=gen) < prob) & ~special
    r = torch.rand(ids.shape, device=ids.device, generator=gen)
    inp = ids.clone()
    inp[sel & (r < 0.8)] = mask_id
    rand_pos = sel & (r >= 0.8) & (r < 0.9)
    inp[rand_pos] = torch.randint(n_special, vocab, (int(rand_pos.sum()),), device=ids.device, generator=gen)
    flat_idx = sel.flatten().nonzero().squeeze(1)
    return inp, flat_idx, ids.flatten()[flat_idx]


def run_pretrain(cfg: VerqenConfig, out_dir: str, budget_hours: float, device: torch.device) -> str:
    torch.manual_seed(cfg.training.seed)
    tok = VerqenTokenizer.from_pretrained(cfg.data.tokenizer_path)
    sp = tok.special_token_ids
    cfg.model.vocab_size = tok.vocab_size
    model = VerqenModel(cfg.model).to(device)
    params = model.count_parameters()
    print(f"[pretrain] parameters: {params}", flush=True)

    tc = cfg.training
    L = cfg.data.max_seq_len
    micro = tc.micro_rows
    accum = max(1, tc.batch_rows // micro)
    stream = TokenStream(cfg.data.pretrain_bin, micro, L, sp["bos"], tc.seed)
    val_stream = TokenStream(cfg.data.pretrain_bin, micro, L, sp["bos"], tc.seed + 1, split="val", prefetch=1)
    opt = build_optimizer(model, tc.learning_rate, tc.weight_decay, tc.betas)
    sched = WSDSchedule(tc.learning_rate, tc.warmup_steps, tc.max_steps, tc.decay_ratio, tc.min_lr_ratio,
                        budget_hours * 3600)
    backbone = maybe_compile(model.transformer, tc.compile)
    seg = torch.full((micro, L), SEG_STATE, dtype=torch.long, device=device)
    seg[:, 0] = SEG_BOS
    pos = torch.arange(L, device=device)[None].expand(micro, L).contiguous()
    gen = torch.Generator(device=device).manual_seed(tc.seed)
    n_special = max(sp.values()) + 1
    log = JsonlLogger(os.path.join(out_dir, "metrics.jsonl"))

    def mlm_loss(ids):
        inp, idx, labels = mask_tokens(ids, cfg.data.mlm_probability, sp["mask"], tok.vocab_size, n_special, gen)
        with autocast(device):
            h = backbone(inp, seg, pos, None)
            logits = model.mlm_head(h.reshape(-1, h.size(-1))[idx])
        loss = F.cross_entropy(logits.float(), labels)
        acc = (logits.argmax(-1) == labels).float().mean()
        return loss, acc

    step, t0, tokens = 0, time.time(), 0
    best_val = float("inf")
    latest = os.path.join(out_dir, "latest")
    if os.path.exists(os.path.join(latest, "training_state.pt")):  # resume after a crash / reboot
        from verqen.utils.checkpoint import load_checkpoint
        st = torch.load(os.path.join(latest, "training_state.pt"), map_location="cpu", weights_only=False)
        load_checkpoint(model, latest, opt)
        step, tokens = int(st["step"]), int(st.get("tokens", 0))
        elapsed = float(st.get("elapsed_s", 0.0))
        t0 = time.time() - elapsed
        sched.t0 = t0
        best_val = float(st.get("best_val", best_val))
        print(f"[pretrain] resumed from step {step} ({tokens / 1e9:.2f}B tokens, {elapsed / 3600:.2f} h elapsed)", flush=True)
    run_loss = torch.zeros((), device=device)
    run_acc = torch.zeros((), device=device)
    while not sched.done(step):
        lr = sched.lr(step)
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(accum):
            ids = stream.next().to(device, non_blocking=True)
            loss, acc = mlm_loss(ids)
            (loss / accum).backward()
            run_loss += loss.detach() / accum  # stays on GPU: no host sync per micro-step
            run_acc += acc.detach() / accum
            tokens += ids.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.logging.log_every == 0:
            el = time.time() - t0
            log.log(stage="pretrain", step=step, total=sched.total, loss=run_loss.item() / cfg.logging.log_every,
                    mlm_acc=run_acc.item() / cfg.logging.log_every, lr=lr, tok_per_s=int(tokens / el),
                    tokens_M=round(tokens / 1e6, 1), hours=round(el / 3600, 3))
            run_loss.zero_()
            run_acc.zero_()
        if step % tc.eval_every == 0 or sched.done(step):
            model.eval()
            with torch.no_grad():
                vl = [mlm_loss(val_stream.next().to(device))[0].item() for _ in range(20)]
            model.train()
            val = sum(vl) / len(vl)
            log.log(stage="pretrain_eval", step=step, val_mlm_loss=val)
            best_val = min(best_val, val)
        if step % cfg.logging.save_every == 0:
            save_checkpoint(model, latest, cfg.data.tokenizer_path, opt, step,
                            extra={"tokens": tokens, "elapsed_s": time.time() - t0, "best_val": best_val})

    final = os.path.join(out_dir, "final")
    save_checkpoint(model, final, cfg.data.tokenizer_path, None, step,
                    extra={"tokens": tokens, "best_val_mlm": best_val})
    print(f"[pretrain] done: {step} steps, {tokens / 1e9:.2f}B tokens -> {final}", flush=True)
    return final
