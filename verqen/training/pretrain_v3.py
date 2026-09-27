"""T2 pretraining: RTD or MLM, multi-source mix, DDP, wall-clock resumable, context extension.

    torchrun --nproc_per_node 2 scripts/v3/run_stage.py --config configs/v3/t2.yaml --stage pretrain

Objectives
* ``rtd`` (default): ELECTRA / DeBERTa-v3 replaced-token detection. A small generator
  (same width, ``gen_layers`` deep) is trained with MLM; its samples replace the masked
  tokens, and the discriminator (the model we keep) classifies every token as original or
  replaced. Every position produces a training signal instead of 15-30%, which is why RTD
  reaches a given downstream accuracy with far less compute. Embeddings use DeBERTa-v3's
  gradient-disentangled sharing: discriminator embedding = stop_grad(generator embedding)
  + a zero-initialized delta, so the discriminator's loss cannot drag the generator's
  embeddings around (the "tug-of-war" of plain sharing).
* ``mlm``: T1's objective, kept for the A/B.

Engineering rules (docs/production/T2_PLAN.md section 4):
* Stop / LR decisions come from rank 0 via broadcast; ranks never diverge in step count.
* Checkpoints every ``save_minutes`` of wall clock, atomic, holding everything a resume
  needs (both networks, optimizer, schedule plan, token count, elapsed time).
* The exported artifact is a plain VerqenModel (the discriminator with its embedding
  baked), loadable by every later stage with strict key checking.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from verqen.config import ModelConfig, VerqenConfig
from verqen.model.attention_mask import SEG_BOS, SEG_STATE
from verqen.model.transformer import VerqenTransformer
from verqen.model.verqen import VerqenModel
from verqen.tokenizer import VerqenTokenizer
from verqen.training import distributed as D
from verqen.training.common import JsonlLogger, WSDSchedule, autocast, build_optimizer, maybe_compile
from verqen.training.pretrain import mask_tokens
from verqen.utils.checkpoint import save_checkpoint


# ── data ──────────────────────────────────────────────────────────────────────
class MixedTokenStream:
    """Rows of ``seq_len`` tokens (BOS + window) drawn from several uint16 token files by weight.

    Each source reserves its last ``holdout`` tokens for validation. Rows are sampled
    independently, so ranks only need different seeds (no sharding bookkeeping), and a
    resume only needs a reseed.
    """

    def __init__(self, sources: list[dict], rows: int, seq_len: int, bos: int, seed: int, split: str = "train",
                 holdout: int = 2_000_000, prefetch: int = 4):
        self.maps, self.bounds, w = [], [], []
        for s in sources:
            m = np.memmap(s["path"], dtype=np.uint16, mode="r")
            n = len(m)
            if n <= holdout + 2 * seq_len:
                raise ValueError(f"source {s['path']} has only {n} tokens")
            self.maps.append(m)
            self.bounds.append((0, n - holdout - seq_len) if split == "train" else (n - holdout, n - seq_len))
            w.append(float(s.get("weight", 1.0)))
        self.p = np.asarray(w) / sum(w)
        self.names = [s.get("name", os.path.basename(s["path"])) for s in sources]
        self.rows, self.seq_len, self.bos = rows, seq_len, bos
        self.rng = np.random.default_rng(seed)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        threading.Thread(target=self._fill, daemon=True).start()

    def _make(self) -> torch.Tensor:
        src = self.rng.choice(len(self.maps), size=self.rows, p=self.p)
        out = np.empty((self.rows, self.seq_len), dtype=np.int64)
        out[:, 0] = self.bos
        for r, s in enumerate(src):
            lo, hi = self.bounds[s]
            st = int(self.rng.integers(lo, hi))
            out[r, 1:] = self.maps[s][st: st + self.seq_len - 1]
        t = torch.from_numpy(out)
        return t.pin_memory() if torch.cuda.is_available() else t

    def _fill(self):
        while True:
            self.q.put(self._make())

    def next(self) -> torch.Tensor:
        return self.q.get()


# ── objectives ────────────────────────────────────────────────────────────────
class GDESEmbedding(nn.Module):
    """Discriminator token embedding = stop_grad(generator embedding) + delta (DeBERTa-v3 GDES)."""

    def __init__(self, source: nn.Embedding):
        super().__init__()
        self._source = [source]  # outside the module tree: the generator owns and saves it
        self.delta = nn.Embedding(source.num_embeddings, source.embedding_dim)
        nn.init.zeros_(self.delta.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self._source[0].weight.detach()) + self.delta(ids)

    def baked(self) -> torch.Tensor:
        return (self._source[0].weight.detach() + self.delta.weight.detach()).clone()


def _strip_decision_head(m: VerqenModel) -> VerqenModel:
    m.decision_head = nn.Identity()  # unused in pretraining: DDP must not see unused parameters
    return m


class MLMCore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.model = _strip_decision_head(VerqenModel(cfg))

    def forward(self, ids, seg, pos, sp, prob, vocab, n_special, gen):
        inp, idx, labels = mask_tokens(ids, prob, sp["mask"], vocab, n_special, gen)
        h = self.model.transformer(inp, seg, pos, None)
        logits = self.model.mlm_head(h.reshape(-1, h.size(-1))[idx])
        loss = F.cross_entropy(logits.float(), labels)
        return loss, {"mlm_loss": loss.detach(), "mlm_acc": (logits.argmax(-1) == labels).float().mean().detach()}

    def discriminator_transformer(self) -> VerqenTransformer:
        return D.unwrap(self.model.transformer)


class RTDCore(nn.Module):
    def __init__(self, cfg: ModelConfig, gen_layers: int, rtd_weight: float):
        super().__init__()
        gcfg = ModelConfig.from_dict({**cfg.__dict__, "num_layers": gen_layers, "set_layers": 0})
        self.gen = _strip_decision_head(VerqenModel(gcfg))
        self.disc = VerqenTransformer(cfg)
        dm = VerqenModel(cfg)  # borrow T1's init scheme for the discriminator stack
        self.disc.load_state_dict(dm.transformer.state_dict())
        del dm
        self.disc.embeddings.token_embedding = GDESEmbedding(self.gen.transformer.embeddings.token_embedding)
        self.rtd_head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.rtd_head.bias)
        self.rtd_weight = rtd_weight

    def forward(self, ids, seg, pos, sp, prob, vocab, n_special, gen):
        inp, idx, labels = mask_tokens(ids, prob, sp["mask"], vocab, n_special, gen)
        gh = self.gen.transformer(inp, seg, pos, None)
        glogits = self.gen.mlm_head(gh.reshape(-1, gh.size(-1))[idx]).float()
        gen_loss = F.cross_entropy(glogits, labels)
        with torch.no_grad():  # Gumbel-max sample from the generator; not differentiated through
            u = torch.rand(glogits.shape, device=glogits.device, generator=gen).clamp_(1e-9, 1 - 1e-9)
            sampled = (glogits - torch.log(-torch.log(u))).argmax(-1)
            sampled = torch.where(sampled < n_special, labels, sampled)  # never emit control tokens
            disc_in = ids.clone().view(-1)
            disc_in[idx] = sampled
            disc_in = disc_in.view_as(ids)
            replaced = (disc_in != ids).float()
            real = (ids >= n_special).float()
        dh = self.disc(disc_in, seg, pos, None)
        dlogit = self.rtd_head(dh).squeeze(-1).float()
        disc_loss = (F.binary_cross_entropy_with_logits(dlogit, replaced, reduction="none") * real).sum() / real.sum()
        loss = gen_loss + self.rtd_weight * disc_loss
        with torch.no_grad():
            dacc = (((dlogit > 0).float() == replaced).float() * real).sum() / real.sum()
            gacc = (glogits.argmax(-1) == labels).float().mean()
        return loss, {"gen_loss": gen_loss.detach(), "gen_acc": gacc, "rtd_loss": disc_loss.detach(), "rtd_acc": dacc,
                      "replaced": (replaced * real).sum() / real.sum()}

    def discriminator_transformer(self) -> VerqenTransformer:
        return D.unwrap(self.disc)


def build_core(cfg: VerqenConfig) -> nn.Module:
    pc = cfg.pretrain
    if pc.objective == "rtd":
        return RTDCore(cfg.model, pc.gen_layers, pc.rtd_weight)
    if pc.objective == "mlm":
        return MLMCore(cfg.model)
    raise ValueError(f"unknown pretrain objective {pc.objective!r}")


def export_model(core: nn.Module, cfg: VerqenConfig, out_dir: str, tokenizer_path: str, meta: dict) -> str:
    """Plain VerqenModel from the discriminator (GDES embedding baked); strict round-trip checked."""
    core = D.unwrap(core)
    model = VerqenModel(cfg.model)
    src = core.discriminator_transformer()
    state = {k: v for k, v in src.state_dict().items() if not k.startswith("embeddings.token_embedding.")}
    if isinstance(src.embeddings.token_embedding, GDESEmbedding):
        state["embeddings.token_embedding.weight"] = src.embeddings.token_embedding.baked()
    else:
        state["embeddings.token_embedding.weight"] = src.embeddings.token_embedding.weight.detach().clone()
    model.transformer.load_state_dict(clean_state(state), strict=True)
    D.atomic_dir_write(out_dir, lambda d: save_checkpoint(model, d, tokenizer_path, None, meta.get("step", 0), extra=meta))
    return out_dir


# ── training loop ─────────────────────────────────────────────────────────────
def clean_state(state: dict) -> dict:
    """Drop torch.compile (`_orig_mod.`) and DDP (`module.`) prefixes, so compiled and eager runs
    save and load identical keys (SoreQen LESSONS #1: a mismatch silently trained from scratch)."""
    return {k.replace("_orig_mod.", "").removeprefix("module."): v for k, v in state.items()}


def _save_resume(core, opt, sched, state: dict, path: str, tokenizer_path: str):
    def write(d):
        torch.save({"core": clean_state(D.unwrap(core).state_dict()), "opt": opt.state_dict(),
                    "sched": {"total": sched.total, "decay_start": sched._decay_start}, **state},
                   os.path.join(d, "resume.pt"))
        with open(os.path.join(d, "state.json"), "w") as f:
            json.dump({k: v for k, v in state.items() if isinstance(v, (int, float, str))}, f)
    D.atomic_dir_write(path, write)


def run_pretrain_v3(cfg: VerqenConfig, out_dir: str, budget_hours: float, info: D.DistInfo, *,
                    stage: str = "pretrain", init_core: str | None = None, seq_len: int | None = None,
                    micro_rows: int | None = None, max_steps: int | None = None) -> str:
    """Train until ``budget_hours`` of wall clock (or ``max_steps``); returns the exported model dir.

    ``init_core``: a ``resume`` directory from an earlier stage to start from (context extension).
    """
    pc, tc = cfg.pretrain, cfg.training
    L = seq_len or pc.seq_len
    micro = micro_rows or tc.micro_rows
    accum = max(1, tc.batch_rows // (micro * info.world))
    device = info.device
    torch.manual_seed(tc.seed)
    tok = VerqenTokenizer.from_pretrained(cfg.data.tokenizer_path)
    sp = tok.special_token_ids
    cfg.model.vocab_size = tok.vocab_size
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, L)
    assert L <= 4096, "RotaryEmbedding clamps positions >= 4096"
    core = build_core(cfg).to(device)
    resume_dir = os.path.join(out_dir, "resume")
    start_state: dict[str, Any] = {"step": 0, "tokens": 0, "elapsed_s": 0.0}
    opt = build_optimizer(core, tc.learning_rate, tc.weight_decay, tc.betas)
    sched = WSDSchedule(tc.learning_rate, tc.warmup_steps, max_steps or tc.max_steps, tc.decay_ratio,
                        tc.min_lr_ratio, budget_hours * 3600)
    if os.path.exists(os.path.join(resume_dir, "resume.pt")):
        ck = torch.load(os.path.join(resume_dir, "resume.pt"), map_location="cpu", weights_only=False)
        core.load_state_dict(ck["core"], strict=True)
        opt.load_state_dict(ck["opt"])
        sched.total, sched._decay_start = ck["sched"]["total"], ck["sched"]["decay_start"]
        start_state = {k: ck[k] for k in ("step", "tokens", "elapsed_s")}
        if info.is_main:
            print(f"[{stage}] resumed at step {start_state['step']} ({start_state['elapsed_s'] / 3600:.2f} h)", flush=True)
    elif init_core:
        ck = torch.load(os.path.join(init_core, "resume.pt"), map_location="cpu", weights_only=False)
        core.load_state_dict(ck["core"], strict=True)  # a stage load that matches nothing must crash
        if info.is_main:
            print(f"[{stage}] initialized from {init_core} (step {ck['step']})", flush=True)
    step, tokens = int(start_state["step"]), int(start_state["tokens"])
    t0 = time.time() - float(start_state["elapsed_s"])
    sched.t0 = t0

    for name, mod in (("gen", getattr(core, "gen", None)), ("disc", getattr(core, "disc", None)),
                      ("model", getattr(core, "model", None))):
        if mod is not None and tc.compile:
            tgt = mod.transformer if hasattr(mod, "transformer") else mod
            compiled = maybe_compile(tgt, True)
            if hasattr(mod, "transformer"):
                mod.transformer = compiled
            else:
                setattr(core, name, compiled)
    if info.enabled:
        torch._dynamo.config.optimize_ddp = False  # DDPOptimizer graph-splitting fails with compiled submodules
        ddp = DDP(core, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=False)
    else:
        ddp = core

    stream = MixedTokenStream(pc.sources, micro, L, sp["bos"], tc.seed * 1000 + info.rank + 7919 * step,
                              holdout=pc.val_holdout_tokens)
    val = MixedTokenStream(pc.sources, micro, L, sp["bos"], tc.seed + 1, split="val", holdout=pc.val_holdout_tokens,
                           prefetch=1) if info.is_main else None
    seg = torch.full((micro, L), SEG_STATE, dtype=torch.long, device=device)
    seg[:, 0] = SEG_BOS
    pos = torch.arange(L, device=device)[None].expand(micro, L).contiguous()
    gen = torch.Generator(device=device).manual_seed(tc.seed * 7 + info.rank * 104729 + step)
    n_special = max(sp.values()) + 1
    args = (sp, pc.mask_prob, tok.vocab_size, n_special, gen)
    log = JsonlLogger(os.path.join(out_dir, "metrics.jsonl")) if info.is_main else None
    last_save = time.time()
    run: dict[str, torch.Tensor] = {}
    if info.is_main:
        n = sum(p.numel() for p in D.unwrap(core).parameters())
        print(f"[{stage}] objective={pc.objective} params(core)={n / 1e6:.1f}M world={info.world} micro={micro} "
              f"accum={accum} seq={L} tokens/step={micro * accum * info.world * L:,}", flush=True)

    while True:
        lr, total = D.broadcast_floats([sched.lr(step), float(sched.total)], info)
        sched.total = int(total)
        if step >= sched.total:
            break
        for g in opt.param_groups:
            g["lr"] = lr
        for k in range(accum):
            ids = stream.next().to(device, non_blocking=True)
            with D.maybe_no_sync(ddp, info.enabled and k < accum - 1):
                with autocast(device):
                    loss, m = ddp(ids, seg, pos, *args)
                (loss / accum).backward()
            for key, v in m.items():
                run[key] = run.get(key, torch.zeros((), device=device)) + v.float() / accum
            tokens += ids.numel() * info.world
        torch.nn.utils.clip_grad_norm_(core.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if info.is_main and step % cfg.logging.log_every == 0:
            el = time.time() - t0
            log.log(stage=stage, step=step, total=sched.total, lr=lr, tok_per_s=int(tokens / max(el, 1e-9)),
                    tokens_M=round(tokens / 1e6, 1), hours=round(el / 3600, 3),
                    **{k: round(v.item() / cfg.logging.log_every, 5) for k, v in run.items()})
            run = {}
        if info.is_main and (step % tc.eval_every == 0 or step >= sched.total):
            core.eval()
            with torch.no_grad(), autocast(device):
                vals = [D.unwrap(core)(val.next().to(device), seg, pos, *args)[1] for _ in range(20)]
            core.train()
            log.log(stage=f"{stage}_eval", step=step,
                    **{f"val_{k}": round(float(np.mean([v[k].item() for v in vals])), 5) for k in vals[0]})
        if info.is_main and (time.time() - last_save) / 60 >= pc.save_minutes:
            _save_resume(core, opt, sched, {"step": step, "tokens": tokens, "elapsed_s": time.time() - t0},
                         resume_dir, cfg.data.tokenizer_path)
            last_save = time.time()

    final = os.path.join(out_dir, "final")
    D.assert_in_sync(core, info)
    if info.is_main:
        meta = {"step": step, "tokens": tokens, "elapsed_s": time.time() - t0, "objective": pc.objective, "seq_len": L}
        _save_resume(core, opt, sched, meta, resume_dir, cfg.data.tokenizer_path)  # next stage initializes from it
        export_model(core, cfg, final, cfg.data.tokenizer_path, meta)
        print(f"[{stage}] done: {step} steps, {tokens / 1e9:.3f}B tokens -> {final}", flush=True)
    D.barrier(info)
    return final
