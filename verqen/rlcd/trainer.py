"""VerQen RLCD — Reinforcement Learning from Calibrated Decisions.

Two interleaved objectives, both kept close to the fine-tuned reference policy by a KL term:

1. Decision RLCD (labelled data). The action space is the option set, so the expected
   reward is computed *exactly*, with no sampling variance:

       J = sum_a pi(a) R(a),  R(a) = +1 if a is a best option, else -wrong_penalty

   A proper-scoring anchor (log score + Brier) keeps the probabilities calibrated;
   maximizing expected accuracy alone would push every distribution toward one-hot.

2. Environment RLCD (procedural multi-step environments). A group of rollouts per
   episode seed is sampled from the policy, and each action is weighted by its
   group-relative return advantage (GRPO style). When the environment has an oracle,
   a proper-scoring term on the oracle's optimal-action distribution teaches the model
   to be *calibrated about which actions are optimal*, not just to act well.

v1 "RLCD" never ran (its targets check was always False) and, had it run, would have
pushed probability away from correct answers because its rewards were mostly negative.
"""

from __future__ import annotations

import collections
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from verqen.data.collator import collate_rows, pack_rows, to_device
from verqen.data.packing import SequencePacker
from verqen.data.store import PackedStore
from verqen.environments.procedural import ENVIRONMENTS
from verqen.losses.decision import correctness
from verqen.model.attention_mask import build_attention_mask
from verqen.model.verqen import VerqenModel
from verqen.tokenizer import VerqenTokenizer
from verqen.training.common import JsonlLogger, autocast
from verqen.training.decisions import token_budget_batches
from verqen.utils.checkpoint import save_checkpoint


@dataclass
class RLCDConfig:
    lr: float = 3e-5
    steps: int = 3000
    budget_hours: float = 1.5
    wrong_penalty: float = 0.5
    proper_weight: float = 0.5
    kl_weight: float = 0.05
    env_every: int = 2               # one environment update every N decision updates
    env_seeds: int = 16              # distinct episodes per environment update
    group: int = 4                   # rollouts per episode seed (group-relative baseline)
    oracle_weight: float = 0.5
    decision_rows: int = 16
    env_chunk_rows: int = 16         # rows per environment-gradient chunk (memory bound, like decision_rows)
    log_every: int = 50
    eval_every: int = 500
    val_batches: int = 24            # fixed held-out slice scored every eval_every steps (see run_rlcd)


def _forward(model, b):
    mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
    h = model.transformer(b["input_ids"], b["seg"], b["pos"], mask)
    return model.decision_readout(h, b)


def decision_rlcd_loss(out, ref_logp, target, mask, cfg: RLCDConfig):
    probs, logp = out["probs"], out["logp"]
    best = target.masked_fill(~mask, -1).max(-1, keepdim=True).values
    reward = torch.where(target >= best - 1e-6, 1.0, -cfg.wrong_penalty).masked_fill(~mask, 0.0)
    expected_reward = (probs * reward).sum(-1)
    safe = logp.masked_fill(~mask, 0.0)
    ce = -(target * safe).sum(-1)
    brier = ((probs - target) ** 2 * mask).sum(-1)
    kl = (probs * (safe - ref_logp.masked_fill(~mask, 0.0))).sum(-1)
    loss = (-expected_reward + cfg.proper_weight * (ce + brier) + cfg.kl_weight * kl).mean()
    return loss, {"expected_reward": expected_reward.mean().item(), "kl": kl.mean().item(),
                  "acc": correctness(probs.detach(), target, mask).mean().item()}


class EnvRunner:
    """Batched rollouts of procedural environments through the model."""

    def __init__(self, model: VerqenModel, tok: VerqenTokenizer, device: torch.device, max_len: int = 512):
        self.model, self.device = model, device
        self.packer = SequencePacker(tok, max_len)
        self.max_len = max_len

    def _pack(self, obs_list):
        exs = []
        for o in obs_list:
            n = len(o["options"])
            exs.append(self.packer.pack({"state": o["state"], "questions": [
                {"type": "choice", "text": o["question"], "options": o["options"], "target": [1.0 / n] * n}]}))
        return exs

    @torch.no_grad()
    def policy(self, exs, greedy=False):
        packed, dropped = pack_rows(exs, self.max_len, None)
        if dropped:
            raise ValueError(f"{len(dropped)} environment observation(s) do not fit max_len={self.max_len} even after "
                             "state truncation (question + options alone are too long); raise data.max_seq_len")
        order = [e for row in packed for e in row]
        b = to_device(collate_rows(packed, self.max_len, 0), self.device)
        with autocast(self.device):
            out = _forward(self.model, b)
        probs = out["probs"].float().cpu()
        pos_of = {id(e): i for i, e in enumerate(order)}
        acts = []
        for e in exs:
            p = probs[pos_of[id(e)], : e.questions[0].n_opts]
            acts.append(int(p.argmax()) if greedy else int(torch.multinomial(p / p.sum(), 1)))
        return acts

    def rollout(self, env_names, seeds, greedy=False):
        """Run one episode per (env, seed); return per-episode transitions and returns."""
        envs = []
        for name, seed in zip(env_names, seeds):
            env = ENVIRONMENTS[name]()
            env.reset(random.Random(seed))
            envs.append(env)
        traj = [[] for _ in envs]
        ret = [0.0] * len(envs)
        optimal_hits = [[0, 0] for _ in envs]
        active = list(range(len(envs)))
        while active:
            obs = [envs[i].observe() for i in active]
            valid = [k for k, o in enumerate(obs) if len(o["options"]) >= 2]
            if len(valid) < len(obs):
                active = [active[k] for k in valid]
                obs = [obs[k] for k in valid]
                if not active:
                    break
            exs = self._pack(obs)
            acts = self.policy(exs, greedy)
            nxt = []
            for i, ex, a in zip(active, exs, acts):
                oracle = envs[i].oracle()
                if oracle is not None:
                    optimal_hits[i][0] += int(a in oracle)
                    optimal_hits[i][1] += 1
                    n = ex.questions[0].n_opts
                    ex.questions[0].target = [1.0 / len(oracle) if k in oracle else 0.0 for k in range(n)] if oracle else ex.questions[0].target
                r, done = envs[i].step(a)
                ret[i] += r
                traj[i].append((ex, a, oracle is not None and len(oracle) > 0))
                if not done:
                    nxt.append(i)
            active = nxt
        return traj, ret, optimal_hits


def env_rlcd_backward(model, ref_model, runner: EnvRunner, cfg: RLCDConfig, rng: random.Random, device,
                      chunk_rows: int = 16) -> dict:
    """Roll out, then accumulate the environment-RLCD gradient chunk by chunk (bounded memory)."""
    names = list(ENVIRONMENTS)
    env_names, seeds = [], []
    for _ in range(cfg.env_seeds):
        name, seed = rng.choice(names), rng.randrange(10**9)
        env_names += [name] * cfg.group
        seeds += [seed] * cfg.group
    model.eval()
    traj, ret, hits = runner.rollout(env_names, seeds)
    model.train()
    ret_t = torch.tensor(ret).view(cfg.env_seeds, cfg.group)
    adv = ((ret_t - ret_t.mean(1, keepdim=True)) / (ret_t.std(1, keepdim=True) + 1e-4)).flatten()
    items = [(ex, a, float(adv[i]), o) for i, t in enumerate(traj) for ex, a, o in t]
    if not items:
        return {}
    n_total = len(items)
    n_oracle = max(1, sum(1 for it in items if it[3]))
    # group transitions into chunks that pack into <= chunk_rows rows
    chunks, cur, used = [], [], 0
    for it in items:
        if used + len(it[0]) > chunk_rows * runner.max_len * 0.9 and cur:
            chunks.append(cur)
            cur, used = [], 0
        cur.append(it)
        used += len(it[0])
    if cur:
        chunks.append(cur)
    kl_sum = 0.0
    for ch in chunks:
        exs = [c[0] for c in ch]
        packed, dropped = pack_rows(exs, runner.max_len, None)
        order = [e for row in packed for e in row]
        meta = {id(c[0]): c for c in ch}
        b = to_device(collate_rows(packed, runner.max_len, 0), device)
        with autocast(device):
            out = _forward(model, b)
            with torch.no_grad():
                ref = _forward(ref_model, b)
        logp, mask = out["logp"], out["opt_mask"]
        a = torch.tensor([meta[id(e)][1] for e in order], device=device)
        A = torch.tensor([meta[id(e)][2] for e in order], device=device)
        o = torch.tensor([meta[id(e)][3] for e in order], device=device, dtype=torch.float32)
        safe = logp.masked_fill(~mask, 0.0)
        pg = -(A * logp.gather(1, a[:, None]).squeeze(1)).sum() / n_total
        target = b["target"]
        oracle = ((-(target * safe).sum(-1) + ((out["probs"] - target) ** 2 * mask).sum(-1)) * o).sum() / n_oracle
        kl = (out["probs"] * (safe - ref["logp"].masked_fill(~mask, 0.0))).sum(-1).sum() / n_total
        (pg + cfg.oracle_weight * oracle + cfg.kl_weight * kl).backward()
        kl_sum += kl.item()
    opt_rate = sum(h[0] for h in hits) / max(1, sum(h[1] for h in hits))
    return {"env_return": float(np.mean(ret)), "env_optimal_action_rate": opt_rate, "env_kl": kl_sum,
            "env_transitions": n_total}


@torch.no_grad()
def evaluate_envs(model, runner: EnvRunner, episodes: int = 40, seed: int = 999) -> dict:
    """Greedy policy: mean return and rate of oracle-optimal actions per environment."""
    model.eval()
    res = {}
    for name in ENVIRONMENTS:
        seeds = [seed + k for k in range(episodes)]
        _, ret, hits = runner.rollout([name] * episodes, seeds, greedy=True)
        res[name] = {"mean_return": round(float(np.mean(ret)), 3),
                     "optimal_action_rate": round(sum(h[0] for h in hits) / max(1, sum(h[1] for h in hits)), 4)}
    model.train()
    return res


@torch.no_grad()
def fixed_val(model, ref_model, val: PackedStore, batches: list, cfg: RLCDConfig, max_len: int, device) -> dict:
    """Decision metrics on the same held-out batches every time: comparable across steps."""
    was = model.training
    model.eval()
    tot = collections.defaultdict(float)
    for idx in batches:
        packed, _ = pack_rows([val[i] for i in idx], max_len, cfg.decision_rows)
        b = to_device(collate_rows(packed, max_len, 0, cfg.decision_rows), device)
        with autocast(device):
            out, ref = _forward(model, b), _forward(ref_model, b)
        _, m = decision_rlcd_loss(out, ref["logp"], b["target"], out["opt_mask"], cfg)
        for k, v in m.items():
            tot["val_" + k] += v / len(batches)
    model.train(was)
    return {k: round(v, 5) for k, v in tot.items()}


def run_rlcd(model: VerqenModel, ref_model: VerqenModel, train: PackedStore, tok: VerqenTokenizer, out_dir: str,
             device: torch.device, cfg: RLCDConfig, tokenizer_path: str, max_len: int = 512,
             resume_dir: str | None = None, save_minutes: float = 20.0, val: PackedStore | None = None) -> str:
    """``resume_dir``: wall-clock checkpoints (model, optimizer, step, elapsed) every ``save_minutes``;
    a restart continues from the last one instead of from step 0 (T2 plan P2).

    ``val``: a held-out store scored on a FIXED slice at start and every ``eval_every`` steps. The
    per-step ``dec_*`` numbers come from the training stream, whose difficulty drifts with batch
    order: on AB_rc (24 Sep) they fell 0.96 -> 0.91 while the frozen pre-RLCD model scored the
    same 0.91 on those later batches, so the "decline" was the data, not RLCD."""
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    params = [p for n, p in model.named_parameters() if not n.startswith("mlm_head")]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0, betas=(0.9, 0.95))
    runner = EnvRunner(model, tok, device, max_len)
    log = JsonlLogger(os.path.join(out_dir, "metrics.jsonl"))
    log.log(stage="rlcd_env_eval_before", **{k: str(v) for k, v in evaluate_envs(model, runner).items()})
    val_batches = None
    if val is not None and len(val):
        vit = iter(token_budget_batches(val, cfg.decision_rows, max_len, seed=5))
        val_batches = [next(vit) for _ in range(cfg.val_batches)]
        log.log(stage="rlcd_val", step=0, **fixed_val(model, ref_model, val, val_batches, cfg, max_len, device))
    batches = iter(token_budget_batches(train, cfg.decision_rows, max_len, seed=77))
    rng = random.Random(1234)
    t0, step = time.time(), 0
    if resume_dir and os.path.exists(os.path.join(resume_dir, "resume.pt")):
        ck = torch.load(os.path.join(resume_dir, "resume.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=True)
        opt.load_state_dict(ck["opt"])
        step, t0 = int(ck["step"]), time.time() - float(ck["elapsed_s"])
        for _ in range(step):  # one decision batch per step: fast-forward the data cursor
            next(batches)
        rng = random.Random(1234 + step)
        print(f"[rlcd] resumed at step {step}", flush=True)
    last_save = time.time()
    run = collections.defaultdict(float)
    cnt = collections.Counter()
    while step < cfg.steps and time.time() - t0 < cfg.budget_hours * 3600:
        idx = next(batches)
        packed, _ = pack_rows([train[i] for i in idx], max_len, cfg.decision_rows)
        b = to_device(collate_rows(packed, max_len, 0, cfg.decision_rows), device)
        with autocast(device):
            out = _forward(model, b)
            with torch.no_grad():
                ref = _forward(ref_model, b)
        loss, m = decision_rlcd_loss(out, ref["logp"], b["target"], out["opt_mask"], cfg)
        for k, v in m.items():
            run["dec_" + k] += v
        cnt["dec"] += 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % cfg.env_every == 0:
            em = env_rlcd_backward(model, ref_model, runner, cfg, rng, device, chunk_rows=cfg.env_chunk_rows)
            for k, v in em.items():
                run[k] += v
            cnt["env"] += bool(em)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.log_every == 0:
            log.log(stage="rlcd", step=step, hours=round((time.time() - t0) / 3600, 3),
                    **{k: v / max(1, cnt["dec" if k.startswith("dec_") else "env"]) for k, v in run.items()})
            run.clear()
            cnt.clear()
        if step % cfg.eval_every == 0:
            log.log(stage="rlcd_env_eval", step=step, **{k: str(v) for k, v in evaluate_envs(model, runner).items()})
            if val_batches:
                log.log(stage="rlcd_val", step=step, **fixed_val(model, ref_model, val, val_batches, cfg, max_len, device))
        if resume_dir and (time.time() - last_save) / 60 >= save_minutes:
            from verqen.training.distributed import atomic_dir_write
            state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
            atomic_dir_write(resume_dir, lambda d: torch.save(
                {"model": state, "opt": opt.state_dict(), "step": step, "elapsed_s": time.time() - t0},
                os.path.join(d, "resume.pt")))
            last_save = time.time()
    final = os.path.join(out_dir, "final")
    log.log(stage="rlcd_env_eval_after", **{k: str(v) for k, v in evaluate_envs(model, runner).items()})
    save_checkpoint(model, final, tokenizer_path, None, step)
    print(f"[rlcd] done: {step} steps -> {final}", flush=True)
    return final
