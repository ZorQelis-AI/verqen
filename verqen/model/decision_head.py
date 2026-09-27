"""VerQen-T1 decision head.

Pipeline per question (all questions of a batch are processed together):

1. Read the ``<DECIDE>`` vector ``d`` and a mean-pooled vector per option ``o_i``.
2. Run ``set_layers`` permutation-equivariant transformer layers over ``[d, o_1..o_n]``
   with no positional information. This is where options are compared with each
   other ("all of the above", "which is largest", elimination), and the result is
   exactly invariant to option order.
3. Score:
   * choice / noul: an MLP maps each refined option vector to a logit, softmax.
   * score (ordinal): a cumulative-link model. A scalar ``s`` from ``d`` and
     monotone thresholds built from the level vectors in their given low->high
     order give ``P(y <= k) = sigmoid(b_k - s)``, so probability mass respects
     the ordering of levels.
4. A selective head predicts ``P(top-1 is correct)`` from detached features. It
   is trained on correctness and is what escalation/risk control is built on.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from verqen.config import ModelConfig
from .mlp import SwiGLUMLP
from .normalization import RMSNorm

QTYPE_CHOICE, QTYPE_SCORE, QTYPE_NOUL = 0, 1, 2
QTYPE_IDS = {"choice": QTYPE_CHOICE, "score": QTYPE_SCORE, "noul": QTYPE_NOUL}


class SetBlock(nn.Module):
    """Pre-norm transformer block without positions (permutation-equivariant)."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, ffn_size: int, eps: float):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.norm1 = RMSNorm(hidden_size, eps)
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.norm2 = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUMLP(hidden_size, ffn_size)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        N, S, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(N, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        x = x + self.o_proj(a.transpose(1, 2).reshape(N, S, -1))
        return x + self.ffn(self.norm2(x))


class DecisionHead(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        D = config.hidden_size
        self.set_layers = nn.ModuleList([
            SetBlock(D, config.num_heads, config.head_dim, config.ffn_size, config.norm_eps)
            for _ in range(config.set_layers)
        ])
        self.norm = RMSNorm(D, config.norm_eps)
        self.option_scorer = nn.Sequential(nn.Linear(D, D, bias=False), nn.GELU(), nn.Linear(D, 1, bias=False))
        self.ordinal_center = nn.Linear(D, 1, bias=False)
        self.ordinal_step = nn.Sequential(nn.Linear(D, D // 2, bias=False), nn.GELU(), nn.Linear(D // 2, 1, bias=True))
        self.selective = nn.Sequential(nn.Linear(D + 4, 256), nn.GELU(), nn.Linear(256, 1))
        self.max_options = config.max_options

    LOGIT_CAP = 30.0

    def forward(self, d: torch.Tensor, opts: torch.Tensor, opt_mask: torch.Tensor, qtype: torch.Tensor) -> dict:
        """d [N, D], opts [N, O, D], opt_mask [N, O] bool, qtype [N] long.

        Runs in float32 even under autocast: the head is tiny, and bf16 quantizes large
        logits into steps of 8-32, which wrecks calibration.
        """
        with torch.autocast(device_type=d.device.type, enabled=False):
            return self._forward(d.float(), opts.float(), opt_mask, qtype)

    def _forward(self, d, opts, opt_mask, qtype) -> dict:
        N, O, _ = opts.shape
        x = torch.cat([d[:, None], opts], dim=1)
        key_mask = torch.cat([torch.ones(N, 1, dtype=torch.bool, device=d.device), opt_mask], dim=1)
        for layer in self.set_layers:
            x = layer(x, key_mask)
        x = self.norm(x)
        d_ref, o_ref = x[:, 0], x[:, 1:]

        # Categorical (choice / noul)
        logits = self.option_scorer(o_ref).squeeze(-1)
        logits = self.LOGIT_CAP * torch.tanh(logits / self.LOGIT_CAP)  # soft cap: no runaway overconfidence
        logits = logits.masked_fill(~opt_mask, float("-inf"))
        cat_logp = torch.log_softmax(logits, dim=-1)

        # Ordinal cumulative link (score)
        s = self.LOGIT_CAP * torch.tanh(self.ordinal_center(d_ref).squeeze(-1) / self.LOGIT_CAP)  # [N]
        steps = F.softplus(self.ordinal_step(o_ref).squeeze(-1).float())      # [N, O]
        base = self.ordinal_step(o_ref[:, :1]).squeeze(-1).float()[:, 0]      # [N]
        thresholds = base[:, None] + torch.cumsum(torch.cat([torch.zeros_like(steps[:, :1]), steps[:, 1:]], 1), 1)
        cdf = torch.sigmoid(thresholds - s[:, None])
        n_opts = opt_mask.sum(-1)
        idx = torch.arange(O, device=d.device)[None]
        cdf = torch.where(idx >= (n_opts[:, None] - 1), torch.ones_like(cdf), cdf)
        pmf = torch.diff(cdf, dim=1, prepend=torch.zeros_like(cdf[:, :1])).clamp_min(1e-7)
        norm = pmf.masked_fill(~opt_mask, 0.0).sum(-1, keepdim=True)
        # log of 1.0 at padded slots: torch.where later discards this branch for non-score rows,
        # and a log(0) here would still send 0 * inf = NaN gradients through it.
        ord_logp = torch.log(torch.where(opt_mask, pmf, torch.ones_like(pmf)) / norm).masked_fill(~opt_mask, float("-inf"))

        is_score = (qtype == QTYPE_SCORE)[:, None]
        logp = torch.where(is_score, ord_logp, cat_logp)
        probs = logp.exp()

        # Selective correctness head on detached features
        with torch.no_grad():
            top2 = probs.topk(min(2, O), dim=-1).values
            top1 = top2[:, 0]
            margin = top1 - (top2[:, 1] if O > 1 else torch.zeros_like(top1))
            ent = -(probs * logp.clamp_min(-30).masked_fill(~opt_mask, 0.0)).sum(-1)
            ent_norm = ent / torch.log(n_opts.clamp_min(2).float())
            logk = torch.log(n_opts.clamp_min(2).float()) / math.log(self.max_options)
            feats = torch.stack([top1, margin, ent_norm, logk], -1)
        sel_in = torch.cat([d_ref.detach(), feats], dim=-1)
        sel_logit = self.selective(sel_in).squeeze(-1).float()

        return {"logp": logp, "probs": probs, "logits": logp, "selective_logit": sel_logit, "opt_mask": opt_mask}
