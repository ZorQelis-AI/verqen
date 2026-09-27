"""VerqenModel: encoder backbone + MLM pretraining head + decision head."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from verqen.config import ModelConfig
from verqen.model.attention_mask import build_attention_mask
from verqen.model.decision_head import DecisionHead
from verqen.model.normalization import RMSNorm
from verqen.model.transformer import VerqenTransformer


@dataclass
class DecisionOutput:
    probs: torch.Tensor
    logits: torch.Tensor
    question_type: str = "choice"
    expected_score: Optional[torch.Tensor] = None
    confidence: float = 1.0
    should_escalate: bool = False
    p_correct: float = 1.0


class MLMHead(nn.Module):
    def __init__(self, config: ModelConfig, embedding: nn.Embedding):
        super().__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self._embedding = [embedding]  # tied weight; kept out of the module tree so it is saved once
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(F.gelu(self.transform(h)))
        return F.linear(h, self._embedding[0].weight, self.bias)


class VerqenModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = VerqenTransformer(config)
        self.decision_head = DecisionHead(config)
        self.mlm_head = MLMHead(config, self.transformer.embeddings.token_embedding)
        self.apply(self._init_weights)
        resid_std = config.init_std / math.sqrt(2 * (config.num_layers + config.set_layers))
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=resid_std)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "VerqenModel":
        return cls(config)

    def count_parameters(self) -> dict[str, int]:
        emb = self.transformer.embeddings.token_embedding.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        mlm_only = sum(p.numel() for p in self.mlm_head.parameters())
        return {
            "total": total,
            "embedding": emb,
            "backbone_non_embedding": sum(p.numel() for p in self.transformer.parameters()) - emb,
            "decision_head": sum(p.numel() for p in self.decision_head.parameters()),
            "mlm_head_only": mlm_only,
            "inference_total": total - mlm_only,
        }

    # ── Backbone ──────────────────────────────────────────────────────────
    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch.get("attn_mask")
        if mask is None and "doc" in batch:
            mask = build_attention_mask(batch["seg"], batch["doc"], batch["qid"], batch["oid"])
        return self.transformer(batch["input_ids"], batch["seg"], batch["pos"], mask)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(batch)

    # ── Pretraining ───────────────────────────────────────────────────────
    def forward_mlm(self, input_ids, seg, pos, mlm_index, attn_mask=None) -> torch.Tensor:
        """Return vocabulary logits at the flattened masked positions ``mlm_index``."""
        h = self.transformer(input_ids, seg, pos, attn_mask)
        h = h.reshape(-1, h.size(-1))[mlm_index]
        return self.mlm_head(h)

    # ── Decisions ─────────────────────────────────────────────────────────
    def decision_readout(self, hidden: torch.Tensor, batch: dict[str, torch.Tensor], head=None) -> dict[str, torch.Tensor]:
        """Gather DECIDE / option vectors from ``hidden`` [B, L, D] and run the head
        (``head`` may be a compiled replacement for ``self.decision_head``)."""
        H = hidden.reshape(-1, hidden.size(-1))
        d = H[batch["dec_flat"]]
        tok_gid = batch["tok_opt_gid"]
        valid = tok_gid >= 0
        n_opt = int(batch["n_opt_total"])
        sums = torch.zeros(n_opt, H.size(-1), device=H.device, dtype=torch.float32)
        sums.index_add_(0, tok_gid[valid], H[valid].float())
        counts = torch.bincount(tok_gid[valid], minlength=n_opt).clamp_min(1).unsqueeze(-1)
        opt_vec = (sums / counts).to(H.dtype)
        table = batch["opt_table"]
        opts = opt_vec[table.clamp_min(0)] * batch["opt_mask"].unsqueeze(-1).to(H.dtype)
        return (head or self.decision_head)(d, opts, batch["opt_mask"], batch["qtype"])

    def forward_decisions(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decision_readout(self.encode(batch), batch)

    # Back-compat alias used by older scripts
    def decide_batch(self, batch: dict[str, torch.Tensor], *_: Any) -> dict[str, torch.Tensor]:
        return self.forward_decisions(batch)
