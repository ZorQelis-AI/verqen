import torch
import torch.nn as nn

from verqen.config import ModelConfig
from .attention import RotaryEmbedding
from .block import VerqenBlock
from .embeddings import VerqenEmbeddings
from .normalization import RMSNorm


class VerqenTransformer(nn.Module):
    """Bidirectional-within-segment encoder backbone (no decision heads)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        c = config
        self.embeddings = VerqenEmbeddings(c.vocab_size, c.hidden_size, c.num_segments)
        self.layers = nn.ModuleList([
            VerqenBlock(c.hidden_size, c.num_heads, c.head_dim, c.ffn_size, c.qk_norm, c.norm_eps)
            for _ in range(c.num_layers)
        ])
        self.final_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.rope = RotaryEmbedding(c.head_dim, max(c.max_seq_len, 4096), c.rope_base)

    def forward(self, input_ids, seg, pos, attn_mask=None, past_kvs=None, return_kvs=False):
        """input_ids/seg/pos [B, L]; attn_mask bool [B,1,L,Lk] or None (full attention).

        Returns final-normed hidden states [B, L, D] (and per-layer (k, v) when return_kvs).
        """
        x = self.embeddings(input_ids, seg)
        rope = self.rope(pos)
        kvs = []
        for i, layer in enumerate(self.layers):
            past = past_kvs[i] if past_kvs is not None else None
            if return_kvs:
                x, kv = layer(x, rope, attn_mask, past, True)
                kvs.append(kv)
            else:
                x = layer(x, rope, attn_mask, past)
        x = self.final_norm(x)
        return (x, kvs) if return_kvs else x
