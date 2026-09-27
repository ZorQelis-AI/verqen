import torch.nn as nn

from .attention import VerqenAttention
from .mlp import SwiGLUMLP
from .normalization import RMSNorm


class VerqenBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, hidden_size, num_heads, head_dim, ffn_size, qk_norm=True, eps=1e-6):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size, eps)
        self.attention = VerqenAttention(hidden_size, num_heads, head_dim, qk_norm, eps)
        self.ffn_norm = RMSNorm(hidden_size, eps)
        self.ffn = SwiGLUMLP(hidden_size, ffn_size)

    def forward(self, x, rope, attn_mask=None, past_kv=None, return_kv=False):
        a = self.attention(self.attention_norm(x), rope, attn_mask, past_kv, return_kv)
        if return_kv:
            a, kv = a
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return (x, kv) if return_kv else x
