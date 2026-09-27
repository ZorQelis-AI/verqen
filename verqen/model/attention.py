import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import RMSNorm


class RotaryEmbedding(nn.Module):
    """RoPE driven by explicit per-token position ids (branch-local positions)."""

    def __init__(self, head_dim: int, max_positions: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_positions, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pos [B, L] -> cos, sin [B, 1, L, head_dim/2] (float32)."""
        pos = pos.clamp(0, self.cos.size(0) - 1)
        return self.cos[pos].unsqueeze(1), self.sin[pos].unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.float().chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class VerqenAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, qk_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim, eps) if qk_norm else nn.Identity()

    def forward(self, x, rope, attn_mask=None, past_kv=None, return_kv=False):
        """x [B, L, D]; rope = (cos, sin) for x's positions; attn_mask bool [B,1,L,Lk] or None.

        past_kv: optional (k, v) [B, H, Lp, Dh] of an already-encoded prefix (keys come first).
        """
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        new_kv = (k, v)
        if past_kv is not None:
            k = torch.cat([past_kv[0].to(k.dtype), k], dim=2)
            v = torch.cat([past_kv[1].to(v.dtype), v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = self.o_proj(out.transpose(1, 2).reshape(B, L, -1))
        return (out, new_kv) if return_kv else out
