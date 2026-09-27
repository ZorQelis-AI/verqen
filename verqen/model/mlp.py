import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int):
        super().__init__()
        self.gate_up = nn.Linear(hidden_size, 2 * ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
