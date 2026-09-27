import torch
import torch.nn as nn


class VerqenEmbeddings(nn.Module):
    """Token + segment embeddings (no sqrt(d) scaling; positions come from RoPE).

    The segment embedding tells every layer which structural role a token plays,
    independently of the (escaped) text it contains.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_segments: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(num_segments, hidden_size)

    def forward(self, input_ids: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(input_ids) + self.segment_embedding(seg)
