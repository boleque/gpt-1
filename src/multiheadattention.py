import torch
import torch.nn as nn
from headattention import HeadAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, emb_size: int, head_size: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        self.heads = nn.ModuleList([
            HeadAttention(emb_size, head_size, max_seq_len)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(head_size * num_heads, emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_outputs = [head(x) for head in self.heads]

        out = torch.cat(head_outputs, dim=-1)

        out = self.proj(out)

        return self.dropout(out)
