import torch
import torch.nn as nn
from multiheadattention import MultiHeadAttention
from feedforward import FeedForward


class Decoder(nn.Module):

    def __init__(self, num_heads: int, emb_size: int, head_size: int,
                 max_seq_len: int, dropout: float = 0.1):
        super().__init__()

        # Multi-head attention
        self.attention = MultiHeadAttention(num_heads, emb_size, head_size,
                                           max_seq_len, dropout)

        # Feed-forward
        self.ffn = FeedForward(emb_size, dropout)

        self.norm1 = nn.LayerNorm(emb_size)

        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Multi-head attention + residual connection
        attention_out = self.attention(x)
        x = x + attention_out  # Residual connection

        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + ffn_out  # Residual connection

        x = self.norm2(x)

        return x
