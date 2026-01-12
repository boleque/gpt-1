import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(emb_size, 4 * emb_size)

        self.relu = nn.ReLU()

        self.linear2 = nn.Linear(4 * emb_size, emb_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)    # (batch_size, seq_len, 4 * emb_size)
        x = self.relu(x)       # (batch_size, seq_len, 4 * emb_size)
        x = self.linear2(x)    # (batch_size, seq_len, emb_size)
        x = self.dropout(x)    # (batch_size, seq_len, emb_size)

        return x
