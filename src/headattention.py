import torch
import torch.nn as nn

class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int):
        super().__init__()

        self.wK = nn.Linear(emb_size, head_size)
        self.wQ = nn.Linear(emb_size, head_size)
        self.wV = nn.Linear(emb_size, head_size)

        self.head_size = head_size

        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, emb_size = x.shape

        K = self.wK(x)
        Q = self.wQ(x)
        V = self.wV(x)

        attention = Q @ K.transpose(-2, -1)

        attention = attention / (self.head_size ** 0.5)

        mask = self.mask[:seq_len, :seq_len]

        attention = attention.masked_fill(mask == 0, float('-inf'))

        attention = F.softmax(attention, dim=-1)

        return attention @ V
