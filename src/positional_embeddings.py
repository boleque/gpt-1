import torch
import torch.nn as nn

class PositionalEmbeddings(nn.Module):

    def __init__(self, max_seq_len: int, emb_sizE: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int):
        positions = torch.arange(0, seq_len)
        return self.embedding(positions)
