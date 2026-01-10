import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)
