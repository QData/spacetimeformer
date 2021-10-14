import torch
from torch import nn


class DataDropout(nn.Module):
    def __init__(self, dropout=None):
        super().__init__()
        self.dropout = dropout

    def forward(self, embed):
        bs, length, d_model = embed.shape
        if self.training:
            mask = torch.bernoulli((1.0 - self.dropout) * torch.ones(bs, length, 1))
            mask.requires_grad = False
            mask = mask.to(embed.device)
            return embed * mask
        else:
            return embed

    def __repr__(self):
        return f"DataDropout(p = {self.dropout})"
