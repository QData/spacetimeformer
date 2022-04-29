import torch
from torch import nn
from einops import rearrange


class LinearModel(nn.Module):
    def __init__(self, context_points: int):
        super().__init__()
        self.window = context_points
        self.linear = nn.Linear(context_points, 1)

    def forward(self, y_c):
        batch, length, dy = y_c.shape
        inp = y_c[:, -self.window :, :]
        inp = rearrange(inp, "batch length dy -> (batch dy) length")
        baseline = self.linear(inp)
        baseline = rearrange(
            baseline, "(batch dy) length -> batch length dy", batch=batch
        )
        return baseline
