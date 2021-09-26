import torch
from torch import nn


class LinearModel(nn.Module):
    def __init__(self, context_points: int):
        super().__init__()
        self.window = context_points
        self.linear = nn.Linear(context_points, 1)

    def forward(self, y_c):
        bs, length, d_y = y_c.shape
        inp = y_c[:, -self.window :, :]
        inp = torch.cat(inp.chunk(d_y, dim=-1), dim=0)
        baseline = self.linear(inp.squeeze(-1))
        baseline = torch.cat(baseline.chunk(d_y, dim=0), dim=-1).unsqueeze(1)
        return baseline
