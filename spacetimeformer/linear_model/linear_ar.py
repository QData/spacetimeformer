import math

import torch
from torch import nn
from einops import rearrange


class LinearModel(nn.Module):
    def __init__(
        self,
        context_points: int,
        shared_weights: bool = False,
        d_yt: int = 7,
        pad_val: float = None,
    ):
        super().__init__()

        if not shared_weights:
            assert d_yt is not None
            layer_count = d_yt
        else:
            layer_count = 1

        self.weights = nn.Parameter(
            torch.ones((context_points, layer_count)), requires_grad=True
        )
        self.bias = nn.Parameter(torch.ones((layer_count)), requires_grad=True)

        d = math.sqrt(1.0 / context_points)
        self.weights.data.uniform_(-d, d)
        self.bias.data.uniform_(-d, d)

        self.window = context_points
        self.shared_weights = shared_weights
        self.d_yt = d_yt

        self.pad_val = pad_val

    def forward(self, y_c: torch.Tensor, pred_len: int, d_yt: int = None):
        batch, _, d_yc = y_c.shape
        d_yt = d_yt or self.d_yt
        output = torch.zeros(batch, pred_len, d_yt).to(y_c.device)
        for i in range(pred_len):
            # inp = torch.cat((y_c[:, min(-self.window + i, 0):, :], output[:, :i, :]), dim=1)
            inp = torch.cat((y_c, output[:, :i, :]), dim=1)[:, -self.window :, :]
            if self.pad_val is not None:
                mask = (inp != self.pad_val).all(-1, keepdims=True)
                inp *= mask
            output[:, i, :] = self._inner_forward(inp)
        return output

    def _inner_forward(self, inp, param_num=0):
        batch, length, _ = inp.shape
        window = min(length, self.window)
        if self.shared_weights:
            inp = rearrange(inp, "batch length dy -> (batch dy) length 1")
        baseline = (self.weights[-window:] * inp[:, -window:, :]).sum(1) + self.bias
        if self.shared_weights:
            baseline = rearrange(baseline, "(batch dy) 1 -> batch dy", batch=batch)
        return baseline
