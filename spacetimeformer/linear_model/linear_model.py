import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf

from .linear_ar import LinearModel


class Linear_Forecaster(stf.Forecaster):
    def __init__(
        self,
        context_points: int,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
    ):
        super().__init__(
            l2_coeff=l2_coeff,
            learning_rate=learning_rate,
            loss=loss,
            linear_window=linear_window,
        )

        self.model = LinearModel(context_points)

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        pred_len = y_t.shape[-2]
        output = torch.zeros_like(y_t).to(y_t.device)

        for i in range(pred_len):
            inp = torch.cat((y_c[:, i:], output[:, :i]), dim=-2)
            output[:, i] = self.model.forward(inp).squeeze(1)

        return (output,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
