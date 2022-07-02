import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf

from .linear_ar import LinearModel


class Linear_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        context_points: int,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            learning_rate=learning_rate,
            loss=loss,
            linear_window=linear_window,
            linear_shared_weights=linear_shared_weights,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
        )

        self.model = LinearModel(
            context_points, shared_weights=linear_shared_weights, d_yt=d_yt
        )

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        _, pred_len, d_yt = y_t.shape
        output = self.model(y_c, pred_len=pred_len, d_yt=d_yt)
        return (output,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
