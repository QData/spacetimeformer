import warnings

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf

from .LSTNet import LSTNet


class LSTNet_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        context_points: int,
        hidRNN: int = 100,
        hidCNN: int = 100,
        hidSkip: int = 5,
        CNN_kernel: int = 7,
        skip: int = 24,
        dropout_p: float = 0.2,
        output_fun: str = None,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        use_revin: bool = False,
    ):
        if linear_window == 0:
            warnings.warn(f"LSTNet linear window arg set to zero!")
        assert (
            d_yc == d_yt
        ), "LSTNet requires the same number of context and target variables"

        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            learning_rate=learning_rate,
            loss=loss,
            linear_window=0,
            linear_shared_weights=False,
            use_revin=use_revin,
            use_seasonal_decomp=False,
        )

        self.model = LSTNet(
            window=context_points,
            hidRNN=hidRNN,
            hidCNN=hidCNN,
            hidSkip=hidSkip,
            CNN_kernel=CNN_kernel,
            skip=skip,
            highway_window=linear_window,
            dropout=dropout_p,
            m=d_yt,
            output_fun=output_fun,
        )

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
            output[:, i] = self.model.forward(inp)

        return (output,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument("--hidRNN", type=int, default=100)
        parser.add_argument("--hidCNN", type=int, default=100)
        parser.add_argument("--hidSkip", type=int, default=5)
        parser.add_argument("--CNN_kernel", type=int, default=6)
        parser.add_argument("--skip", type=int, default=24)
        parser.add_argument("--dropout_p", type=float, default=0.2)
        parser.add_argument(
            "--output_fun", default=None, choices=[None, "sigmoid", "tanh"]
        )
