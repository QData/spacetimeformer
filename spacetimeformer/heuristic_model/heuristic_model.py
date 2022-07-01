import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf


class Heuristic_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        context_points: int,
        target_points: int,
        loss: str = "mse",
        method: str = "repeat_last",
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=0,
            learning_rate=0,
            loss=loss,
            linear_window=0,
            linear_shared_weights=True,
            use_revin=False,
            use_seasonal_decomp=False,
        )
        self.method = method
        self.at_least_one_param = nn.Linear(1, 1)
        # self.automatic_optimization = False

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        batch, pred_len, d_yt = y_t.shape

        if self.method == "repeat_last":
            output = (
                torch.ones_like(y_t).to(y_c.device)
                * y_c[:, -1, :].unsqueeze(1).detach()
            )
        elif self.method == "repeat_mean":
            output = (
                torch.ones_like(y_t).to(y_c.device) * y_c.mean(1, keepdim=True).detach()
            )

        # hack to work with lightning optimization w/o learnable params
        # (i'm sure there's a better way to do this...)
        output = output + 0.0 * self.at_least_one_param(torch.zeros(1).to(y_c.device))

        return (output,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument(
            "--method",
            type=str,
            choices=["repeat_last", "repeat_mean"],
            default="repeat_last",
        )
