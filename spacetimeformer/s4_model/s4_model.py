import warnings

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf

from . import s4_standalone


class S4_Forecaster(stf.Forecaster):
    def __init__(
        self,
        context_points: int,
        target_points: int,
        d_state: int = 64,
        d_model: int = 256,
        d_x: int = 6,
        d_yc: int = 1,
        d_yt: int = 1,
        layers: int = 1,
        time_emb_dim: int = 6,
        channels: int = 1,
        dropout_p: float = 0.2,
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

        self.t2v = stf.Time2Vec(input_dim=d_x, embed_dim=time_emb_dim * d_x)
        self.emb = nn.Linear(d_yc + (time_emb_dim * d_x), d_model)
        self.given = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(layers)])

        self.s4s = nn.ModuleList(
            [
                s4_standalone.S4(
                    d_model=d_model,
                    l_max=context_points + target_points,
                    channels=channels,
                    bidirectional=False,
                    transposed=False,
                    d_state=d_state,
                    dropout=dropout_p,
                )
                for _ in range(layers)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_p)

        self.out = nn.Linear(d_model, d_yt)

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        dev = x_c.device
        b, lc, d_yc = y_c.shape
        _, lt, d_yt = y_t.shape

        x = torch.cat((x_c, x_t), dim=-2)
        x_t2v = self.t2v(x)
        # pad context sequence with zeros
        y = torch.cat((y_c, torch.zeros((b, lt, d_yc)).to(dev)), dim=-2)
        y_x = torch.cat((y, x_t2v), dim=-1)
        val_time_emb = self.emb(y_x)
        given = (
            torch.cat((torch.ones((b, lc)), torch.zeros((b, lt))), dim=-1)
            .long()
            .to(dev)
        )
        given_emb = self.given(given)

        seq = val_time_emb + given_emb
        for norm, s4 in zip(self.norms, self.s4s):
            seq = seq + self.dropout(s4(norm(seq))[0])
        seq = seq[:, lc:, :]  # cut off context part
        out = self.out(seq)
        return (out,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument("--d_state", type=int, default=64)
        parser.add_argument("--d_model", type=int, default=256)
        parser.add_argument("--layers", type=int, default=2)
        parser.add_argument("--time_emb_dim", type=int, default=12)
        parser.add_argument("--channels", type=int, default=1)
        parser.add_argument("--dropout_p", type=float, default=0.1)
