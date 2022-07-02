from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf

try:
    from torch_geometric_temporal.nn import MTGNN
except ImportError:

    class MTGNN:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "\t  Missing `torch_geometric_temporal` package required to use MTGNN\n\
                  model. This is optional for all other model types and not installed\n\
                  with `pip install -r requirements.txt` because of CUDA versioning issues.\n\
                  Please see https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/docs/source/notes/installation.rst\n\
                  for installation instructions."
            )


class MTGNN_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        context_points: int,
        target_points: int,
        use_gcn_layer: bool = True,
        adaptive_adj_mat: bool = True,
        gcn_depth: int = 2,
        dropout_p: float = 0.2,
        node_dim: int = 40,
        dilation_exponential: int = 1,
        conv_channels: int = 32,
        subgraph_size: int = 8,
        skip_channels: int = 64,
        end_channels: int = 128,
        residual_channels: int = 32,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3,
        kernel_set: List[int] = [2, 3, 6, 7],
        kernel_size: int = 7,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        time_emb_dim: int = 0,
        loss: str = "mae",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
    ):
        assert (
            d_yc == d_yt
        ), "MTGNN requires the same number of context and target variables"
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
        subgraph_size = min(subgraph_size, d_yt)
        self.learning_rate = learning_rate

        self.time2vec = stf.Time2Vec(input_dim=d_x, embed_dim=time_emb_dim)

        self.model = MTGNN(
            gcn_true=use_gcn_layer,
            build_adj=adaptive_adj_mat,
            gcn_depth=gcn_depth,
            num_nodes=d_yt,
            kernel_set=kernel_set,
            kernel_size=kernel_size,
            dropout=dropout_p,
            subgraph_size=subgraph_size,
            node_dim=node_dim,
            conv_channels=conv_channels,
            residual_channels=residual_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            seq_length=context_points,
            in_dim=d_x + 1 if time_emb_dim == 0 else time_emb_dim + 1,
            out_dim=target_points,
            layers=layers,
            propalpha=propalpha,
            tanhalpha=tanhalpha,
            dilation_exponential=dilation_exponential,
            layer_norm_affline=True,
        )

    @property
    def eval_step_forward_kwargs(self):
        return {}

    @property
    def train_step_forward_kwargs(self):
        return {}

    def forward_model_pass(self, x_c, y_c, x_t, y_t):
        x_c = self.time2vec(x_c)
        pred_len = y_t.shape[-2]
        output = torch.zeros_like(y_t).to(y_t.device)
        # y_c = (batch, len, nodes) --> (batch, 1, nodes, len)
        y_c = y_c.transpose(-1, 1).unsqueeze(1)
        # x_c = (batch, len, d_x) --> (batch, d_x, nodes, len)
        x_c = x_c.transpose(-1, 1).unsqueeze(-2).repeat(1, 1, self.d_yc, 1)
        ctxt = torch.cat((x_c, y_c), dim=1)
        output = self.model.forward(ctxt).squeeze(-1)
        return (output,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument("--gcn_depth", type=int, default=2)
        parser.add_argument("--dropout_p", type=float, default=0.3)
        parser.add_argument("--node_dim", type=int, default=40)
        parser.add_argument("--dilation_exponential", type=int, default=1)
        parser.add_argument("--conv_channels", type=int, default=32)
        parser.add_argument("--subgraph_size", type=int, default=20)
        parser.add_argument("--skip_channels", type=int, default=64)
        parser.add_argument("--end_channels", type=int, default=128)
        parser.add_argument("--residual_channels", type=int, default=32)
        parser.add_argument("--layers", type=int, default=3)
        parser.add_argument("--propalpha", type=float, default=0.05)
        parser.add_argument("--tanhalpha", type=float, default=3.0)
        parser.add_argument("--kernel_size", type=int, default=7)
        parser.add_argument("--time_emb_dim", type=int, default=12)
