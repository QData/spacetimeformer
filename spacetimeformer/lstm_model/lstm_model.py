import random

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import spacetimeformer as stf


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )

    def forward(self, x_context: torch.Tensor):
        outputs, (hidden, cell) = self.lstm(x_context)
        return hidden, cell


class LSTM_Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 1,
        input_dim: int = 1,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_t, hidden, cell):
        output, (hidden, cell) = self.lstm(x_t, (hidden, cell))
        y_t1 = self.fc(output)
        return y_t1, hidden, cell


class LSTM_Seq2Seq(nn.Module):
    def __init__(self, t2v: stf.Time2Vec, encoder: LSTM_Encoder, decoder: LSTM_Decoder):
        super().__init__()
        self.t2v = t2v
        self.encoder = encoder
        self.decoder = decoder

    def _merge(self, x, y):
        return torch.cat((x, y), dim=-1)

    def forward(
        self,
        x_context,
        y_context,
        x_target,
        y_target,
        teacher_forcing_prob,
    ):
        if self.t2v is not None:
            x_context = self.t2v(x_context)
            x_target = self.t2v(x_target)

        pred_len = y_target.shape[1]
        batch_size = y_target.shape[0]
        y_dim = y_target.shape[2]
        outputs = -torch.ones(batch_size, pred_len, y_dim).to(y_target.device)
        merged_context = self._merge(x_context, y_context)
        hidden, cell = self.encoder(merged_context)

        decoder_input = (
            self._merge(x_context[:, -1], torch.zeros_like(y_target[:, 0]))
            .unsqueeze(1)
            .to(y_context.device)
        )

        for t in range(0, pred_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)

            decoder_y = (
                y_target[:, t].unsqueeze(1)
                if random.random() < teacher_forcing_prob
                else output
            )
            decoder_input = self._merge(x_target[:, t].unsqueeze(1), decoder_y)
        return outputs


class LSTM_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_x: int = 6,
        d_yc: int = 1,
        d_yt: int = 1,
        time_emb_dim: int = 0,
        n_layers: int = 2,
        hidden_dim: int = 32,
        dropout_p: float = 0.2,
        # training
        learning_rate: float = 1e-3,
        teacher_forcing_prob: float = 0.5,
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
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
        )
        self.t2v = stf.Time2Vec(input_dim=d_x, embed_dim=time_emb_dim * d_x)

        time_dim = time_emb_dim * d_x if time_emb_dim > 0 else d_x

        self.encoder = LSTM_Encoder(
            input_dim=time_dim + d_yc,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.decoder = LSTM_Decoder(
            output_dim=d_yt,
            input_dim=time_dim + d_yt,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_p,
        )
        self.model = LSTM_Seq2Seq(self.t2v, self.encoder, self.decoder).to(self.device)

        self.teacher_forcing_prob = teacher_forcing_prob

    @property
    def train_step_forward_kwargs(self):
        return {"force": self.teacher_forcing_prob}

    @property
    def eval_step_forward_kwargs(self):
        return {"force": 0.0}

    def forward_model_pass(self, x_c, y_c, x_t, y_t, force=None):
        assert force is not None
        with torch.no_grad():
            # need to normalize y_t in LSTM because it is sometimes used
            # as input (teacher forcing). important to not leak the target
            # stats (update_stats = False).
            y_t = self.revin(y_t, mode="norm", update_stats=False)
        preds = self.model.forward(x_c, y_c, x_t, y_t, teacher_forcing_prob=force)
        return (preds,)

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument(
            "--hidden_dim",
            type=int,
            default=128,
            help="Hidden dimension for LSTM network.",
        )
        parser.add_argument(
            "--n_layers",
            type=int,
            default=2,
            help="Number of stacked LSTM layers",
        )
        parser.add_argument(
            "--dropout_p",
            type=float,
            default=0.3,
            help="Dropout fraction for LSTM.",
        )
        parser.add_argument(
            "--time_emb_dim",
            type=int,
            default=12,
            help="Embedding dimension for Tim2Vec encoding. Set to zero to disable T2V.",
        )
