import torch
import torch.nn as nn
import torch.nn.functional as F

from .scalenorm import ScaleNorm
from .powernorm import MaskPowerNorm


class DownsampleConv(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class VariableDownsample(nn.Module):
    def __init__(self, d_y, d_model):
        super().__init__()
        self.downConv = DownsampleConv(d_model)
        self.d_y = d_y

    def forward(self, x):
        node_chunks = x.chunk(self.d_y, dim=-2)
        downsampled = []
        for node in node_chunks:
            downsampled.append(self.downConv(node))
        return torch.cat(downsampled, dim=-2)


class Normalization(nn.Module):
    def __init__(self, method, d_model=None):
        super().__init__()
        assert method in ["layer", "scale", "batch", "power", "none"]
        if method == "layer":
            assert d_model
            self.norm = nn.LayerNorm(d_model)
        elif method == "scale":
            self.norm = ScaleNorm(d_model)
        elif method == "power":
            self.norm = MaskPowerNorm(d_model, warmup_iters=1000)
        elif method == "none":
            self.norm = lambda x: x
        else:
            assert d_model
            self.norm = nn.BatchNorm1d(d_model)
        self.method = method

    def forward(self, x):
        if self.method == "batch":
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        global_attention,
        local_attention,
        d_model,
        d_ff=None,
        dropout_ff=0.1,
        activation="relu",
        post_norm=True,
        norm="layer",
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_attention = local_attention
        self.global_attention = global_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)

        self.dropout = nn.Dropout(dropout_ff)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.post_norm = post_norm

    def forward(self, x, attn_mask=None, output_attn=False):
        # x [B, L, D]
        # see https://arxiv.org/abs/2002.04745 Figure 1
        attn = None
        if not self.post_norm:
            if self.local_attention:
                x_norm = self.norm1(x)
                new_x, _ = self.local_attention(
                    x_norm, x_norm, x_norm, attn_mask=attn_mask, output_attn=output_attn
                )
                x = x + self.dropout(new_x)

            if self.global_attention:
                x_norm = self.norm2(x)
                new_x, attn = self.global_attention(
                    x_norm, x_norm, x_norm, attn_mask=attn_mask, output_attn=output_attn
                )
                x = x + self.dropout(new_x)

            x_norm = self.norm3(x)
            x_norm = self.dropout(self.activation(self.conv1(x_norm.transpose(-1, 1))))
            x_norm = self.dropout(self.conv2(x_norm).transpose(-1, 1))
            output = x_norm + x
        else:
            if self.local_attention:
                new_x, _ = self.local_attention(
                    x,
                    x,
                    x,
                    attn_mask=attn_mask,
                    output_attn=output_attn,
                )
                x = x + self.dropout(new_x)
                x = self.norm1(x)
            if self.global_attention:
                new_x, attn = self.global_attention(
                    x, x, x, attn_mask=attn_mask, output_attn=output_attn
                )
                x = x + self.dropout(new_x)
                x = self.norm2(x)

            new_x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
            new_x = self.dropout(self.conv2(new_x).transpose(-1, 1))
            x = x + new_x
            output = self.norm3(x)

        return output, attn


from .data_dropout import DataDropout


class Encoder(nn.Module):
    def __init__(
        self, attn_layers, conv_layers, norm_layer, emb_dropout=0.0, data_dropout=0.0
    ):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layer = norm_layer

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.data_dropout = DataDropout(data_dropout)

    def forward(self, val_time_emb, space_emb, attn_mask=None, output_attn=False):
        # x [B, L, D]
        x = val_time_emb + space_emb
        x = self.data_dropout(self.emb_dropout(x))
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, output_attn=output_attn)
            if len(self.conv_layers) > i:
                if self.conv_layers[i] is not None:
                    x = conv_layer(x)
            attns.append(attn)
        if self.norm_layer:
            x = self.norm_layer(x)

        return x, attns
