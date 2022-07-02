import torch
import torch.nn as nn
import torch.nn.functional as F

from .extra_layers import (
    Flatten,
    ConvBlock,
    Normalization,
    Localize,
    ReverseLocalize,
    WindowTime,
    ReverseWindowTime,
    MakeSelfMaskFromSeq,
)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        global_attention,
        local_attention,
        d_model,
        d_yc,
        time_windows=1,
        time_window_offset=0,
        d_ff=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
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

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.time_windows = time_windows
        self.time_window_offset = time_window_offset
        self.d_yc = d_yc

    def forward(self, x, self_mask_seq=None, output_attn=False):
        # uses pre-norm Transformer architecture
        attn = None
        if self.local_attention:
            # attention on tokens of each variable ind.
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_yc)
            # TODO: localize self_mask_seq
            x1, _ = self.local_attention(
                x1, x1, x1, attn_mask=self_mask_seq, output_attn=False
            )
            x1 = ReverseLocalize(x1, self.d_yc)
            x = x + self.dropout_attn_out(x1)

        if self.global_attention:
            # attention on tokens of every variable together
            x1 = self.norm2(x)

            x1 = WindowTime(
                x1,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )

            self_mask_seq = WindowTime(
                self_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, attn = self.global_attention(
                x1,
                x1,
                x1,
                attn_mask=MakeSelfMaskFromSeq(self_mask_seq),
                output_attn=output_attn,
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = ReverseWindowTime(
                self_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        x1 = self.norm3(x)
        # feedforward layers (done here as 1x1 convs)
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1
        return output, attn


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers,
        conv_layers,
        norm_layer,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(self, val_time_emb, space_emb, self_mask_seq=None, output_attn=False):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(
                x, self_mask_seq=self_mask_seq, output_attn=output_attn
            )
            if len(self.conv_layers) > i:
                if self.conv_layers[i] is not None:
                    x = self.conv_layers[i](x)
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns
