import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .extra_layers import (
    Normalization,
    Localize,
    ReverseLocalize,
    WindowTime,
    ReverseWindowTime,
    MakeSelfMaskFromSeq,
    MakeCrossMaskFromSeq,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        local_self_attention,
        global_cross_attention,
        local_cross_attention,
        d_model,
        d_yt,
        d_yc,
        time_windows=1,
        time_window_offset=0,
        d_ff=None,
        dropout_ff=0.1,
        dropout_attn_out=0.0,
        activation="relu",
        norm="layer",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_self_attention = local_self_attention
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        if local_cross_attention is not None and d_yc != d_yt:
            assert d_yt < d_yc
            warnings.warn(
                "The implementation of Local Cross Attn with exogenous variables \n\
                makes an unintuitive assumption about variable order. Please see \n\
                spacetimeformer_model.nn.decoder.DecoderLayer source code and comments"
            )
            """
            The unintuitive part is that if there are N variables in the context
            sequence (the encoder input) and K (K < N) variables in the target sequence
            (the decoder input), then this implementation of Local Cross Attn
            assumes that the first K variables in the context correspond to the
            first K in the target. This means that if the context sequence is shape 
            (batch, length, N), then context[:, :, :K] gets you the context of the
            K target variables (target[..., i] is the same variable
            as context[..., i]). If this isn't true the model will still train but
            you will be connecting variables by cross attention in a very arbitrary
            way. Note that the built-in CSVDataset *does* account for this and always
            puts the target variables in the same order in the lowest indices of both
            sequences. ** If your target variables do not appear in the context sequence
            Local Cross Attention should almost definitely be turned off
            (--local_cross_attn none) **.
            """

        self.local_cross_attention = local_cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = Normalization(method=norm, d_model=d_model)
        self.norm5 = Normalization(method=norm, d_model=d_model)

        self.dropout_ff = nn.Dropout(dropout_ff)
        self.dropout_attn_out = nn.Dropout(dropout_attn_out)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.time_windows = time_windows
        self.time_window_offset = time_window_offset
        self.d_yt = d_yt
        self.d_yc = d_yc

    def forward(
        self, x, cross, self_mask_seq=None, cross_mask_seq=None, output_cross_attn=False
    ):
        # pre-norm Transformer architecture
        attn = None
        if self.local_self_attention:
            # self attention on each variable in target sequence ind.
            assert self_mask_seq is None
            x1 = self.norm1(x)
            x1 = Localize(x1, self.d_yt)
            x1, _ = self.local_self_attention(x1, x1, x1, attn_mask=self_mask_seq)
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_self_attention:
            x1 = self.norm2(x)
            x1 = WindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = WindowTime(
                self_mask_seq,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, _ = self.global_self_attention(
                x1,
                x1,
                x1,
                attn_mask=MakeSelfMaskFromSeq(self_mask_seq),
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            self_mask_seq = ReverseWindowTime(
                self_mask_seq,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        if self.local_cross_attention:
            # cross attention between target/context on each variable ind.
            assert cross_mask_seq is None
            x1 = self.norm3(x)
            bs, *_ = x1.shape
            x1 = Localize(x1, self.d_yt)
            # see above warnings and explanations about a potential
            # silent bug here.
            cross_local = Localize(cross, self.d_yc)[: self.d_yt * bs]
            x1, _ = self.local_cross_attention(
                x1,
                cross_local,
                cross_local,
                attn_mask=cross_mask_seq,
            )
            x1 = ReverseLocalize(x1, self.d_yt)
            x = x + self.dropout_attn_out(x1)

        if self.global_cross_attention:
            x1 = self.norm4(x)
            x1 = WindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross = WindowTime(
                cross,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = WindowTime(
                cross_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1, attn = self.global_cross_attention(
                x1,
                cross,
                cross,
                attn_mask=MakeCrossMaskFromSeq(self_mask_seq, cross_mask_seq),
                output_attn=output_cross_attn,
            )
            cross = ReverseWindowTime(
                cross,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            cross_mask_seq = ReverseWindowTime(
                cross_mask_seq,
                dy=self.d_yc,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x1 = ReverseWindowTime(
                x1,
                dy=self.d_yt,
                windows=self.time_windows,
                window_offset=self.time_window_offset,
            )
            x = x + self.dropout_attn_out(x1)

        x1 = self.norm5(x)
        # feedforward layers as 1x1 convs
        x1 = self.dropout_ff(self.activation(self.conv1(x1.transpose(-1, 1))))
        x1 = self.dropout_ff(self.conv2(x1).transpose(-1, 1))
        output = x + x1

        return output, attn


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, emb_dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm_layer = norm_layer
        self.emb_dropout = nn.Dropout(emb_dropout)

    def forward(
        self,
        val_time_emb,
        space_emb,
        cross,
        self_mask_seq=None,
        cross_mask_seq=None,
        output_cross_attn=False,
    ):
        x = self.emb_dropout(val_time_emb) + self.emb_dropout(space_emb)

        attns = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(
                x,
                cross,
                self_mask_seq=self_mask_seq,
                cross_mask_seq=cross_mask_seq,
                output_cross_attn=output_cross_attn,
            )
            attns.append(attn)

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        return x, attns
