import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Normalization


class DecoderLayer(nn.Module):
    def __init__(
        self,
        global_self_attention,
        local_self_attention,
        global_cross_attention,
        local_cross_attention,
        d_model,
        d_ff=None,
        dropout_ff=0.1,
        activation="relu",
        post_norm=True,
        norm="layer",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.local_self_attention = local_self_attention
        self.global_self_attention = global_self_attention
        self.global_cross_attention = global_cross_attention
        self.local_cross_attention = local_cross_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = Normalization(method=norm, d_model=d_model)
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = Normalization(method=norm, d_model=d_model)
        self.norm5 = Normalization(method=norm, d_model=d_model)

        self.dropout = nn.Dropout(dropout_ff)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.post_norm = post_norm

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        # see https://arxiv.org/abs/2002.04745 Figure 1
        if self.post_norm:
            if self.local_self_attention:
                x = x + self.dropout(
                    self.local_self_attention(x, x, x, attn_mask=x_mask)[0]
                )
                x = self.norm1(x)

            if self.global_self_attention:
                x = x + self.dropout(
                    self.global_self_attention(x, x, x, attn_mask=x_mask)[0]
                )
                x = self.norm2(x)

            if self.local_cross_attention:
                x = x + self.dropout(
                    self.local_cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
                )
                x = self.norm3(x)

            if self.global_cross_attention:
                x = x + self.dropout(
                    self.global_cross_attention(x, cross, cross, attn_mask=cross_mask)[
                        0
                    ]
                )
                x = self.norm4(x)

            y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
            output = self.norm5(x + y)
        else:
            if self.local_self_attention:
                x_norm = self.norm1(x)
                x = x + self.dropout(
                    self.local_self_attention(x_norm, x_norm, x_norm, attn_mask=x_mask)[
                        0
                    ]
                )

            if self.global_self_attention:
                x_norm = self.norm2(x)
                x = x + self.dropout(
                    self.global_self_attention(
                        x_norm, x_norm, x_norm, attn_mask=x_mask
                    )[0]
                )

            if self.local_cross_attention:
                x_norm = self.norm3(x)
                x = x + self.dropout(
                    self.local_cross_attention(
                        x_norm, cross, cross, attn_mask=cross_mask
                    )[0]
                )

            if self.global_cross_attention:
                x_norm = self.norm4(x)
                x = x + self.dropout(
                    self.global_cross_attention(
                        x_norm, cross, cross, attn_mask=cross_mask
                    )[0]
                )

            x_norm = self.norm5(x)
            x_norm = self.dropout(self.activation(self.conv1(x_norm.transpose(-1, 1))))
            x_norm = self.dropout(self.conv2(x_norm).transpose(-1, 1))
            output = x + x_norm

        return output


from .data_dropout import DataDropout


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, emb_dropout=0.0, data_dropout=0.0):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.data_dropout = DataDropout(data_dropout)

    def forward(self, val_time_emb, space_emb, cross, x_mask=None, cross_mask=None):
        x = self.data_dropout(self.emb_dropout(val_time_emb + space_emb))
        for i, layer in enumerate(self.layers):
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        return x
