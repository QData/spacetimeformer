from functools import partial
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as pyd
from einops import rearrange, repeat

from .extra_layers import ConvBlock, Normalization, FoldForPred
from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from .attn import (
    FullAttention,
    ProbAttention,
    AttentionLayer,
    PerformerAttention,
)
from .embed import Embedding
from .data_dropout import ReconstructionDropout


class Spacetimeformer(nn.Module):
    def __init__(
        self,
        d_yc: int = 1,
        d_yt: int = 1,
        d_x: int = 4,
        max_seq_len: int = None,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys: int = 30,
        d_values: int = 30,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 3,
        d_ff: int = 800,
        start_token_len: int = 0,
        time_emb_dim: int = 6,
        dropout_emb: float = 0.1,
        dropout_attn_matrix: float = 0.0,
        dropout_attn_out: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_qkv: float = 0.0,
        pos_emb_type: str = "abs",
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_attn_kernel: str = "relu",
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        embed_method: str = "spatio-temporal",
        activation: str = "gelu",
        norm: str = "batch",
        use_final_norm: bool = True,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        device=torch.device("cuda:0"),
        null_value: float = None,
        pad_value: float = None,
        out_dim: int = None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.1,
        recon_mask_drop_standard: float = 0.2,
        recon_mask_drop_full: float = 0.05,
        verbose: bool = True,
    ):
        super().__init__()
        if e_layers:
            assert intermediate_downsample_convs <= e_layers - 1
        if embed_method == "temporal":
            assert (
                local_self_attn == "none"
            ), "local attention not compatible with Temporal-only embedding"
            assert (
                local_cross_attn == "none"
            ), "Local Attention not compatible with Temporal-only embedding"
            split_length_into = 1
        else:
            split_length_into = d_yc

        self.pad_value = pad_value
        self.embed_method = embed_method
        self.d_yt = d_yt
        self.d_yc = d_yc
        self.start_token_len = start_token_len

        # generates random masks of context sequence for encoder to reconstruct
        recon_dropout = ReconstructionDropout(
            drop_full_timesteps=recon_mask_drop_full,
            drop_standard=recon_mask_drop_standard,
            drop_seq=recon_mask_drop_seq,
            drop_max_seq_len=recon_mask_max_seq_len,
            skip_all_drop=recon_mask_skip_all,
        )

        # embeddings. seperate enc/dec in case the variable indices are not aligned
        self.enc_embedding = Embedding(
            d_y=d_yc,
            d_x=d_x,
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            downsample_convs=initial_downsample_convs,
            method=embed_method,
            null_value=null_value,
            pad_value=pad_value,
            start_token_len=start_token_len,
            is_encoder=True,
            position_emb=pos_emb_type,
            max_seq_len=max_seq_len,
            data_dropout=recon_dropout,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
        )
        self.dec_embedding = Embedding(
            d_y=d_yt,
            d_x=d_x,
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            downsample_convs=initial_downsample_convs,
            method=embed_method,
            null_value=null_value,
            pad_value=pad_value,
            start_token_len=start_token_len,
            is_encoder=False,
            position_emb=pos_emb_type,
            max_seq_len=max_seq_len,
            data_dropout=None,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
        )

        # Select Attention Mechanisms
        attn_kwargs = {
            "d_model": d_model,
            "n_heads": n_heads,
            "d_qk": d_queries_keys,
            "d_v": d_values,
            "dropout_qkv": dropout_qkv,
            "dropout_attn_matrix": dropout_attn_matrix,
            "attn_factor": attn_factor,
            "performer_attn_kernel": performer_attn_kernel,
            "performer_redraw_interval": performer_redraw_interval,
        }

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    global_attention=self._attn_switch(
                        global_self_attn,
                        **attn_kwargs,
                    ),
                    local_attention=self._attn_switch(
                        local_self_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    d_yc=d_yc if embed_method == "spatio-temporal" else 1,
                    time_windows=attn_time_windows,
                    # encoder layers alternate using shifted windows, if applicable
                    time_window_offset=2
                    if use_shifted_time_windows and (l % 2 == 1)
                    else 0,
                    d_ff=d_ff,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(e_layers)
            ],
            conv_layers=[
                ConvBlock(split_length_into=split_length_into, d_model=d_model)
                for l in range(intermediate_downsample_convs)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        # Decoder
        self.decoder = Decoder(
            layers=[
                DecoderLayer(
                    global_self_attention=self._attn_switch(
                        global_self_attn,
                        **attn_kwargs,
                    ),
                    local_self_attention=self._attn_switch(
                        local_self_attn,
                        **attn_kwargs,
                    ),
                    global_cross_attention=self._attn_switch(
                        global_cross_attn,
                        **attn_kwargs,
                    ),
                    local_cross_attention=self._attn_switch(
                        local_cross_attn,
                        **attn_kwargs,
                    ),
                    d_model=d_model,
                    time_windows=attn_time_windows,
                    # decoder layers alternate using shifted windows, if applicable
                    time_window_offset=2
                    if use_shifted_time_windows and (l % 2 == 1)
                    else 0,
                    d_ff=d_ff,
                    # temporal embedding effectively has 1 variable
                    # for the purposes of time windowing.
                    d_yt=d_yt if embed_method == "spatio-temporal" else 1,
                    d_yc=d_yc if embed_method == "spatio-temporal" else 1,
                    dropout_ff=dropout_ff,
                    dropout_attn_out=dropout_attn_out,
                    activation=activation,
                    norm=norm,
                )
                for l in range(d_layers)
            ],
            norm_layer=Normalization(norm, d_model=d_model) if use_final_norm else None,
            emb_dropout=dropout_emb,
        )

        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint(f"GlobalSelfAttn: {self.decoder.layers[0].global_self_attention}")
        qprint(f"GlobalCrossAttn: {self.decoder.layers[0].global_cross_attention}")
        qprint(f"LocalSelfAttn: {self.decoder.layers[0].local_self_attention}")
        qprint(f"LocalCrossAttn: {self.decoder.layers[0].local_cross_attention}")
        qprint(f"Using Embedding: {embed_method}")
        qprint(f"Time Emb Dim: {time_emb_dim}")
        qprint(f"Space Embedding: {self.dec_embedding.use_space}")
        qprint(f"Time Embedding: {self.dec_embedding.use_time}")
        qprint(f"Val Embedding: {self.dec_embedding.use_val}")
        qprint(f"Given Embedding: {self.dec_embedding.use_given}")
        qprint(f"Null Value: {self.dec_embedding.null_value}")
        qprint(f"Pad Value: {self.dec_embedding.pad_value}")
        qprint(f"Reconstruction Dropout: {self.enc_embedding.data_drop}")

        if not out_dim:
            out_dim = 1 if self.embed_method == "spatio-temporal" else d_yt
            recon_dim = 1 if self.embed_method == "spatio-temporal" else d_yc

        # final linear layers turn Transformer output into predictions
        self.forecaster = nn.Linear(d_model, out_dim, bias=True)
        self.reconstructor = nn.Linear(d_model, recon_dim, bias=True)
        self.classifier = nn.Linear(d_model, d_yc, bias=True)

    def forward(
        self,
        enc_x,
        enc_y,
        dec_x,
        dec_y,
        output_attention=False,
    ):
        # embed context sequence
        enc_vt_emb, enc_s_emb, enc_var_idxs, enc_mask_seq = self.enc_embedding(
            y=enc_y, x=enc_x
        )

        # encode context sequence
        enc_out, enc_self_attns = self.encoder(
            val_time_emb=enc_vt_emb,
            space_emb=enc_s_emb,
            self_mask_seq=enc_mask_seq,
            output_attn=output_attention,
        )

        # embed target sequence
        dec_vt_emb, dec_s_emb, _, dec_mask_seq = self.dec_embedding(y=dec_y, x=dec_x)
        if enc_mask_seq is not None:
            enc_dec_mask_seq = enc_mask_seq.clone()
        else:
            enc_dec_mask_seq = enc_mask_seq

        # decode target sequence w/ encoded context
        dec_out, dec_cross_attns = self.decoder(
            val_time_emb=dec_vt_emb,
            space_emb=dec_s_emb,
            cross=enc_out,
            self_mask_seq=dec_mask_seq,
            cross_mask_seq=enc_dec_mask_seq,
            output_cross_attn=output_attention,
        )

        # forecasting predictions
        forecast_out = self.forecaster(dec_out)
        # reconstruction predictions
        recon_out = self.reconstructor(enc_out)
        if self.embed_method == "spatio-temporal":
            # fold flattened spatiotemporal format back into (batch, length, d_yt)
            forecast_out = FoldForPred(forecast_out, dy=self.d_yt)
            recon_out = FoldForPred(recon_out, dy=self.d_yc)
        forecast_out = forecast_out[:, self.start_token_len :, :]

        if enc_var_idxs is not None:
            # note that detaching the input like this means the transformer layers
            # are not optimizing for classification accuracy (but the linear classifier
            # layer still is). This is just a test to see how much unique spatial info
            # remains in the output after all the global attention layers.
            classifier_enc_out = self.classifier(enc_out.detach())
        else:
            classifier_enc_out, enc_var_idxs = None, None

        return (
            forecast_out,
            recon_out,
            (classifier_enc_out, enc_var_idxs),
            (enc_self_attns, dec_cross_attns),
        )

    def _attn_switch(
        self,
        attn_str: str,
        d_model: int,
        n_heads: int,
        d_qk: int,
        d_v: int,
        dropout_qkv: float,
        dropout_attn_matrix: float,
        attn_factor: int,
        performer_attn_kernel: str,
        performer_redraw_interval: int,
    ):

        if attn_str == "full":
            # standard full (n^2) attention
            Attn = AttentionLayer(
                attention=partial(FullAttention, attention_dropout=dropout_attn_matrix),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "prob":
            # Informer-style ProbSparse cross attention
            Attn = AttentionLayer(
                attention=partial(
                    ProbAttention,
                    factor=attn_factor,
                    attention_dropout=dropout_attn_matrix,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "performer":
            # Performer Linear Attention
            Attn = AttentionLayer(
                attention=partial(
                    PerformerAttention,
                    dim_heads=d_qk,
                    kernel=performer_attn_kernel,
                    feature_redraw_interval=performer_redraw_interval,
                ),
                d_model=d_model,
                d_queries_keys=d_qk,
                d_values=d_v,
                n_heads=n_heads,
                mix=False,
                dropout_qkv=dropout_qkv,
            )
        elif attn_str == "none":
            Attn = None
        else:
            raise ValueError(f"Unrecognized attention str code '{attn_str}'")
        return Attn
