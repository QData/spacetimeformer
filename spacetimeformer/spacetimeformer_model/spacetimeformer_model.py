from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

import spacetimeformer as stf


class Spacetimeformer_Forecaster(stf.Forecaster):
    def __init__(
        self,
        d_yc: int = 1,
        d_yt: int = 1,
        d_x: int = 4,
        max_seq_len: int = None,
        start_token_len: int = 64,
        attn_factor: int = 5,
        d_model: int = 200,
        d_queries_keys=50,
        d_values=50,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 2,
        d_ff: int = 800,
        dropout_emb: float = 0.1,
        dropout_qkv: float = 0.0,
        dropout_ff: float = 0.2,
        dropout_attn_out: float = 0.0,
        dropout_attn_matrix: float = 0.0,
        pos_emb_type: str = "abs",
        global_self_attn: str = "performer",
        local_self_attn: str = "performer",
        global_cross_attn: str = "performer",
        local_cross_attn: str = "performer",
        performer_kernel: str = "relu",
        embed_method: str = "spatio-temporal",
        performer_relu: bool = True,
        performer_redraw_interval: int = 1000,
        attn_time_windows: int = 1,
        use_shifted_time_windows: bool = True,
        activation: str = "gelu",
        norm: str = "batch",
        use_final_norm: bool = True,
        init_lr: float = 1e-10,
        base_lr: float = 5e-4,
        warmup_steps: float = 1000,
        decay_factor: float = 0.8,
        initial_downsample_convs: int = 0,
        intermediate_downsample_convs: int = 0,
        l2_coeff: float = 1e-3,
        loss: str = "mse",
        class_loss_imp: float = 1e-3,
        recon_loss_imp: float = 0,
        time_emb_dim: int = 6,
        null_value: float = None,
        pad_value: float = None,
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
        recon_mask_skip_all: float = 1.0,
        recon_mask_max_seq_len: int = 5,
        recon_mask_drop_seq: float = 0.1,
        recon_mask_drop_standard: float = 0.2,
        recon_mask_drop_full: float = 0.05,
        verbose=True,
    ):
        super().__init__(
            d_x=d_x,
            d_yc=d_yc,
            d_yt=d_yt,
            l2_coeff=l2_coeff,
            loss=loss,
            linear_window=linear_window,
            use_revin=use_revin,
            use_seasonal_decomp=use_seasonal_decomp,
            linear_shared_weights=linear_shared_weights,
        )
        self.spacetimeformer = stf.spacetimeformer_model.nn.Spacetimeformer(
            d_yc=d_yc,
            d_yt=d_yt,
            d_x=d_x,
            start_token_len=start_token_len,
            attn_factor=attn_factor,
            d_model=d_model,
            d_queries_keys=d_queries_keys,
            d_values=d_values,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            initial_downsample_convs=initial_downsample_convs,
            intermediate_downsample_convs=intermediate_downsample_convs,
            dropout_emb=dropout_emb,
            dropout_attn_out=dropout_attn_out,
            dropout_attn_matrix=dropout_attn_matrix,
            dropout_qkv=dropout_qkv,
            dropout_ff=dropout_ff,
            pos_emb_type=pos_emb_type,
            global_self_attn=global_self_attn,
            local_self_attn=local_self_attn,
            global_cross_attn=global_cross_attn,
            local_cross_attn=local_cross_attn,
            activation=activation,
            device=self.device,
            norm=norm,
            use_final_norm=use_final_norm,
            embed_method=embed_method,
            performer_attn_kernel=performer_kernel,
            performer_redraw_interval=performer_redraw_interval,
            attn_time_windows=attn_time_windows,
            use_shifted_time_windows=use_shifted_time_windows,
            time_emb_dim=time_emb_dim,
            verbose=True,
            null_value=null_value,
            pad_value=pad_value,
            max_seq_len=max_seq_len,
            use_val=use_val,
            use_time=use_time,
            use_space=use_space,
            use_given=use_given,
            recon_mask_skip_all=recon_mask_skip_all,
            recon_mask_max_seq_len=recon_mask_max_seq_len,
            recon_mask_drop_seq=recon_mask_drop_seq,
            recon_mask_drop_standard=recon_mask_drop_standard,
            recon_mask_drop_full=recon_mask_drop_full,
        )
        self.start_token_len = start_token_len
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_factor = decay_factor
        self.embed_method = embed_method
        self.class_loss_imp = class_loss_imp
        self.recon_loss_imp = recon_loss_imp
        self.set_null_value(null_value)
        self.pad_value = pad_value
        self.save_hyperparameters()

        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint(f" *** Spacetimeformer (v1.5) Summary: *** ")
        qprint(f"\t\tModel Dim: {d_model}")
        qprint(f"\t\tFF Dim: {d_ff}")
        qprint(f"\t\tEnc Layers: {e_layers}")
        qprint(f"\t\tDec Layers: {d_layers}")
        qprint(f"\t\tEmbed Dropout: {dropout_emb}")
        qprint(f"\t\tFF Dropout: {dropout_ff}")
        qprint(f"\t\tAttn Out Dropout: {dropout_attn_out}")
        qprint(f"\t\tAttn Matrix Dropout: {dropout_attn_matrix}")
        qprint(f"\t\tQKV Dropout: {dropout_qkv}")
        qprint(f"\t\tL2 Coeff: {l2_coeff}")
        qprint(f"\t\tWarmup Steps: {warmup_steps}")
        qprint(f"\t\tNormalization Scheme: {norm}")
        qprint(f"\t\tAttention Time Windows: {attn_time_windows}")
        qprint(f"\t\tShifted Time Windows: {use_shifted_time_windows}")
        qprint(f"\t\tPosition Emb Type: {pos_emb_type}")
        qprint(f"\t\tRecon Loss Imp: {recon_loss_imp}")
        qprint(f" ***                                  *** ")

    @property
    def train_step_forward_kwargs(self):
        return {"output_attn": False}

    @property
    def eval_step_forward_kwargs(self):
        return {"output_attn": False}

    def step(self, batch: Tuple[torch.Tensor], train: bool):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )

        time_mask = self.time_masked_idx if train else None

        # compute all loss values
        loss_dict = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )

        forecast_out = loss_dict["forecast_out"]
        forecast_mask = loss_dict["forecast_mask"]
        *_, y_t = batch

        # compute prediction accuracy stats for logging
        stats = self._compute_stats(forecast_out, y_t, forecast_mask)

        stats["forecast_loss"] = loss_dict["forecast_loss"]
        stats["class_loss"] = loss_dict["class_loss"]
        stats["recon_loss"] = loss_dict["recon_loss"]

        # loss is a combination of forecasting, reconstruction and classification goals
        stats["loss"] = (
            loss_dict["forecast_loss"]
            + self.class_loss_imp * loss_dict["class_loss"]
            + self.recon_loss_imp * loss_dict["recon_loss"]
        )
        stats["acc"] = loss_dict["acc"]
        return stats

    def classification_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor]:

        labels = labels.view(-1).to(logits.device)
        d_y = labels.max() + 1

        logits = logits.view(-1, d_y)

        class_loss = F.cross_entropy(logits, labels)
        acc = torchmetrics.functional.accuracy(
            torch.softmax(logits, dim=1),
            labels,
        )
        return class_loss, acc

    def compute_loss(self, batch, time_mask=None, forward_kwargs={}):
        x_c, y_c, x_t, y_t = batch

        forecast_out, recon_out, (logits, labels) = self(
            x_c, y_c, x_t, y_t, **forward_kwargs
        )

        # forecast (target seq prediction) loss
        forecast_loss, forecast_mask = self.forecasting_loss(
            outputs=forecast_out, y_t=y_t, time_mask=time_mask
        )

        if self.recon_loss_imp > 0:
            # reconstruction (masked? context seq prediction) loss
            recon_loss, recon_mask = self.forecasting_loss(
                outputs=recon_out, y_t=y_c, time_mask=None
            )
        else:
            recon_loss, recon_mask = -1.0, 0.0

        if self.embed_method == "spatio-temporal" and self.class_loss_imp > 0:
            # space emb classification loss (detached)
            class_loss, acc = self.classification_loss(logits=logits, labels=labels)
        else:
            class_loss, acc = 0.0, -1.0

        return {
            "forecast_loss": forecast_loss,
            "class_loss": class_loss,
            "acc": acc,
            "forecast_out": forecast_out,
            "forecast_mask": forecast_mask,
            "recon_out": recon_out,
            "recon_loss": recon_loss,
            "recon_mask": recon_mask,
        }

    def nan_to_num(self, *inps):
        # override to let embedding handle NaNs
        return inps

    def forward_model_pass(self, x_c, y_c, x_t, y_t, output_attn=False):
        # set data to [batch, length, dim] format
        if len(y_c.shape) == 2:
            y_c = y_c.unsqueeze(-1)
        if len(y_t.shape) == 2:
            y_t = y_t.unsqueeze(-1)

        enc_x = x_c
        enc_y = y_c
        dec_x = x_t

        # zero out target sequence
        dec_y = torch.zeros_like(y_t).to(self.device)
        if self.start_token_len > 0:
            # add "start token" from informer. not really needed anymore...
            dec_y = torch.cat((y_c[:, -self.start_token_len :, :], dec_y), dim=1).to(
                self.device
            )
            dec_x = torch.cat((x_c[:, -self.start_token_len :, :], dec_x), dim=1)

        forecast_output, recon_output, (logits, labels), attn = self.spacetimeformer(
            enc_x=enc_x,
            enc_y=enc_y,
            dec_x=dec_x,
            dec_y=dec_y,
            output_attention=output_attn,
        )

        if output_attn:
            return forecast_output, recon_output, (logits, labels), attn
        return forecast_output, recon_output, (logits, labels)

    def validation_epoch_end(self, outs):
        total = 0
        count = 0
        for dict_ in outs:
            if "forecast_loss" in dict_:
                total += dict_["forecast_loss"].mean()
                count += 1
        avg_val_loss = total / count
        # manually tell scheduler it's the end of an epoch to activate
        # ReduceOnPlateau functionality from a step-based scheduler
        self.scheduler.step(avg_val_loss, is_end_epoch=True)

    def training_step_end(self, outs):
        self._log_stats("train", outs)
        self.scheduler.step()
        return {"loss": outs["loss"].mean()}

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.base_lr,
            weight_decay=self.l2_coeff,
        )
        self.scheduler = stf.lr_scheduler.WarmupReduceLROnPlateau(
            self.optimizer,
            init_lr=self.init_lr,
            peak_lr=self.base_lr,
            warmup_steps=self.warmup_steps,
            patience=3,
            factor=self.decay_factor,
        )
        return [self.optimizer], [self.scheduler]

    @classmethod
    def add_cli(self, parser):
        super().add_cli(parser)
        parser.add_argument(
            "--start_token_len",
            type=int,
            default=0,
            help="Length of decoder start token. Adds this many of the final context points to the start of the target sequence.",
        )
        parser.add_argument(
            "--d_model", type=int, default=200, help="Transformer embedding dimension."
        )
        parser.add_argument(
            "--d_qk",
            type=int,
            default=200,
        )
        parser.add_argument(
            "--d_v",
            type=int,
            default=200,
        )
        parser.add_argument(
            "--n_heads", type=int, default=4, help="Number of self-attention heads."
        )
        parser.add_argument(
            "--enc_layers", type=int, default=3, help="Transformer encoder layers."
        )
        parser.add_argument(
            "--dec_layers", type=int, default=3, help="Transformer decoder layers."
        )
        parser.add_argument(
            "--d_ff",
            type=int,
            default=800,
            help="Dimension of Transformer up-scaling MLP layer. (often 4 * d_model)",
        )
        parser.add_argument(
            "--attn_factor",
            type=int,
            default=5,
            help="ProbSparse attention factor. N/A to other attn mechanisms.",
        )
        parser.add_argument(
            "--dropout_emb",
            type=float,
            default=0.2,
            help="Embedding dropout rate. Drop out elements of the embedding vectors during training.",
        )
        parser.add_argument(
            "--dropout_attn_matrix",
            type=float,
            default=0.0,
            help="Attention dropout rate. Dropout elements of the attention matrix. Only applicable to attn mechanisms that explicitly compute the attn matrix (e.g. Full).",
        )
        parser.add_argument(
            "--dropout_qkv",
            type=float,
            default=0.0,
            help="Query, Key and Value dropout rate. Dropout elements of these attention vectors during training.",
        )
        parser.add_argument(
            "--dropout_ff",
            type=float,
            default=0.3,
            help="Standard dropout applied to activations of FF networks in the Transformer.",
        )
        parser.add_argument(
            "--dropout_attn_out",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--global_self_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "prob",
                "performer",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--global_cross_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "performer",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--local_self_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "prob",
                "performer",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--local_cross_attn",
            type=str,
            default="performer",
            choices=[
                "full",
                "performer",
                "none",
            ],
            help="Attention mechanism type.",
        )
        parser.add_argument(
            "--activation",
            type=str,
            default="gelu",
            choices=["relu", "gelu"],
            help="Activation function for Transformer encoder and decoder layers.",
        )
        parser.add_argument(
            "--norm",
            type=str,
            choices=["layer", "batch", "scale", "power", "none"],
            default="batch",
        )
        parser.add_argument(
            "--init_lr", type=float, default=1e-10, help="Initial learning rate."
        )
        parser.add_argument(
            "--base_lr",
            type=float,
            default=5e-4,
            help="Base/peak LR. The LR is annealed to this value from --init_lr over --warmup_steps training steps.",
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=0, help="LR anneal steps."
        )
        parser.add_argument(
            "--decay_factor",
            type=float,
            default=0.25,
            help="Factor to reduce LR on plateau (after warmup period is over).",
        )
        parser.add_argument(
            "--initial_downsample_convs",
            type=int,
            default=0,
            help="Add downsampling Conv1Ds to the encoder embedding layer to reduce context sequence length.",
        )
        parser.add_argument(
            "--class_loss_imp",
            type=float,
            default=0.1,
            help="Coefficient for node classification loss function. Set to 0 to disable this feature. Does not significantly impact forecasting results due to detached gradient.",
        )
        parser.add_argument(
            "--recon_loss_imp",
            type=float,
            default=0.0,
        )
        parser.add_argument(
            "--intermediate_downsample_convs",
            type=int,
            default=0,
            help="Add downsampling Conv1Ds between encoder layers.",
        )
        parser.add_argument(
            "--time_emb_dim",
            type=int,
            default=6,
            help="Time embedding dimension. Embed *each dimension of x* with this many learned periodic values.",
        )
        parser.add_argument(
            "--performer_kernel",
            type=str,
            default="relu",
            choices=["softmax", "relu"],
            help="Performer attention kernel. See Performer paper for details.",
        )
        parser.add_argument(
            "--performer_redraw_interval",
            type=int,
            default=100,
            help="Training steps between resampling orthogonal random features for FAVOR+ attention",
        )
        parser.add_argument(
            "--embed_method",
            type=str,
            choices=["spatio-temporal", "temporal"],
            default="spatio-temporal",
            help="Embedding method. spatio-temporal enables long-sequence spatio-temporal transformer mode while temporal recovers default architecture.",
        )
        parser.add_argument(
            "--attn_time_windows",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--use_shifted_time_windows",
            action="store_true",
        )
        parser.add_argument(
            "--pos_emb_type",
            type=str,
            default="abs",
            choices=["abs", "t2v"],
        )
        parser.add_argument(
            "--no_final_norm",
            action="store_true",
        )
        parser.add_argument(
            "--no_val",
            action="store_true",
        )
        parser.add_argument(
            "--no_time",
            action="store_true",
        )
        parser.add_argument(
            "--no_space",
            action="store_true",
            help="Remove `space (variable)` embeddings. Has no effect when using standard temporal attention. Is very likely to hurt performance w/ spatiotemporal attention",
        )
        parser.add_argument(
            "--no_given",
            action="store_true",
            help="Remove `given` embeddings. Only relevant when using a start token, padding, reconstruction dropout, or null values",
        )
        parser.add_argument(
            "--recon_mask_skip_all",
            type=float,
            default=1.0,
            help="Skip all kinds of reconstruction (context sequence) dropout at this probability. 1.0 turns off all data dropout",
        )
        parser.add_argument(
            "--recon_mask_max_seq_len",
            type=int,
            default=5,
            help="Max sequence length that will be dropped from the context sequence during reconstruction dropout",
        )
        parser.add_argument(
            "--recon_mask_drop_seq",
            type=float,
            default=0.2,
            help="Pct of the context sequence that will be masked by subsequence dropout during reconstruction",
        )
        parser.add_argument(
            "--recon_mask_drop_standard",
            type=float,
            default=0.1,
            help="Pct of the total context sequence masked by standard droput (each element dropped ind.) during reconstruction",
        )
        parser.add_argument(
            "--recon_mask_drop_full",
            type=float,
            default=0.05,
            help="Pct of timesteps in the context sequence that will be completely masked during reconstruction (vs. `recon_mask_drop_standard` which can mask some but not all vars at a given timestep)",
        )
