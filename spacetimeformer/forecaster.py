from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

import spacetimeformer as stf


class Forecaster(pl.LightningModule, ABC):
    def __init__(
        self,
        d_x: int,
        d_yc: int,
        d_yt: int,
        learning_rate: float = 1e-3,
        l2_coeff: float = 0,
        loss: str = "mse",
        linear_window: int = 0,
        linear_shared_weights: bool = False,
        use_revin: bool = False,
        use_seasonal_decomp: bool = False,
        verbose: int = True,
    ):
        super().__init__()

        qprint = lambda _msg_: print(_msg_) if verbose else None
        qprint("Forecaster")
        qprint(f"\tL2: {l2_coeff}")
        qprint(f"\tLinear Window: {linear_window}")
        qprint(f"\tLinear Shared Weights: {linear_shared_weights}")
        qprint(f"\tRevIN: {use_revin}")
        qprint(f"\tDecomposition: {use_seasonal_decomp}")

        self._inv_scaler = lambda x: x
        self.l2_coeff = l2_coeff
        self.learning_rate = learning_rate
        self.time_masked_idx = None
        self.null_value = None
        self.loss = loss

        if linear_window:
            self.linear_model = stf.linear_model.LinearModel(
                linear_window, shared_weights=linear_shared_weights, d_yt=d_yt
            )
        else:
            self.linear_model = lambda x, *args, **kwargs: 0.0

        self.use_revin = use_revin
        if use_revin:
            assert d_yc == d_yt, "TODO: figure out exo case for revin"
            self.revin = stf.revin.RevIN(num_features=d_yc)
        else:
            self.revin = lambda x, *args, **kwargs: x

        self.use_seasonal_decomp = use_seasonal_decomp
        if use_seasonal_decomp:
            self.seasonal_decomp = stf.revin.SeriesDecomposition(kernel_size=25)
        else:
            self.seasonal_decomp = lambda x: (x, x.clone())

        self.d_x = d_x
        self.d_yc = d_yc
        self.d_yt = d_yt
        self.save_hyperparameters()

    def set_null_value(self, val: float) -> None:
        self.null_value = val

    def set_inv_scaler(self, scaler) -> None:
        self._inv_scaler = scaler

    def set_scaler(self, scaler) -> None:
        self._scaler = scaler

    @property
    @abstractmethod
    def train_step_forward_kwargs(self):
        return {}

    @property
    @abstractmethod
    def eval_step_forward_kwargs(self):
        return {}

    def loss_fn(
        self, true: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:

        true = torch.nan_to_num(true)

        if self.loss == "mse":
            loss = (mask * (true - preds)).square().sum() / max(mask.sum(), 1)
        elif self.loss == "mae":
            loss = torch.abs(mask * (true - preds)).sum() / max(mask.sum(), 1)
        elif self.loss == "smape":
            num = 2.0 * abs(preds - true)
            den = abs(preds.detach()) + abs(true) + 1e-5
            loss = 100.0 * (mask * (num / den)).sum() / max(mask.sum(), 1)
        else:
            raise ValueError(f"Unrecognized Loss Function : {self.loss}")
        return loss

    def forecasting_loss(
        self, outputs: torch.Tensor, y_t: torch.Tensor, time_mask: int
    ) -> Tuple[torch.Tensor]:

        if self.null_value is not None:
            null_mask_mat = y_t != self.null_value
        else:
            null_mask_mat = torch.ones_like(y_t)

        # genuine NaN failsafe
        null_mask_mat *= ~torch.isnan(y_t)

        time_mask_mat = torch.ones_like(y_t)
        if time_mask is not None:
            time_mask_mat[:, time_mask:] = False

        full_mask = time_mask_mat * null_mask_mat
        forecasting_loss = self.loss_fn(y_t, outputs, full_mask)
        return forecasting_loss, full_mask

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor],
        time_mask: int = None,
        forward_kwargs: dict = {},
    ) -> Tuple[torch.Tensor]:
        x_c, y_c, x_t, y_t = batch

        outputs, *_ = self(x_c, y_c, x_t, y_t, **forward_kwargs)

        loss, mask = self.forecasting_loss(
            outputs=outputs, y_t=y_t, time_mask=time_mask
        )
        return loss, outputs, mask

    def predict(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        sample_preds: bool = False,
    ) -> torch.Tensor:
        og_device = y_c.device
        # move to model device
        x_c = x_c.to(self.device).float()
        x_t = x_t.to(self.device).float()
        # move y_c to cpu if it isn't already there, scale, and then move back to the model device
        y_c = torch.from_numpy(self._scaler(y_c.cpu().numpy())).to(self.device).float()
        # create dummy y_t of zeros
        y_t = (
            torch.zeros((x_t.shape[0], x_t.shape[1], self.d_yt)).to(self.device).float()
        )

        with torch.no_grad():
            # gradient-free prediction
            normalized_preds, *_ = self.forward(
                x_c, y_c, x_t, y_t, **self.eval_step_forward_kwargs
            )

        # preds --> cpu --> inverse scale to original units --> original device of y_c
        preds = (
            torch.from_numpy(self._inv_scaler(normalized_preds.cpu().numpy()))
            .to(og_device)
            .float()
        )
        return preds

    @abstractmethod
    def forward_model_pass(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        return NotImplemented

    def nan_to_num(self, *inps):
        return (torch.nan_to_num(i) for i in inps)

    def forward(
        self,
        x_c: torch.Tensor,
        y_c: torch.Tensor,
        x_t: torch.Tensor,
        y_t: torch.Tensor,
        **forward_kwargs,
    ) -> Tuple[torch.Tensor]:
        x_c, y_c, x_t, y_t = self.nan_to_num(x_c, y_c, x_t, y_t)
        _, pred_len, d_yt = y_t.shape

        y_c = self.revin(y_c, mode="norm")  # does nothing if use_revin = False

        seasonal_yc, trend_yc = self.seasonal_decomp(
            y_c
        )  # both are the original if use_seasonal_decomp = False

        preds, *extra = self.forward_model_pass(
            x_c, seasonal_yc, x_t, y_t, **forward_kwargs
        )
        baseline = self.linear_model(trend_yc, pred_len=pred_len, d_yt=d_yt)

        output = self.revin(preds + baseline, mode="denorm")

        if extra:
            return (output,) + tuple(extra)
        return (output,)

    def _compute_stats(
        self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor
    ):
        pred = pred * mask
        true = torch.nan_to_num(true) * mask

        adj = mask.mean().cpu().numpy() + 1e-5
        pred = pred.detach().cpu().numpy()
        true = true.detach().cpu().numpy()
        scaled_pred = self._inv_scaler(pred)
        scaled_true = self._inv_scaler(true)
        stats = {
            "mape": stf.eval_stats.mape(scaled_true, scaled_pred) / adj,
            "mae": stf.eval_stats.mae(scaled_true, scaled_pred) / adj,
            "mse": stf.eval_stats.mse(scaled_true, scaled_pred) / adj,
            "smape": stf.eval_stats.smape(scaled_true, scaled_pred) / adj,
            "norm_mae": stf.eval_stats.mae(true, pred) / adj,
            "norm_mse": stf.eval_stats.mse(true, pred) / adj,
        }
        return stats

    def step(self, batch: Tuple[torch.Tensor], train: bool = False):
        kwargs = (
            self.train_step_forward_kwargs if train else self.eval_step_forward_kwargs
        )
        time_mask = self.time_masked_idx if train else None

        loss, output, mask = self.compute_loss(
            batch=batch,
            time_mask=time_mask,
            forward_kwargs=kwargs,
        )
        *_, y_t = batch
        stats = self._compute_stats(output, y_t, mask)
        stats["loss"] = loss
        return stats

    def training_step(self, batch, batch_idx):
        return self.step(batch, train=True)

    def validation_step(self, batch, batch_idx):
        stats = self.step(batch, train=False)
        self.current_val_stats = stats
        return stats

    def test_step(self, batch, batch_idx):
        return self.step(batch, train=False)

    def _log_stats(self, section, outs):
        for key in outs.keys():
            stat = outs[key]
            if isinstance(stat, np.ndarray) or isinstance(stat, torch.Tensor):
                stat = stat.mean()
            self.log(f"{section}/{key}", stat, sync_dist=True)

    def training_step_end(self, outs):
        self._log_stats("train", outs)
        return {"loss": outs["loss"].mean()}

    def validation_step_end(self, outs):
        self._log_stats("val", outs)
        return outs

    def test_step_end(self, outs):
        self._log_stats("test", outs)
        return {"loss": outs["loss"].mean()}

    def predict_step(self, batch, batch_idx):
        return self(*batch, **self.eval_step_forward_kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_coeff
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=3,
            factor=0.2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
            },
        }

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--gpus", type=int, nargs="+")
        parser.add_argument("--l2_coeff", type=float, default=1e-6)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--grad_clip_norm", type=float, default=0)
        parser.add_argument("--linear_window", type=int, default=0)
        parser.add_argument("--use_revin", action="store_true")
        parser.add_argument(
            "--loss", type=str, default="mse", choices=["mse", "mae", "smape"]
        )
        parser.add_argument("--linear_shared_weights", action="store_true")
        parser.add_argument("--use_seasonal_decomp", action="store_true")
