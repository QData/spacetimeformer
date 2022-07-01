import io
import math
import os
import warnings

import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.distributions as pyd
import pandas as pd
import cv2
import random
import torch
import wandb
from einops import rearrange

from spacetimeformer.eval_stats import mape


def _assert_squeeze(x):
    assert len(x.shape) == 2
    return x.squeeze(-1)


def plot(x_c, y_c, x_t, y_t, idx, title, preds, pad_val=None, conf=None):
    y_c = y_c[..., idx]
    y_t = y_t[..., idx]
    preds = preds[..., idx]

    if pad_val is not None:
        y_c = y_c[y_c != pad_val]
        yt_mask = y_t != pad_val
        y_t = y_t[yt_mask]
        preds = preds[yt_mask]

    fig, ax = plt.subplots(figsize=(7, 4))
    xaxis_c = np.arange(len(y_c))
    xaxis_t = np.arange(len(y_c), len(y_c) + len(y_t))
    context = pd.DataFrame({"xaxis_c": xaxis_c, "y_c": y_c})
    target = pd.DataFrame({"xaxis_t": xaxis_t, "y_t": y_t, "pred": preds})
    sns.lineplot(data=context, x="xaxis_c", y="y_c", label="Context", linewidth=5.8)
    ax.scatter(
        x=target["xaxis_t"], y=target["y_t"], c="grey", label="True", linewidth=1.0
    )
    sns.lineplot(data=target, x="xaxis_t", y="pred", label="Forecast", linewidth=5.9)
    if conf is not None:
        conf = conf[..., idx]
        ax.fill_between(
            xaxis_t, (preds - conf), (preds + conf), color="orange", alpha=0.1
        )
    ax.legend(loc="upper left", prop={"size": 12})
    # ax.set_facecolor("#f0f0f0")
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=128)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class PredictionPlotterCallback(pl.Callback):
    def __init__(
        self,
        test_batch,
        var_idxs=None,
        var_names=None,
        pad_val=None,
        total_samples=4,
        log_to_wandb=True,
    ):
        self.test_data = test_batch
        self.total_samples = total_samples
        self.pad_val = pad_val
        self.log_to_wandb = log_to_wandb

        if var_idxs is None and var_names is None:
            d_yt = self.test_data[-1].shape[-1]
            var_idxs = list(range(d_yt))
            var_names = [f"y{i}" for i in var_idxs]

        self.var_idxs = var_idxs
        self.var_names = var_names
        self.imgs = None

    def on_validation_end(self, trainer, model):
        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]
        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        with torch.no_grad():
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
            preds_std = [None for _ in range(preds.shape[0])]

        imgs = []
        for i in range(preds.shape[0]):

            for var_idx, var_name in zip(self.var_idxs, self.var_names):
                img = plot(
                    x_c[i].cpu().numpy(),
                    y_c[i].cpu().numpy(),
                    x_t[i].cpu().numpy(),
                    y_t[i].cpu().numpy(),
                    idx=var_idx,
                    title=var_name,
                    preds=preds[i].cpu().numpy(),
                    pad_val=self.pad_val,
                    conf=preds_std[i] if model.loss == "nll" else None,
                )
                if img is not None:
                    if self.log_to_wandb:
                        img = wandb.Image(img)
                    imgs.append(img)

        if self.log_to_wandb:
            trainer.logger.experiment.log(
                {
                    "test/prediction_plots": imgs,
                    "global_step": trainer.global_step,
                }
            )
        else:
            self.imgs = imgs


class ImageCompletionCallback(pl.Callback):
    def __init__(self, test_batches, total_samples=12, mode="flat"):
        assert mode in ["flat", "left-right"]
        self.mode = mode
        self.test_data = test_batches
        self.total_samples = total_samples
        self._count = 0

    def complete_flat_img(self, trainer, model):
        with torch.no_grad():
            idxs = [i for i in range(self.total_samples)]
            x_c, y_c, x_t, y_t = [
                i[idxs].detach().to(model.device) for i in self.test_data
            ]
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
        completed_imgs = torch.cat((y_c, preds.clamp(0.0, 1.0)), dim=-2)
        shp = int(math.sqrt(completed_imgs.shape[-2]))  # (assumes square images)
        completed_imgs = rearrange(completed_imgs, "b (h w) c -> b c h w", h=shp)
        return completed_imgs

    def complete_left_right_img(self, trainer, model):
        with torch.no_grad():
            idxs = [i for i in range(self.total_samples)]
            x_c, y_c, x_t, y_t = [
                i[idxs].detach().to(model.device) for i in self.test_data
            ]
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)
        completed_imgs = torch.cat((y_c, preds.clamp(0.0, 1.0)), dim=1).transpose(1, 2)
        return completed_imgs

    def on_validation_end(self, trainer, model):

        if self.mode == "flat":
            completed_imgs = self.complete_flat_img(trainer, model)
        elif self.mode == "left-right":
            completed_imgs = self.complete_left_right_img(trainer, model)

        plots = []
        for i in range(completed_imgs.shape[0]):
            plot = wandb.Image(completed_imgs[i])
            plots.append(plot)

        trainer.logger.experiment.log(
            {
                "test/images": plots,
                "global_step": trainer.global_step,
            }
        )


class CopyTaskCallback(pl.Callback):
    def __init__(self, test_batches, total_samples=12):
        self.test_data = test_batches
        self.total_samples = total_samples

    def on_validation_end(self, trainer, model):
        with torch.no_grad():
            idxs = [
                random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)
            ]
            x_c, y_c, x_t, y_t = [
                i[idxs].detach().to(model.device) for i in self.test_data
            ]
            preds, *_ = model(x_c, y_c, x_t, y_t, **model.eval_step_forward_kwargs)

        boundary = torch.ones_like(x_t)
        image_tensor = torch.cat((preds, boundary, boundary, y_t), dim=-1)
        imgs = []
        for i in range(image_tensor.shape[0]):
            img = wandb.Image(image_tensor[i].T)
            imgs.append(img)

        trainer.logger.experiment.log(
            {
                "test/images": imgs,
                "global_step": trainer.global_step,
            }
        )


def show_image(data, title, tick_spacing=None, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(5, 5))

    plt.imshow(data, cmap=cmap)
    if tick_spacing:
        plt.xticks(np.arange(0, data.shape[0] + 1, tick_spacing))
        plt.yticks(np.arange(0, data.shape[0] + 1, tick_spacing))

    plt.title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=128)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close(fig)
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class AttentionMatrixCallback(pl.Callback):
    def __init__(self, test_batches, layer=0, total_samples=32):
        self.test_data = test_batches
        self.total_samples = total_samples
        self.layer = layer

    def _get_attns(self, model):
        idxs = [random.sample(range(self.test_data[0].shape[0]), k=self.total_samples)]
        x_c, y_c, x_t, y_t = [i[idxs].detach().to(model.device) for i in self.test_data]
        enc_attns, dec_attns = None, None
        # save memory by doing inference 1 example at a time
        for i in range(self.total_samples):
            x_ci = x_c[i].unsqueeze(0)
            y_ci = y_c[i].unsqueeze(0)
            x_ti = x_t[i].unsqueeze(0)
            y_ti = y_t[i].unsqueeze(0)
            with torch.no_grad():
                *_, (enc_self_attn, dec_cross_attn) = model(
                    x_ci, y_ci, x_ti, y_ti, output_attn=True
                )
            if enc_attns is None:
                enc_attns = [[a] for a in enc_self_attn]
            else:
                for cum_attn, attn in zip(enc_attns, enc_self_attn):
                    cum_attn.append(attn)
            if dec_attns is None:
                dec_attns = [[a] for a in dec_cross_attn]
            else:
                for cum_attn, attn in zip(dec_attns, dec_cross_attn):
                    cum_attn.append(attn)

        # re-concat over batch dim, avg over batch dim
        if enc_attns:
            enc_attns = [torch.cat(a, dim=0) for a in enc_attns][self.layer].mean(0)
        else:
            enc_attns = None
        if dec_attns:
            dec_attns = [torch.cat(a, dim=0) for a in dec_attns][self.layer].mean(0)
        else:
            dec_attns = None
        return enc_attns, dec_attns

    def _make_imgs(self, attns, img_title_prefix):
        heads = [i for i in range(attns.shape[0])] + ["avg", "sum"]
        imgs = []
        for head in heads:
            if head == "avg":
                a_head = attns.mean(0)
            elif head == "sum":
                a_head = attns.sum(0)
            else:
                a_head = attns[head]

            a_head /= torch.max(a_head, dim=-1)[0].unsqueeze(1)

            imgs.append(
                wandb.Image(
                    show_image(
                        a_head.cpu().numpy(),
                        f"{img_title_prefix} Head {str(head)}",
                        tick_spacing=a_head.shape[-2],
                        cmap="Blues",
                    )
                )
            )
        return imgs

    def _pos_sim_scores(self, embedding, seq_len, device):
        if embedding.position_emb == "t2v":
            inp = torch.arange(seq_len).float().to(device).view(1, -1, 1)
            encoder_embs = embedding.local_emb(inp)[0, :, 1:]
        elif embedding.position_emb == "abs":
            encoder_embs = embedding.local_emb(torch.arange(seq_len).to(device).long())
        cos_sim = torch.nn.CosineSimilarity(dim=0)
        scores = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(0, i + 1):
                sim = cos_sim(encoder_embs[i], encoder_embs[j])
                scores[i, j] = sim
                scores[j, i] = sim
        return scores

    def on_validation_end(self, trainer, model):
        self_attns, cross_attns = self._get_attns(model)

        if self_attns is not None:
            self_attn_imgs = self._make_imgs(
                self_attns, f"Self Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/self_attn": self_attn_imgs, "global_step": trainer.global_step}
            )
        if cross_attns is not None:
            cross_attn_imgs = self._make_imgs(
                cross_attns, f"Cross Attn, Layer {self.layer},"
            )
            trainer.logger.experiment.log(
                {"test/cross_attn": cross_attn_imgs, "global_step": trainer.global_step}
            )

        enc_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.enc_embedding,
            seq_len=self.test_data[1].shape[1],
            device=model.device,
        )
        dec_emb_sim = self._pos_sim_scores(
            model.spacetimeformer.dec_embedding,
            seq_len=self.test_data[3].shape[1],
            device=model.device,
        )
        emb_sim_imgs = [
            wandb.Image(
                show_image(
                    enc_emb_sim,
                    f"Encoder Position Emb. Similarity",
                    tick_spacing=enc_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
            wandb.Image(
                show_image(
                    dec_emb_sim,
                    f"Decoder Position Emb. Similarity",
                    tick_spacing=dec_emb_sim.shape[-1],
                    cmap="Greens",
                )
            ),
        ]
        trainer.logger.experiment.log(
            {"test/pos_embs": emb_sim_imgs, "global_step": trainer.global_step}
        )
