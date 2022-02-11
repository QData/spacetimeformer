import torch
import torch.nn as nn
import torch.nn.functional as F


import spacetimeformer as stf

from .encoder import VariableDownsample


class SpacetimeformerEmbedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x,
        d_model=256,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=1,
        start_token_len=0,
        null_value=None,
    ):
        super().__init__()

        assert method in ["spatio-temporal", "temporal"]
        self.method = method

        # account for added local position indicator "relative time"
        d_x += 1

        self.x_emb = stf.Time2Vec(d_x, embed_dim=time_emb_dim * d_x)

        if self.method == "temporal":
            y_emb_inp_dim = d_y + (time_emb_dim * d_x)
        else:
            y_emb_inp_dim = 1 + (time_emb_dim * d_x)

        self.y_emb = nn.Linear(y_emb_inp_dim, d_model)

        if self.method == "spatio-temporal":
            self.var_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [VariableDownsample(d_y, d_model) for _ in range(downsample_convs)]
        )

        self._benchmark_embed_enc = None
        self._benchmark_embed_dec = None
        self.d_model = d_model
        self.null_value = null_value

    def __call__(self, y, x, is_encoder=True):
        if self.method == "spatio-temporal":
            val_time_emb, space_emb, var_idxs = self.spatio_temporal_embed(
                y, x, is_encoder
            )
        else:
            val_time_emb, space_emb = self.temporal_embed(y, x, is_encoder)
            var_idxs = None

        return val_time_emb, space_emb, var_idxs

    def temporal_embed(self, y, x, is_encoder=True):
        bs, length, d_y = y.shape

        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        if not self.TIME:
            x = torch.zeros_like(x)
        x = torch.cat((x, local_pos), dim=-1)
        t2v_emb = self.x_emb(x)

        emb_inp = torch.cat((y, t2v_emb), dim=-1)
        emb = self.y_emb(emb_inp)

        # "given" embedding
        given = torch.ones((bs, length)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given)
        emb += given_emb

        if is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        return emb, torch.zeros_like(emb)

    SPACE = True
    TIME = True
    VAL = True
    GIVEN = True

    def spatio_temporal_embed(self, y, x, is_encoder=True):
        bs, length, d_y = y.shape

        # val  + time embedding
        y = torch.cat(y.chunk(d_y, dim=-1), dim=1)
        local_pos = (
            torch.arange(length).view(1, -1, 1).repeat(bs, 1, 1).to(x.device) / length
        )
        x = torch.cat((x, local_pos), dim=-1)
        if not self.TIME:
            x = torch.zeros_like(x)
        if not self.VAL:
            y = torch.zeros_like(y)
        t2v_emb = self.x_emb(x).repeat(1, d_y, 1)
        val_time_inp = torch.cat((y, t2v_emb), dim=-1)
        val_time_emb = self.y_emb(val_time_inp)

        # "given" embedding
        if self.GIVEN:
            given = torch.ones((bs, length, d_y)).long().to(x.device)  # start as T
            if not is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0
            given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1)
            if self.null_value is not None:
                # mask null values
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask
            given_emb = self.given_emb(given)
            val_time_emb += given_emb

        if is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # var embedding
        var_idx = torch.Tensor([[i for j in range(length)] for i in range(d_y)])
        var_idx = var_idx.long().to(x.device).view(-1).unsqueeze(0).repeat(bs, 1)
        var_idx_true = var_idx.clone()
        if not self.SPACE:
            var_idx = torch.zeros_like(var_idx)
        var_emb = self.var_emb(var_idx)

        return val_time_emb, var_emb, var_idx_true
