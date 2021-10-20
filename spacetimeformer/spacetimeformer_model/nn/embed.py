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

    def __call__(self, y, x, is_encoder=True):
        if self.method == "spatio-temporal":
            val_time_emb, space_emb, var_idxs = self.parallel_spatio_temporal_embed(
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

    def benchmark_spatio_temporal_embed(self, y, x, is_encoder=True):
        # use pre-made fake embedding matrix to simulate the fastest
        # possible embedding speed and measure whether this implementation
        # is a bottleneck. (it isn't)
        if self._benchmark_embed_enc is None and is_encoder:
            bs, length, d_y = y.shape
            self._benchmark_embed_enc = torch.ones(bs, d_y * length, self.d_model).to(
                y.device
            )

        elif self._benchmark_embed_dec is None and not is_encoder:
            bs, length, d_y = y.shape
            self._benchmark_embed_dec = torch.ones(bs, d_y * length, self.d_model).to(
                y.device
            )

        node_emb = (
            self._benchmark_embed_enc if is_encoder else self._benchmark_embed_dec
        )

        if is_encoder:
            for conv in self.downsize_convs:
                node_emb = conv(node_emb)
        return node_emb, torch.zeros_like(node_emb)

    SPACE = True
    TIME = True
    VAL = True
    GIVEN = True

    def parallel_spatio_temporal_embed(self, y, x, is_encoder=True):
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
        given = torch.ones((bs, length, d_y)).long().to(x.device)
        if not is_encoder and self.GIVEN:
            given[:, self.start_token_len :, :] = 0
        given = torch.cat(given.chunk(d_y, dim=-1), dim=1).squeeze(-1)
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

    def iter_spatio_temporal_embed(self, y, x, is_encoder=True):
        assert len(self.downsize_convs) == 0

        bs, length, d_y = y.shape
        # split y into d_y sequences
        ys = y.chunk(d_y, axis=-1)

        # time embedding
        if not self.TIME:
            x = torch.zeros_like(x)
        time_emb = self.x_emb(x)

        val_time_embs = []
        var_embs = []
        for i, y in enumerate(ys):
            emb_inp = torch.cat((y, time_emb), dim=-1)
            val_time_emb = self.y_emb(emb_inp)

            # spatial (variable) embedding for variable i
            var_idx = (
                torch.Tensor([i for _ in range(length)])
                .long()
                .to(y.device)
                .repeat(bs, 1)
            )
            if not self.SPACE:
                var_idx = torch.zeros_like(var_idx)
            var_emb = self.var_emb(var_idx)

            val_time_embs.append(val_time_emb)
            var_embs.append(self.var_emb(var_idx))

        val_time_embs = torch.cat(val_time_embs, dim=1)
        var_embs = torch.cat(var_embs, dim=1)
        return val_time_embs, var_embs
