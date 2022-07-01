import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


import spacetimeformer as stf

from .extra_layers import ConvBlock, Flatten


class Embedding(nn.Module):
    def __init__(
        self,
        d_y,
        d_x,
        d_model,
        time_emb_dim=6,
        method="spatio-temporal",
        downsample_convs=0,
        start_token_len=0,
        null_value=None,
        pad_value=None,
        is_encoder: bool = True,
        position_emb="abs",
        data_dropout=None,
        max_seq_len=None,
        use_val: bool = True,
        use_time: bool = True,
        use_space: bool = True,
        use_given: bool = True,
    ):
        super().__init__()

        assert method in ["spatio-temporal", "temporal"]
        if data_dropout is None:
            self.data_drop = lambda y: y
        else:
            self.data_drop = data_dropout

        self.method = method

        time_dim = time_emb_dim * d_x
        self.time_emb = stf.Time2Vec(d_x, embed_dim=time_dim)

        assert position_emb in ["t2v", "abs"]
        self.max_seq_len = max_seq_len
        self.position_emb = position_emb
        if self.position_emb == "t2v":
            # standard periodic pos emb but w/ learnable coeffs
            self.local_emb = stf.Time2Vec(1, embed_dim=d_model + 1)
        elif self.position_emb == "abs":
            # lookup-based learnable pos emb
            assert max_seq_len is not None
            self.local_emb = nn.Embedding(
                num_embeddings=max_seq_len, embedding_dim=d_model
            )

        y_emb_inp_dim = d_y if self.method == "temporal" else 1
        self.val_time_emb = nn.Linear(y_emb_inp_dim + time_dim, d_model)

        if self.method == "spatio-temporal":
            self.space_emb = nn.Embedding(num_embeddings=d_y, embedding_dim=d_model)
            split_length_into = d_y
        else:
            split_length_into = 1

        self.start_token_len = start_token_len
        self.given_emb = nn.Embedding(num_embeddings=2, embedding_dim=d_model)

        self.downsize_convs = nn.ModuleList(
            [ConvBlock(split_length_into, d_model) for _ in range(downsample_convs)]
        )

        self.d_model = d_model
        self.null_value = null_value
        self.pad_value = pad_value
        self.is_encoder = is_encoder

        # turning off parts of the embedding is only really here for ablation studies
        self.use_val = use_val
        self.use_time = use_time
        self.use_given = use_given
        self.use_space = use_space

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if self.method == "spatio-temporal":
            emb = self.spatio_temporal_embed
        else:
            emb = self.temporal_embed
        return emb(y=y, x=x)

    def make_mask(self, y):
        # we make padding-based masks here due to outdated
        # feature where the embedding randomly drops tokens by setting
        # them to the pad value as a form of regularization
        if self.pad_value is None:
            return None
        return (y == self.pad_value).any(-1, keepdim=True)

    def temporal_embed(self, y: torch.Tensor, x: torch.Tensor):
        bs, length, d_y = y.shape

        # protect against true NaNs. without
        # `spatio_temporal_embed`'s multivariate "Given"
        # concept there isn't much else we can do here.
        # NaNs should probably be set to a magic number value
        # in the dataset and passed to the null_value arg.
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)

        if self.is_encoder:
            # optionally mask the context sequence for reconstruction
            y = self.data_drop(y)
        mask = self.make_mask(y)

        # position embedding ("local_emb")
        local_pos = torch.arange(length).to(x.device)
        if self.position_emb == "t2v":
            # first idx of Time2Vec output is unbounded so we drop it to
            # reuse code as a learnable pos embb
            local_emb = self.local_emb(
                local_pos.view(1, -1, 1).repeat(bs, 1, 1).float()
            )[:, :, 1:]
        elif self.position_emb == "abs":
            assert length <= self.max_seq_len
            local_emb = self.local_emb(local_pos.long().view(1, -1).repeat(bs, 1))

        # time embedding (Time2Vec)
        if not self.use_time:
            x = torch.zeros_like(x)
        time_emb = self.time_emb(x)
        if not self.use_val:
            y = torch.zeros_like(y)
        # concat time emb to value --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, y), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding. not important for temporal emb
        # when not using a start token
        given = torch.ones((bs, length)).long().to(x.device)
        if not self.is_encoder and self.use_given:
            given[:, self.start_token_len :] = 0
        given_emb = self.given_emb(given)

        emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            # shorten the sequence
            for i, conv in enumerate(self.downsize_convs):
                emb = conv(emb)

        # space emb not used for temporal method
        space_emb = torch.zeros_like(emb)
        var_idxs = None
        return emb, space_emb, var_idxs, mask

    def spatio_temporal_embed(self, y: torch.Tensor, x: torch.Tensor):
        # full spatiotemopral emb method. lots of shape rearrange code
        # here to create artifically long (length x dim) spatiotemporal sequence
        batch, length, dy = y.shape

        # position emb ("local_emb")
        local_pos = repeat(
            torch.arange(length).to(x.device), f"length -> {batch} ({dy} length)"
        )
        if self.position_emb == "t2v":
            # periodic pos emb
            local_emb = self.local_emb(local_pos.float().unsqueeze(-1).float())[
                :, :, 1:
            ]
        elif self.position_emb == "abs":
            # lookup pos emb
            local_emb = self.local_emb(local_pos.long())

        # time emb
        if not self.use_time:
            x = torch.zeros_like(x)
        x = torch.nan_to_num(x)
        x = repeat(x, f"batch len x_dim -> batch ({dy} len) x_dim")
        time_emb = self.time_emb(x)

        # protect against NaNs in y, but keep track for Given emb
        true_null = torch.isnan(y)
        y = torch.nan_to_num(y)
        if not self.use_val:
            y = torch.zeros_like(y)

        # keep track of pre-dropout y for given emb
        y_original = y.clone()
        y_original = Flatten(y_original)
        y = self.data_drop(y)
        y = Flatten(y)
        mask = self.make_mask(y)

        # concat time_emb, y --> FF --> val_time_emb
        val_time_inp = torch.cat((time_emb, y), dim=-1)
        val_time_emb = self.val_time_emb(val_time_inp)

        # "given" embedding
        if self.use_given:
            given = torch.ones((batch, length, dy)).long().to(x.device)  # start as True
            if not self.is_encoder:
                # mask missing values that need prediction...
                given[:, self.start_token_len :, :] = 0  # (False)

            # if y was NaN, set Given = False
            given *= ~true_null

            # flatten now to make the rest easier to figure out
            given = rearrange(given, "batch len dy -> batch (dy len)")

            # use given embeddings to identify data that was dropped out
            given *= (y == y_original).squeeze(-1)

            if self.null_value is not None:
                # mask null values that were set to a magic number in the dataset itself
                null_mask = (y != self.null_value).squeeze(-1)
                given *= null_mask

            given_emb = self.given_emb(given)
        else:
            given_emb = 0.0

        val_time_emb = local_emb + val_time_emb + given_emb

        if self.is_encoder:
            for conv in self.downsize_convs:
                val_time_emb = conv(val_time_emb)
                length //= 2

        # space embedding
        var_idx = repeat(
            torch.arange(dy).long().to(x.device), f"dy -> {batch} (dy {length})"
        )
        var_idx_true = var_idx.clone()
        if not self.use_space:
            var_idx = torch.zeros_like(var_idx)
        space_emb = self.space_emb(var_idx)

        return val_time_emb, space_emb, var_idx_true, mask
