from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        scale=None,
        attention_dropout=0.1,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)  # expand heads dimension
            scores.masked_fill_(attn_mask, -np.inf)

        A = torch.nan_to_num(self.dropout(torch.softmax(scale * scores, dim=-1)))
        # A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if output_attn:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
    ):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(
            L_K, (L_Q, sample_k)
        )  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
        ]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(
        self, context_in, V, scores, index, L_Q, attn_mask, output_attn
    ):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ] = torch.matmul(attn, V).type_as(context_in)
        if output_attn:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask, output_attn=output_attn
        )

        return context.transpose(2, 1).contiguous(), attn


from performer_pytorch import FastAttention as _FastAttention


class PerformerAttention(_FastAttention):
    def __init__(
        self,
        mask_flag=False,
        dim_heads=None,
        ortho_scaling=0,
        feature_redraw_interval=1000,
        kernel="softmax",
    ):
        assert dim_heads is not None
        super().__init__(
            dim_heads=dim_heads,
            ortho_scaling=ortho_scaling,
            nb_features=max(100, int(dim_heads * log(dim_heads))),
            causal=mask_flag,
            generalized_attention=kernel == "relu",
            kernel_fn=nn.ReLU() if kernel == "relu" else "N/A",
        )
        self.redraw_interval = feature_redraw_interval
        self.register_buffer("calls_since_last_redraw", torch.tensor(0))

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        if self.training:
            if self.calls_since_last_redraw >= self.redraw_interval:
                self.redraw_projection_matrix(queries.device)
                self.calls_since_last_redraw.zero_()
            else:
                self.calls_since_last_redraw += 1

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        v = super().forward(queries, keys, values)

        return v.transpose(1, 2), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_queries_keys,
        d_values,
        n_heads,
        dropout_qkv=0.0,
        mix=False,
    ):
        super(AttentionLayer, self).__init__()

        self.inner_attention = attention()
        self.query_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_queries_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, output_attn=False):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.dropout_qkv(self.query_projection(queries)).view(B, L, H, -1)
        keys = self.dropout_qkv(self.key_projection(keys)).view(B, S, H, -1)
        values = self.dropout_qkv(self.value_projection(values)).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask,
            output_attn=output_attn,
        )

        if output_attn and attn is None:
            onehot_values = (
                torch.eye(S).unsqueeze(0).repeat(B, 1, 1).unsqueeze(2).to(values.device)
            )
            with torch.no_grad():
                attn, _ = self.inner_attention(
                    queries=queries,
                    keys=keys,
                    values=onehot_values,
                    attn_mask=attn_mask,
                )
                attn = attn.transpose(2, 1)

        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        if not output_attn:
            assert attn is None

        out = self.out_projection(out)
        return out, attn
