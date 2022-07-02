import random

import torch
from torch import nn
from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial


def create_subsequence_mask(o, r=0.15, lm=3, stateful=True, sync=False):
    # mask random subsequences of the input
    # (borrowed from IBM codeflare)
    if r <= 0:
        return torch.zeros_like(o).bool()
    device = o.device
    if o.ndim == 2:
        o = o[None]
    n_masks, mask_dims, mask_len = o.shape
    if sync == "random":
        sync = random.random() > 0.5
    dims = 1 if sync else mask_dims
    if stateful:
        numels = n_masks * dims * mask_len
        pm = torch.tensor([1 / lm], device=device)
        pu = torch.clip(pm * (r / max(1e-6, 1 - r)), 1e-3, 1)
        zot, proba_a, proba_b = (
            (torch.as_tensor([False, True], device=device), pu, pm)
            if random.random() > pm
            else (torch.as_tensor([True, False], device=device), pm, pu)
        )
        max_len = max(
            1,
            2
            * torch.div(numels, (1 / pm + 1 / pu), rounding_mode="floor").long().item(),
        )
        for i in range(10):
            _dist_a = (Geometric(probs=proba_a).sample([max_len]) + 1).long()
            _dist_b = (Geometric(probs=proba_b).sample([max_len]) + 1).long()
            dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)
            dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)
            add = torch.add(dist_a, dist_b)
            if torch.gt(torch.sum(add), numels):
                break
        dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
        if dist_len % 2:
            dist_len += 1
        repeats = torch.cat((dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
        zot = zot.repeat(dist_len)
        mask = torch.repeat_interleave(zot, repeats)[:numels].reshape(
            n_masks, dims, mask_len
        )
    else:
        probs = torch.tensor(r, device=device)
        mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
    if sync:
        mask = mask.repeat(1, mask_dims, 1)
    return mask


class ReconstructionDropout(nn.Module):
    def __init__(
        self,
        drop_full_timesteps=0.0,
        drop_standard=0.0,
        drop_seq=0.0,
        drop_max_seq_len=5,
        skip_all_drop=1.0,
    ):
        super().__init__()
        self.drop_full_timesteps = drop_full_timesteps
        self.drop_standard = drop_standard
        self.drop_seq = drop_seq
        self.drop_max_seq_len = drop_max_seq_len
        self.skip_all_drop = skip_all_drop

    def forward(self, y):
        bs, length, dim = y.shape
        dev = y.device

        if self.training and self.skip_all_drop < 1.0:
            # mask full timesteps
            full_timestep_mask = torch.bernoulli(
                (1.0 - self.drop_full_timesteps) * torch.ones(bs, length, 1)
            ).to(dev)

            # mask each element indp
            standard_mask = torch.bernoulli(
                (1.0 - self.drop_standard) * torch.ones(bs, length, dim)
            ).to(dev)

            # subsequence mask
            seq_mask = (
                1.0
                - create_subsequence_mask(
                    y.transpose(1, 2), r=self.drop_seq, lm=self.drop_max_seq_len
                )
                .transpose(1, 2)
                .float()
            )

            # skip all dropout occasionally so when there is no dropout
            # at test time the model has seen that before. (I am not sure
            # the usual activation strength adjustment makes sense here)
            skip_all_drop_mask = torch.bernoulli(
                1.0 - self.skip_all_drop * torch.ones(bs, 1, 1)
            ).to(dev)

            mask = 1.0 - (
                (1.0 - (full_timestep_mask * standard_mask * seq_mask))
                * skip_all_drop_mask
            )

            return y * mask
        else:
            return y

    def __repr__(self):
        return f"Timesteps {self.drop_full_timesteps}, Standard {self.drop_standard}, Seq (max len = {self.drop_max_seq_len}) {self.drop_seq}, Skip All Drop {self.skip_all_drop}"


class RandomMask(nn.Module):
    def __init__(self, prob, change_to_val):
        super().__init__()
        self.prob = prob
        self.change_to_val = change_to_val

    def forward(self, y):
        bs, length, dy = y.shape
        if not self.training or self.change_to_val is None:
            return y
        mask = torch.bernoulli((1.0 - self.prob) * torch.ones(bs, length, 1))
        mask.requires_grad = False
        mask = mask.to(y.device)
        masked_y = (y * mask) + (self.change_to_val * (1.0 - mask))
        return masked_y

    def __repr__(self):
        return f"RandomMask(prob = {self.prob}, val = {self.change_to_val}"
