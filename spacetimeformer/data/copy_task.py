from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np


def _make_copy_sequence(L: int, N: int, lag_N: List[int], p: float, include_lags: bool):
    assert len(lag_N) == N

    seq = np.random.choice([0, 1], size=(L, N), p=[1.0 - p, p])

    lagged_seq = np.zeros_like(seq)
    for i in range(N):
        if lag_N[i]:
            lagged_seq[lag_N[i] :, i] = seq[: -(lag_N[i]), i]
        else:
            lagged_seq[:, i] = seq[:, i]

    x = np.arange(0, L + 1)[:, np.newaxis] / L
    lag_N = np.array(lag_N)[np.newaxis, :]
    seq = np.concatenate((lag_N, seq), axis=0)

    x_c = x.astype(np.float32)
    y_c = seq.astype(np.float32)
    x_t = x[1:].astype(np.float32)
    y_t = lagged_seq.astype(np.float32)

    if not include_lags:
        x_c = x_c[1:]
        y_c = y_c[1:]

    return x_c, y_c, x_t, y_t


class CopyTaskDset(Dataset):
    def __init__(
        self,
        split,
        length: int,
        copy_vars: int = 4,
        lags: List[int] = None,
        mask_prob: float = 0.2,
        include_lags: bool = False,
    ):
        assert split in ["train", "val", "test"]
        if lags:
            assert len(lags) == copy_vars
        else:
            lags = [0 for _ in range(copy_vars)]
        self.L = length
        self.N = copy_vars
        self.lag_N = lags
        self.p = mask_prob
        self.split = split
        self.include_lags = include_lags

    def __len__(self):
        # arbitrary; determines lightning epoch length
        if self.split == "train":
            return 100_000
        else:
            return 5_000

    def __getitem__(self, i):
        return _make_copy_sequence(
            self.L, self.N, self.lag_N, self.p, self.include_lags
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--copy_length", type=int, default=20)
        parser.add_argument("--copy_vars", type=int, default=4)
        parser.add_argument("--copy_lags", type=int, nargs="+", default=None)
        parser.add_argument("--copy_mask_prob", type=float, default=0.2)
        parser.add_argument("--copy_include_lags", action="store_true")
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x_c, y_c, x_t, y_t = _make_copy_sequence(100, 5, [0, 5, 10, 15, 25], p=0.1)

    red = [255, 102, 102]
    orange = [255, 179, 25]
    green = [98, 181, 60]
    light_blue = [41, 255, 234]
    purple = [159, 41, 255]
    colors = np.array([red, green, orange, light_blue, purple])
    original = (y_c[1:, :, np.newaxis] * colors).astype(np.uint8).transpose(1, 0, 2)
    copied = (y_t[..., np.newaxis] * colors).astype(np.uint8).transpose(1, 0, 2)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].imshow(original)
    axs[1].imshow(copied)
    plt.tight_layout()
    plt.show()
