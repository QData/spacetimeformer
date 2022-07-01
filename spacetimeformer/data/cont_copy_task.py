from typing import List

import torch
from torch.utils.data import Dataset
import numpy as np


def _make_copy_sequence(
    L: int,
    N: int,
    lag_N: List[int],
    include_lags: bool,
    split: str,
    magnitude_matters: bool,
    freq_shift: bool,
):
    assert len(lag_N) == N

    a = 1.0 if not freq_shift else np.random.uniform(0.9, 1.1, size=(N,))
    b = 0.0 if not freq_shift else np.random.uniform(-0.1, 0.1, size=(N,))
    c = 1.0 if not freq_shift else np.random.uniform(0.9, 1.1, size=(N,))
    if split == "train":
        random_start = np.random.uniform(-1.0, 1.0, size=(N,))
    elif split == "val":
        random_start = np.random.uniform(0.0, 2.0, size=(N,))
        a = a if not freq_shift else np.random.uniform(0.8, 1.2, size=(N,))
        b = b if not freq_shift else np.random.uniform(-0.3, 0.3, size=(N,))
        c = c if not freq_shift else np.random.uniform(0.7, 1.3, size=(N,))
    elif split == "test":
        random_start = np.random.uniform(2.0, 4.0, size=(N,))
        a = a if not freq_shift else np.random.uniform(0.7, 1.3, size=(N,))
        b = b if not freq_shift else np.random.uniform(-1, 1, size=(N,))
        c = c if not freq_shift else np.random.uniform(0.25, 1.75, size=(N,))

    seq = (
        random_start
        + a * np.sin(c * np.expand_dims(np.arange(L), 1) + b)
        + np.random.randn(L, N) * 0.2
    )

    lagged_seq = np.ones_like(seq) * random_start
    for i in range(N):
        if lag_N[i]:
            lagged_seq[lag_N[i] :, i] = seq[: -(lag_N[i]), i]
        else:
            lagged_seq[:, i] = seq[:, i]

    if magnitude_matters:
        lagged_seq += (1.0 + abs(random_start)) ** 2

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


class ContCopyTaskDset(Dataset):
    def __init__(
        self,
        split,
        length: int,
        copy_vars: int = 4,
        lags: List[int] = None,
        include_lags: bool = False,
        magnitude_matters: bool = False,
        freq_shift: bool = False,
    ):
        assert split in ["train", "val", "test"]
        if lags is None:
            lags = [j for j in range(0, length, length // copy_vars)]
        assert len(lags) == copy_vars
        self.L = length
        self.N = copy_vars
        self.lag_N = lags
        self.split = split
        self.include_lags = include_lags
        self.magnitude_matters = magnitude_matters
        self.freq_shift = freq_shift

    def __len__(self):
        # arbitrary; determines lightning epoch length
        if self.split == "train":
            return 100_000
        else:
            return 5_000

    def __getitem__(self, i):
        return _make_copy_sequence(
            L=self.L,
            N=self.N,
            lag_N=self.lag_N,
            include_lags=self.include_lags,
            split=self.split,
            magnitude_matters=self.magnitude_matters,
            freq_shift=self.freq_shift,
        )

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--copy_length", type=int, default=20)
        parser.add_argument("--copy_vars", type=int, default=4)
        parser.add_argument("--copy_lags", type=int, nargs="+", default=None)
        parser.add_argument("--copy_include_lags", action="store_true")
        parser.add_argument("--copy_mag_matters", action="store_true")
        parser.add_argument("--copy_freq_shift", action="store_true")
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
