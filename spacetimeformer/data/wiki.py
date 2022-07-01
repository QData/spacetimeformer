import os
import datetime
import random
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class Wikipedia:
    def __init__(self, data_path, drop_na=False):
        path = os.path.join(data_path, "train_2.csv")
        self.df = pd.read_csv(path)
        self.times = pd.to_datetime(self.df.columns[1:], format="%Y-%m-%d")

        if drop_na:
            self.df.dropna(axis=0, how="any", inplace=True)

    def __len__(self):
        return len(self.df)

    def get_series(self, i):
        y = self.df.iloc[i, 1:].astype(np.float32)
        nan_mask = y.notna()
        y = y[nan_mask]
        times = self.times[nan_mask]

        # years = list(map(lambda x: (x.year - 2015) / 2., times))
        months = list(map(lambda x: x.month / 12, times))
        days = list(map(lambda x: x.day / 31, times))

        x = pd.DataFrame(
            {
                "month": months,
                "day": days,
            }
        )
        y = pd.DataFrame({"y": y})
        return x, y


class WikipediaTorchDset(Dataset):
    def __init__(
        self, data_path: str, split, forecast_duration: int = 64, max_len=1000
    ):
        super().__init__()
        self.data = Wikipedia(data_path=data_path)
        self.max_len = max_len
        self.duration = forecast_duration
        self.split = split

        self.idxs = list(range(len(self.data)))

        if split == "train":
            _path = os.path.join(os.path.dirname(__file__), "wiki_train_invalid.csv")
            with open(_path, "r") as f:
                invalid_idxs = f.readlines()
            for idx in invalid_idxs:
                self.idxs.remove(int(idx))

    def __len__(self):
        return len(self.idxs)

    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    @staticmethod
    def scale(y):
        if isinstance(y, torch.Tensor):
            return torch.log1p(y)
        elif isinstance(y, np.ndarray):
            return np.log1p(y)
        else:
            raise TypeError("WikiTorchDset.scale")

    @staticmethod
    def inverse_scale(y):
        if isinstance(y, torch.Tensor):
            return torch.expm1(y)
        elif isinstance(y, np.ndarray):
            return np.expm1(y)
        else:
            raise TypeError("WikiTorchDset.inverse_scale")

    def __getitem__(self, i):
        x, y = self.data.get_series(self.idxs[i])

        test_set_split = len(y) - self.duration
        if self.split == "train":
            last_start_point = test_set_split - self.duration + 1
            if last_start_point <= self.duration:
                raise IndexError(f"series {i} has length {len(y)}")
            start_point = random.randrange(self.duration, last_start_point)
        else:
            start_point = test_set_split

        yt = y[start_point : start_point + self.duration]
        xt = x[start_point : start_point + self.duration]

        yc = y[:start_point]
        xc = x[:start_point]

        xc, yc, xt, yt = self._torch(xc, yc, xt, yt)

        yc = self.scale(yc)
        yt = self.scale(yt)

        xc = xc[-self.max_len :]
        yc = yc[-self.max_len :]
        return xc, yc, xt, yt

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")
        parser.add_argument("--max_len", type=int, default=1000)
        parser.add_argument("--forecast_duration", type=int, default=63)


def pad_wiki_collate(samples):
    xc = pad_sequence([x[0] for x in samples], batch_first=True, padding_value=-1.0)
    yc = pad_sequence([y[1] for y in samples], batch_first=True, padding_value=-1.0)
    xt = pad_sequence([x[2] for x in samples], batch_first=True, padding_value=-1.0)
    yt = pad_sequence([y[3] for y in samples], batch_first=True, padding_value=-1.0)
    return xc, yc, xt, yt


if __name__ == "__main__":
    from spacetimeformer.data import DataModule

    DATA_PATH = "TODO"
    dm = DataModule(
        WikipediaTorchDset,
        dataset_kwargs={
            "data_path": DATA_PATH,
            "forecast_duration": 63,
            "max_len": 1000,
        },
        batch_size=10,
        workers=0,
        collate_fn=pad_wiki_collate,
    )
    dset = WikipediaTorchDset(path, split="train")
    """
    for i in range(len(dset)):
        try:
            xc, yc, xt, yt = dset[i]
        except IndexError:
            print(i)
            continue
    """
    for batch in dm.train_dataloader():
        xc, yc, xt, yt = batch
        print(yc.shape)
        print("min max mean std")
        print(yc.min())
        print(yc.max())
        print(yc.mean())
        print(yc.std())
        input()
