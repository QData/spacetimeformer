import os
import datetime
import random
from typing import List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class M4:
    def __init__(self, data_path, resolutions=None):
        all_resolutions = [
            "yearly",
            "quarterly",
            "monthly",
            "weekly",
            "daily",
            "hourly",
        ]
        resolutions = resolutions or all_resolutions

        load = (
            lambda split, res: pd.read_csv(
                os.path.join(
                    data_path,
                    "Train" if split == "train" else "Test",
                    f"{res.capitalize()}-{split}.csv",
                )
            )
            if res in resolutions
            else pd.DataFrame()
        )

        concat = lambda tr, te: pd.concat(tr[:, 1:], te[:, 1:], axis=1)

        self.train_dfs = {res: load("train", res) for res in all_resolutions}
        self.test_dfs = {res: load("test", res) for res in all_resolutions}

        self.resolution_totals = {
            resolution: len(df) for resolution, df in self.train_dfs.items()
        }
        self.resolution_freq = {
            k: v / sum(self.resolution_totals.values())
            for k, v in self.resolution_totals.items()
        }

    def get_series(self, i):
        hour_cutoff = self.resolution_totals["hourly"]
        daily_cutoff = hour_cutoff + self.resolution_totals["daily"]
        weekly_cutoff = daily_cutoff + self.resolution_totals["weekly"]
        monthly_cutoff = weekly_cutoff + self.resolution_totals["monthly"]
        quarterly_cutoff = monthly_cutoff + self.resolution_totals["quarterly"]
        yearly_cutoff = quarterly_cutoff + self.resolution_totals["yearly"]

        if i < hour_cutoff:
            return self._get_series("hourly", i)
        elif i < daily_cutoff:
            return self._get_series("daily", i - hour_cutoff)
        elif i < weekly_cutoff:
            return self._get_series("weekly", i - daily_cutoff)
        elif i < monthly_cutoff:
            return self._get_series("monthly", i - weekly_cutoff)
        elif i < quarterly_cutoff:
            return self._get_series("quarterly", i - monthly_cutoff)
        elif i < yearly_cutoff:
            return self._get_series("yearly", i - quarterly_cutoff)
        else:
            raise IndexError(f"Index Eror for M4.get_series with index i = {i}")

    def _get_series(self, resolution, i):
        assert resolution in self.train_dfs
        train_series = self.train_dfs[resolution].iloc[i, 1:].astype(np.float32)
        test_series = self.test_dfs[resolution].iloc[i, 1:].astype(np.float32)
        y = np.concatenate(
            (train_series[train_series.notna()].values, test_series.values)
        )

        # M4 doesn't give a time axis, which I think was supposed
        # to make it harder to cheat and identify the real-life time series
        # in the database. We create a fake time axis that starts
        # on Jan 1 2000. This choice should not matter as long as we're
        # consistent.
        current_time = datetime.datetime(
            year=1, month=1, day=1, hour=0, minute=0, second=0
        )
        if resolution == "yearly":
            delta = datetime.timedelta(days=365)
        elif resolution == "quarterly":
            delta = datetime.timedelta(days=365 // 4)
        elif resolution == "monthly":
            delta = datetime.timedelta(days=365 // 12)
        elif resolution == "weekly":
            delta = datetime.timedelta(weeks=1)
        elif resolution == "daily":
            delta = datetime.timedelta(days=1)
        elif resolution == "hourly":
            delta = datetime.timedelta(hours=1)

        times = [current_time]
        for t in range(len(y) - 1):
            current_time += delta
            times.append(current_time)

        years = list(map(lambda x: x.year / 1000, times))
        months = list(map(lambda x: x.month / 12, times))
        days = list(map(lambda x: x.day / 31, times))
        hours = list(map(lambda x: x.hour / 24, times))

        x = pd.DataFrame(
            {
                "year": years,
                "month": months,
                "day": days,
                "hour": hours,
            }
        )
        y = pd.DataFrame({"y": y})
        return resolution, (x, y)


class M4TorchDset(Dataset):
    def __init__(self, data_path: str, split, resolutions=None, max_len=1000):
        super().__init__()
        self.m4data = M4(data_path=data_path, resolutions=resolutions)
        self.max_len = max_len
        self.split = split

        # forecast durations we're expcected
        # to make according to M4 overview paper
        self.durations = {
            "yearly": 6,
            "quarterly": 8,
            "monthly": 18,
            "weekly": 13,
            "daily": 14,
            "hourly": 48,
        }

    def __len__(self):
        return sum(self.m4data.resolution_totals.values())

    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    def __getitem__(self, i):
        resolution, (x, y) = self.m4data.get_series(i)
        duration = self.durations[resolution]
        test_set_split = len(y) - duration
        if self.split == "train":
            last_start_point = test_set_split - duration + 1
            assert duration < last_start_point
            start_point = random.randrange(duration, last_start_point)
        else:
            start_point = test_set_split
        yt = y[start_point : start_point + duration]
        xt = x[start_point : start_point + duration]

        yc = y[:start_point]
        xc = x[:start_point]

        xc, yc, xt, yt = self._torch(xc, yc, xt, yt)

        min_ = yc.min()
        max_ = abs(yc - min_).max()
        yc -= min_
        yt -= min_
        if max_ > 1e-3:
            yc /= max_
            yt /= max_

        xc = xc[-self.max_len :]
        yc = yc[-self.max_len :]
        return xc, yc, xt, yt

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")
        parser.add_argument("--resolutions", type=str, nargs="+", default=None)
        parser.add_argument("--max_len", type=int, default=1000)


def pad_m4_collate(samples):
    xc = pad_sequence([x[0] for x in samples], batch_first=True, padding_value=-1.0)
    yc = pad_sequence([y[1] for y in samples], batch_first=True, padding_value=-1.0)
    xt = pad_sequence([x[2] for x in samples], batch_first=True, padding_value=-1.0)
    yt = pad_sequence([y[3] for y in samples], batch_first=True, padding_value=-1.0)
    return xc, yc, xt, yt


if __name__ == "__main__":
    from spacetimeformer.data import DataModule

    path = "TODO"
    dm = DataModule(
        M4TorchDset,
        dataset_kwargs={
            "data_path": path,
            "resolutions": None,
            "max_len": 1000,
        },
        batch_size=32,
        workers=0,
        collate_fn=pad_m4_collate,
    )
    for batch in dm.train_dataloader():
        xc, yc, xt, yt = batch
        print(yc.min())
        print(yc.max())
        print(yt.min())
        print(yt.max())
        input()
