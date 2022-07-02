import random
from typing import List
import os
import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

import spacetimeformer as stf

import matplotlib.pyplot as plt


class CSVTimeSeries:
    def __init__(
        self,
        data_path: str = None,
        raw_df: pd.DataFrame = None,
        target_cols: List[str] = [],
        ignore_cols: List[str] = [],
        remove_target_from_context_cols: List[str] = [],
        time_col_name: str = "Datetime",
        read_csv_kwargs={},
        val_split: float = 0.15,
        test_split: float = 0.15,
        normalize: bool = True,
        drop_all_nan: bool = False,
        time_features: List[str] = [
            "year",
            "month",
            "day",
            "weekday",
            "hour",
            "minute",
        ],
    ):

        assert data_path is not None or raw_df is not None

        if raw_df is None:
            self.data_path = data_path
            assert os.path.exists(self.data_path)
            raw_df = pd.read_csv(
                self.data_path,
                **read_csv_kwargs,
            )

        if drop_all_nan:
            raw_df.dropna(axis=0, how="any", inplace=True)

        self.time_col_name = time_col_name
        assert self.time_col_name in raw_df.columns

        if not target_cols:
            target_cols = raw_df.columns.tolist()
            target_cols.remove(time_col_name)

        if ignore_cols:
            if ignore_cols == "all":
                ignore_cols = raw_df.columns.difference(target_cols).tolist()
                ignore_cols.remove(self.time_col_name)
            raw_df.drop(columns=ignore_cols, inplace=True)

        time_df = pd.to_datetime(raw_df[self.time_col_name], format="%Y-%m-%d %H:%M")
        df = stf.data.timefeatures.time_features(
            time_df,
            raw_df,
            time_col_name=self.time_col_name,
            use_features=time_features,
        )
        self.time_cols = df.columns.difference(raw_df.columns)

        # Train/Val/Test Split using holdout approach #

        def mask_intervals(mask, intervals, cond):
            for (interval_low, interval_high) in intervals:
                if interval_low is None:
                    interval_low = df[self.time_col_name].iloc[0].year
                if interval_high is None:
                    interval_high = df[self.time_col_name].iloc[-1].year
                mask[
                    (df[self.time_col_name] >= interval_low)
                    & (df[self.time_col_name] <= interval_high)
                ] = cond
            return mask

        test_cutoff = len(time_df) - max(round(test_split * len(time_df)), 1)
        val_cutoff = test_cutoff - round(val_split * len(time_df))

        val_interval_low = time_df.iloc[val_cutoff]
        val_interval_high = time_df.iloc[test_cutoff - 1]
        val_intervals = [(val_interval_low, val_interval_high)]

        test_interval_low = time_df.iloc[test_cutoff]
        test_interval_high = time_df.iloc[-1]
        test_intervals = [(test_interval_low, test_interval_high)]

        train_mask = df[self.time_col_name] > pd.Timestamp.min
        val_mask = df[self.time_col_name] > pd.Timestamp.max
        test_mask = df[self.time_col_name] > pd.Timestamp.max
        train_mask = mask_intervals(train_mask, test_intervals, False)
        train_mask = mask_intervals(train_mask, val_intervals, False)
        val_mask = mask_intervals(val_mask, val_intervals, True)
        test_mask = mask_intervals(test_mask, test_intervals, True)

        if (train_mask == False).all():
            print(f"No training data detected for file {data_path}")

        self._train_data = df[train_mask]
        self._scaler = StandardScaler()

        self.target_cols = target_cols
        for col in remove_target_from_context_cols:
            assert (
                col in self.target_cols
            ), "`remove_target_from_context_cols` should be target cols that you want to remove from the context"

        self.remove_target_from_context_cols = remove_target_from_context_cols
        not_exo_cols = self.time_cols.tolist() + target_cols
        self.exo_cols = df.columns.difference(not_exo_cols).tolist()
        self.exo_cols.remove(self.time_col_name)

        self._train_data = df[train_mask]
        self._val_data = df[val_mask]
        if test_split == 0.0:
            print("`test_split` set to 0. Using Val set as Test set.")
            self._test_data = df[val_mask]
        else:
            self._test_data = df[test_mask]

        self.normalize = normalize
        if normalize:
            self._scaler = self._scaler.fit(
                self._train_data[target_cols + self.exo_cols].values
            )
        self._train_data = self.apply_scaling_df(self._train_data)
        self._val_data = self.apply_scaling_df(self._val_data)
        self._test_data = self.apply_scaling_df(self._test_data)

    def make_hists(self):
        for col in self.target_cols + self.exo_cols:
            train = self._train_data[col]
            test = self._test_data[col]
            bins = np.linspace(-5, 5, 80)  # warning: edit bucket limits
            plt.hist(train, bins, alpha=0.5, label="Train", density=True)
            plt.hist(test, bins, alpha=0.5, label="Test", density=True)
            plt.legend(loc="upper right")
            plt.title(col)
            plt.tight_layout()
            plt.savefig(f"{col}-hist.png")
            plt.clf()

    def get_slice(self, split, start, stop, skip):
        assert split in ["train", "val", "test"]
        if split == "train":
            return self.train_data.iloc[start:stop:skip]
        elif split == "val":
            return self.val_data.iloc[start:stop:skip]
        else:
            return self.test_data.iloc[start:stop:skip]

    def apply_scaling(self, array):
        if not self.normalize:
            return array
        dim = array.shape[-1]
        return (array - self._scaler.mean_[:dim]) / self._scaler.scale_[:dim]

    def apply_scaling_df(self, df):
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = self.target_cols + self.exo_cols
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values - self._scaler.mean_.astype(dtype)
        ) / self._scaler.scale_.astype(dtype)
        return scaled

    def reverse_scaling_df(self, df):
        if not self.normalize:
            return df
        scaled = df.copy(deep=True)
        cols = self.target_cols + self.exo_cols
        dtype = df[cols].values.dtype
        scaled[cols] = (
            df[cols].values * self._scaler.scale_.astype(dtype)
        ) + self._scaler.mean_.astype(dtype)
        return scaled

    def reverse_scaling(self, array):
        if not self.normalize:
            return array
        # self._scaler is fit for target_cols + exo_cols
        # if the array dim is less than this length we start
        # slicing from the target cols
        dim = array.shape[-1]
        return (array * self._scaler.scale_[:dim]) + self._scaler.mean_[:dim]

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    def length(self, split):
        return {
            "train": len(self.train_data),
            "val": len(self.val_data),
            "test": len(self.test_data),
        }[split]

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="auto")


class CSVTorchDset(Dataset):
    def __init__(
        self,
        csv_time_series: CSVTimeSeries,
        split: str = "train",
        context_points: int = 128,
        target_points: int = 32,
        time_resolution: int = 1,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.series = csv_time_series
        self.context_points = context_points
        self.target_points = target_points
        self.time_resolution = time_resolution

        self._slice_start_points = [
            i
            for i in range(
                0,
                self.series.length(split)
                + time_resolution * (-target_points - context_points)
                + 1,
            )
        ]

    def __len__(self):
        return len(self._slice_start_points)

    def _torch(self, *dfs):
        return tuple(torch.from_numpy(x.values).float() for x in dfs)

    def __getitem__(self, i):
        start = self._slice_start_points[i]
        series_slice = self.series.get_slice(
            self.split,
            start=start,
            stop=start
            + self.time_resolution * (self.context_points + self.target_points),
            skip=self.time_resolution,
        )
        series_slice = series_slice.drop(columns=[self.series.time_col_name])
        ctxt_slice, trgt_slice = (
            series_slice.iloc[: self.context_points],
            series_slice.iloc[self.context_points :],
        )

        ctxt_x = ctxt_slice[self.series.time_cols]
        trgt_x = trgt_slice[self.series.time_cols]

        ctxt_y = ctxt_slice[self.series.target_cols + self.series.exo_cols]
        ctxt_y = ctxt_y.drop(columns=self.series.remove_target_from_context_cols)

        trgt_y = trgt_slice[self.series.target_cols]

        return self._torch(ctxt_x, ctxt_y, trgt_x, trgt_y)

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
            "--context_points",
            type=int,
            default=128,
            help="number of previous timesteps given to the model in order to make predictions",
        )
        parser.add_argument(
            "--target_points",
            type=int,
            default=32,
            help="number of future timesteps to predict",
        )
        parser.add_argument(
            "--time_resolution",
            type=int,
            default=1,
        )
