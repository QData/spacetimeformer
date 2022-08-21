import numpy as np
import os
import pickle

import torch
from torch.utils.data import TensorDataset
import pandas as pd
from einops import rearrange


class MetroData:
    def _read(self, split):
        with open(os.path.join(self.path, f"{split}.pkl"), "rb") as f:
            data = pickle.load(f)
            x_c = self._convert_time(data["xtime"])
            y_c = data["x"]
            x_t = self._convert_time(data["ytime"])
            y_t = data["y"]
            y_c = rearrange(
                y_c, "batch time vars features -> batch time (vars features)"
            )
            y_t = rearrange(
                y_t, "batch time vars features -> batch time (vars features)"
            )
        return x_c, y_c, x_t, y_t

    def _convert_time(self, raw_time):
        datetimes = pd.DatetimeIndex(raw_time)
        times = []
        for i in range(len(raw_time)):
            times.append(pd.DatetimeIndex(raw_time[i]))

        month = np.array([time.month.values for time in times]) / 12.0
        day = np.array([time.day.values for time in times]) / 31.0
        hour = np.array([time.hour.values for time in times]) / 24.0
        minute = np.array([time.minute.values for time in times]) / 60.0
        converted = np.stack([month, day, hour, minute], axis=-1)
        return converted

    def __init__(self, path):
        self.path = path

        x_c_train, y_c_train, x_t_train, y_t_train = self._read("train")
        x_c_val, y_c_val, x_t_val, y_t_val = self._read("val")
        x_c_test, y_c_test, x_t_test, y_t_test = self._read("test")

        # self._scale_max = y_c_train.max((0, 1))
        self.scale_mean = y_c_train.mean((0, 1))
        self.scale_std = y_c_train.std((0, 1))

        y_c_train = self.scale(y_c_train)
        y_t_train = self.scale(y_t_train)

        y_c_val = self.scale(y_c_val)
        y_t_val = self.scale(y_t_val)

        y_c_test = self.scale(y_c_test)
        y_t_test = self.scale(y_t_test)

        self.train_data = (x_c_train, y_c_train, x_t_train, y_t_train)
        self.val_data = (x_c_val, y_c_val, x_t_val, y_t_val)
        self.test_data = (x_c_test, y_c_test, x_t_test, y_t_test)

    def scale(self, x):
        x = (x != 0.0) * ((x - self.scale_mean) / self.scale_std)
        # return (x / self._scale_max) * 100.
        # return (x - self.scale_mean) / self.scale_std
        return x

    def inverse_scale(self, x):
        x = (x != 0.0) * ((x * self.scale_std) + self.scale_mean)
        # return (x / 100.) * self._scale_max
        # return (x * self.scale_std) + self.scale_mean
        return x

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--context_points", type=int, default=4)
        parser.add_argument("--target_points", type=int, default=4)


def MetroTorch(data: MetroData, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        tensors = data.train_data
    elif split == "val":
        tensors = data.val_data
    else:
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors)
