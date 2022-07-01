import numpy as np
import os

import torch
from torch.utils.data import TensorDataset


class METR_LA_Data:
    def _read(self, split):
        with np.load(os.path.join(self.path, f"{split}.npz")) as f:
            x = f["x"]
            y = f["y"]
        return x, y

    def _split_set(self, data):
        # time features are the same across the 207 nodes.
        # just grab the first one
        x = np.squeeze(data[:, :, 0, 1:])

        # normalize time of day
        time = 2.0 * x[:, :, 0] - 1.0

        # convert one-hot day of week feature to scalar [-1, 1]
        day_of_week = x[:, :, 1:]
        day_of_week = np.argmax(day_of_week, axis=-1).astype(np.float32)
        day_of_week = 2.0 * (day_of_week / 6.0) - 1.0
        time = np.expand_dims(time, axis=-1)

        # x has 2 features: time and day of week
        day_of_week = np.expand_dims(day_of_week, axis=-1)
        x = np.concatenate((time, day_of_week), axis=-1)

        y = data[:, :, :, 0]
        return x, y

    def __init__(self, path):
        self.path = path

        context_train, target_train = self._read("train")
        context_val, target_val = self._read("val")
        context_test, target_test = self._read("test")

        x_c_train, y_c_train = self._split_set(context_train)
        x_t_train, y_t_train = self._split_set(target_train)

        x_c_val, y_c_val = self._split_set(context_val)
        x_t_val, y_t_val = self._split_set(target_val)

        x_c_test, y_c_test = self._split_set(context_test)
        x_t_test, y_t_test = self._split_set(target_test)

        self.scale_max = y_c_train.max((0, 1))

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
        return x / self.scale_max

    def inverse_scale(self, x):
        return x * self.scale_max

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="./data/metr_la/")
        parser.add_argument("--context_points", type=int, default=12)

        parser.add_argument("--target_points", type=int, default=12)


def METR_LA_Torch(data: METR_LA_Data, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        tensors = data.train_data
    elif split == "val":
        tensors = data.val_data
    else:
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors)


if __name__ == "__main__":
    data = METR_LA_Data(path="./data/pems-bay/")
    dset = METR_LA_Torch(data, "test")
    breakpoint()
