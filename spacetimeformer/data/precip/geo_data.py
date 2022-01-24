from netCDF4 import Dataset as NCDFDataset
import os
import torch
from typing import Tuple
import random
import numpy as np
import glob
from torch.utils.data import Dataset as TorchDataset


class GeoDset:
    def __init__(
        self,
        dset_dir: str = "./data/usgs_precipitation/",
        var="precip",
        train_val_test: Tuple[float] = (0.6, 0.3, 0.1),
    ):
        files = glob.glob(os.path.join(dset_dir, "*.nc"))
        data = []
        for f in files:
            dset = NCDFDataset(f)
            data.append(dset[var][:].data)
        data = np.concatenate(data, axis=0)
        self.fill_value = dset[var][:].fill_value
        self.valid_locs = [
            loc for loc in zip(*np.where((data != self.fill_value).all(0)))
        ]

        # train/val/test split
        assert (
            abs(sum(train_val_test) - 1.0) < 1e-3
        ), "Train/Val/Test Split should sum to 1.0"
        train_split, val_split, test_split = train_val_test
        time = len(data)
        train_idx = round(time * train_split)
        val_idx = train_idx + round(time * val_split)
        test_idx = -round(time * test_split)
        self._train_data = data[:train_idx]
        self._val_data = data[train_idx:val_idx]
        self._test_data = data[test_idx:]

    @property
    def train_data(self):
        return self._train_data

    @property
    def val_data(self):
        return self._val_data

    @property
    def test_data(self):
        return self._test_data

    @classmethod
    def add_cli(self, parser):
        parser.add_argument(
            "--dset_dir", type=str, default="./data/usgs_precipitation/"
        )


class StationGridDset(TorchDataset):
    def __init__(
        self,
        dset: GeoDset,
        context_points: int = 30,
        target_points: int = 10,
        split="train",
        scaled: bool = True,
    ):

        assert split in ["train", "val", "test"]
        if split == "train":
            self.data = dset.train_data
        elif split == "val":
            self.data = dset.val_data
        else:
            self.data = dset.test_data

        self._start_points = [
            i for i in range(0, len(self.data) - context_points - target_points)
        ]
        random.shuffle(self._start_points)
        self.context_points = context_points
        self.target_points = target_points

        days = (torch.arange(0, len(self.data)).unsqueeze(-1) % 364).float()
        years = torch.round(
            (torch.arange(0, len(self.data)).unsqueeze(-1).float() / 364)
        )
        years /= years.max()
        self.timestamps = torch.cat((years, days), dim=-1)

        self.valid_locs = dset.valid_locs

        self.null_value = dset.fill_value

        if scaled:
            self.data = self.data / abs(dset.train_data).max(0)

    def get_locations(self, valid_locs):
        return NotImplementedError

    def __len__(self):
        return len(self._start_points)

    def __getitem__(self, i):
        start = self._start_points[i]

        full_slice = self.data[start : start + self.context_points + self.target_points]

        ys = []
        for station_x, station_y in self.get_locations(self.valid_locs):
            y = torch.from_numpy(full_slice[:, station_x, station_y]).unsqueeze(-1)
            ys.append(y)
        ys = torch.cat(ys, dim=-1)

        x = self.timestamps[start : start + self.context_points + self.target_points]
        x_c = x[: self.context_points]
        x_t = x[self.context_points :]

        y_c = ys[: self.context_points]
        y_t = ys[self.context_points :]

        return x_c, y_c, x_t, y_t

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--context_points", type=int, default=30)
        parser.add_argument("--target_points", type=int, default=10)


class CONUS_Precip(StationGridDset):
    RADAR = [
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    ]

    def get_locations(self, valid_locs):
        lat = random.randint(40, 90)
        lon = random.randint(20, 220)
        locations = []
        curr_lat, curr_lon = lat, lon
        for row in self.RADAR:
            curr_lat = lat
            for col in row:
                if col:
                    locations.append((curr_lat, curr_lon))
                curr_lat += 1
            curr_lon += 1
        return locations


if __name__ == "__main__":
    import spacetimeformer as stf

    precip_data = GeoDset()
    from torch.utils.data import DataLoader

    mod = DataLoader(
        CONUS_Precip(dset=precip_data, split="train"),
        batch_size=37,
    )
    for batch in mod:
        breakpoint()
