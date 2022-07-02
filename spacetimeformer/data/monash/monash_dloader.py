from dataclasses import dataclass
import random
import os
import glob
from datetime import datetime, timedelta
from distutils.util import strtobool

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import spacetimeformer as stf


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with=np.nan,
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )


class TSF_Data:
    def __init__(self, name: str, root_dir: str):
        assert ".tsf" not in name
        self.name = name
        path = os.path.join(root_dir, f"{name}.tsf")

        (
            data,
            self.freq,
            self.horizon,
            self.has_missing,
            self.equal_length,
        ) = convert_tsf_to_dataframe(path)

        self.data = data["series_value"]

    def __repr__(self):
        return self.name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        assert i < len(
            self
        ), f"series index {i} out of range for TSF dataset with length {len(self)}"
        y = self.data.iloc[i].to_numpy()

        current_time = datetime(year=1, month=1, day=1, hour=0, minute=0, second=0)
        if self.freq == "yearly":
            delta = timedelta(days=365)
        elif self.freq == "quarterly":
            delta = timedelta(days=365 // 4)
        elif self.freq == "monthly":
            delta = timedelta(days=365 // 12)
        elif self.freq == "weekly":
            delta = timedelta(weeks=1)
        elif self.freq == "daily":
            delta = timedelta(days=1)
        elif self.freq == "hourly":
            delta = timedelta(hours=1)
        times = [current_time]
        for t in range(len(y) - 1):
            current_time += delta
            times.append(current_time)
        times = np.array(times)

        nan_mask = ~np.isnan(y)
        y = y[nan_mask]
        times = times[nan_mask].tolist()

        years = list(map(lambda x: x.year / 1000, times))
        months = list(map(lambda x: x.month / 12, times))
        days = list(map(lambda x: x.day / 31, times))
        hours = list(map(lambda x: x.hour / 24, times))

        y = pd.DataFrame({"y": y})
        x = pd.DataFrame(
            {
                "year": years,
                "month": months,
                "day": days,
                "hour": hours,
            }
        )
        return x, y


class MonashScaler:
    def __call__(self, data, reference_data):
        return data


class MonashInvScaler:
    def __call__(self, data, reference_data):
        return data


class PositiveZeroOneLog(MonashScaler):
    def __call__(self, data, reference_data):
        min_ = reference_data.min()
        max_ = abs(reference_data - min_).max()
        scaled_data = data - min_
        if max_ > 1e-3:
            scaled_data /= max_
        sign = np.sign(scaled_data + 1e-3) * 2.0 - 1
        val = np.log1p(abs(scaled_data))
        final = sign * val
        return final


class LogScale(MonashScaler):
    def __call__(self, data, reference_data):
        return np.log1p(data)


class MonashDset:
    def __init__(
        self,
        raw_data: TSF_Data,
        horizon: int,
        max_length: int,
        scale: MonashScaler = None,
        inv_scale: MonashInvScaler = None,
        randomize_train_horizon: bool = False,
    ):
        self.dset = raw_data
        self.horizon = horizon
        self.max_length = max_length
        self.randomize_train_horizon = randomize_train_horizon

        self._scale = scale or MonashScaler()
        assert isinstance(self._scale, MonashScaler)
        self._inv_scale = inv_scale or MonashInvScaler()
        assert isinstance(self._inv_scale, MonashInvScaler)

    def __repr__(self):
        return str(self.dset)

    def __len__(self):
        return len(self.dset)

    def _to_np(self, *dfs):
        return tuple(np.array(df.values).astype(np.float32) for df in dfs)

    def get_series(self, i, split):
        assert split in ["train", "test"]
        x, y = self.dset[i]

        test_set_split = len(y) - self.horizon
        if split == "train":
            last_start_point = test_set_split - self.horizon + 1
            if last_start_point <= self.horizon:
                # there are some datasets that have series too short to make
                # a train split (so far this applies only to wikipedia)
                # for now we'll do nothing about this
                start_point = 1
            else:
                start_point = random.randrange(self.horizon, last_start_point)
            if self.randomize_train_horizon:
                horizon = random.randrange(1, self.horizon + 1)
            else:
                horizon = self.horizon
        else:
            start_point = test_set_split
            horizon = self.horizon

        yt = y[start_point : start_point + horizon]
        xt = x[start_point : start_point + horizon]

        yc = y[:start_point]
        xc = x[:start_point]

        xc, yc, xt, yt = self._to_np(xc, yc, xt, yt)

        yc_scaled = self._scale(yc, yc)
        # if we need to scale, do it based on
        # "train" data (context sequence)
        yt_scaled = self._scale(yt, yc)

        # option to trim length to prevent OOM issues
        xc = xc[-self.max_length :]
        yc_scaled = yc_scaled[-self.max_length :]

        return xc, yc_scaled, xt, yt_scaled

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--max_len", type=int, default=1000)
        parser.add_argument("--include", type=str, nargs="+", default="all")
        parser.add_argument(
            "--root_dir", type=str, default="/dccstor/tst03/datasets/monash/"
        )


class MetaMonashDset(Dataset):
    def __init__(self, dsets, split: str):
        assert split in ["train", "val", "test"]
        self.split = split
        ## TODO: no good val split right now ##
        if self.split == "val":
            self.split = "test"
        #######################################
        self.dsets = dsets
        self.lengths = [len(d) for d in dsets]

    def __repr__(self):
        return f"{[str(dset) for dset in self.dsets]}"

    def __len__(self):
        return sum(self.lengths)

    def _torch(self, *np_arrays):
        return (torch.from_numpy(x).float() for x in np_arrays)

    def __getitem__(self, i):
        total = 0
        for dset_idx, len_ in enumerate(self.lengths):
            if i < total + len_:
                break
            total += len_
        idx = i - total

        dset = self.dsets[dset_idx]
        xc, yc, xt, yt = self._torch(*dset.get_series(idx, self.split))

        return xc, yc, xt, yt


def load_monash_dsets(root_dir, max_len, include=["all"]):
    check = lambda name: "all" in include or name in include

    all_dsets = []

    if check("dominick"):
        dominick_raw = TSF_Data("dominick_dataset", root_dir=root_dir)
        dominick = MonashDset(
            dominick_raw, horizon=8, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(dominick)

    if check("nn5"):
        # checked
        nn5_raw = TSF_Data("nn5_daily_dataset_with_missing_values", root_dir=root_dir)
        nn5 = MonashDset(
            nn5_raw, horizon=30, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(nn5)

    if check("cif"):
        # checked
        cif_raw = TSF_Data("cif_2016_dataset", root_dir=root_dir)
        cif = MonashDset(
            cif_raw, horizon=12, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(cif)

    """
    if check("car_parts"):
        car_parts_raw = TSF_Data("car_parts_dataset_with_missing_values", root_dir=root_dir)
        car_parts = MonashDset(car_parts_raw, horizon=8, max_length=max_len, scale=PositiveZeroOneLog())
        all_dsets.append(car_parts)
    """

    if check("rideshare"):
        # checked
        # they try to use 168 horizon in the paper but this data is too short for our setup
        rideshare_raw = TSF_Data(
            "rideshare_dataset_with_missing_values", root_dir=root_dir
        )
        rideshare = MonashDset(
            rideshare_raw, horizon=20, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(rideshare)

    if check("tourism"):
        # checked
        tourism_raw = TSF_Data("tourism_monthly_dataset", root_dir=root_dir)
        tourism = MonashDset(
            tourism_raw, horizon=12, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(tourism)

    if check("m3"):
        # checked
        all_m3s = []
        for freq, horizon in [
            ("monthly", 18),
            ("quarterly", 8),
            ("yearly", 6),
        ]:
            m3_raw = TSF_Data(f"m3_{freq}_dataset", root_dir=root_dir)
            m3 = MonashDset(
                m3_raw, horizon=horizon, max_length=max_len, scale=PositiveZeroOneLog()
            )
            all_m3s.append(m3)
        all_dsets += all_m3s

    if check("m1"):
        # checked
        all_m1s = []
        for freq, horizon in [
            ("monthly", 18),
            ("quarterly", 8),
            ("yearly", 6),
        ]:
            m1_raw = TSF_Data(f"m1_{freq}_dataset", root_dir=root_dir)
            m1 = MonashDset(
                m1_raw, horizon=horizon, max_length=max_len, scale=PositiveZeroOneLog()
            )
            all_m1s.append(m1)
        all_dsets += all_m1s

    if check("m4"):
        all_m4s = []
        for freq, horizon in [
            ("daily", 14),
            ("hourly", 48),
            ("monthly", 18),
            ("quarterly", 8),
            ("weekly", 13),
            ("yearly", 6),
        ]:
            m4_raw = TSF_Data(f"m4_{freq}_dataset", root_dir=root_dir)
            m4 = MonashDset(
                m4_raw, horizon=horizon, max_length=max_len, scale=PositiveZeroOneLog()
            )
            all_m4s.append(m4)
        all_dsets += all_m4s

    if check("wiki"):
        wiki_raw = TSF_Data(
            "kaggle_web_traffic_dataset_with_missing_values",
            root_dir=root_dir,
        )
        wiki = MonashDset(wiki_raw, horizon=64, max_length=max_len, scale=LogScale())
        all_dsets.append(wiki)

    if check("covid"):
        # checked
        covid_raw = TSF_Data("covid_deaths_dataset", root_dir=root_dir)
        covid = MonashDset(
            covid_raw, horizon=30, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(covid)

    if check("fred"):
        # checked
        fred_raw = TSF_Data("fred_md_dataset", root_dir=root_dir)
        fred = MonashDset(
            fred_raw, horizon=12, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(fred)

    if check("hospital"):
        # checked
        hospital_raw = TSF_Data("hospital_dataset", root_dir=root_dir)
        hospital = MonashDset(
            hospital_raw, horizon=12, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(hospital)

    if check("traffic"):
        # checked, pushes length limit
        traffic_raw = TSF_Data("traffic_hourly_dataset", root_dir=root_dir)
        traffic = MonashDset(
            traffic_raw, horizon=168, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(traffic)

    if check("temperature"):
        # checked, values get a bit extreme in some places
        temperature_raw = TSF_Data(
            "temperature_rain_dataset_with_missing_values", root_dir=root_dir
        )
        temperature = MonashDset(
            temperature_raw, horizon=30, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(temperature)

    """
    if check("vehicle"):
        # checked, some series a bit short
        vehicle_raw = TSF_Data(
            "vehicle_trips_dataset_with_missing_values", root_dir=root_dir
        )
        vehicle = MonashDset(
            vehicle_raw, horizon=30, max_length=max_len, scale=PositiveZeroOneLog()
        )
        all_dsets.append(vehicle)
    """
    return all_dsets


def quick_make_meta_monash(root_dir, max_len, include):
    dsets = load_monash_dsets(root_dir, max_len, include)
    train_dset = MetaMonashDset(dsets, "train")
    test_dset = MetaMonashDset(dsets, "test")
    return train_dset, test_dset


def pad_collate(samples):
    xc = pad_sequence([x[0] for x in samples], batch_first=True, padding_value=-64.0)
    yc = pad_sequence([y[1] for y in samples], batch_first=True, padding_value=-64.0)
    xt = pad_sequence([x[2] for x in samples], batch_first=True, padding_value=-64.0)
    yt = pad_sequence([y[3] for y in samples], batch_first=True, padding_value=-64.0)
    return xc, yc, xt, yt


def make_monash_dmodule(root_dir, max_len, include, batch_size, workers, overfit):
    dsets = load_monash_dsets(root_dir, max_len, include)
    module = stf.data.DataModule(
        MetaMonashDset,
        dataset_kwargs={"dsets": dsets},
        batch_size=batch_size,
        workers=workers,
        collate_fn=pad_collate,
        overfit=overfit,
    )
    return module


if __name__ == "__main__":
    DATA_PATH = "TODO"
    train = make_monash_dmodule(
        root_dir=DATA_PATH,
        max_len=1000,
        include=["m4"],
        batch_size=1,
        workers=0,
        overfit=False,
    )

    min_ = float("inf")
    for i, batch in enumerate(train.train_dataloader()):
        xc, yc, xt, yt = batch
        print(f"Train {i}")
        if yt.min() < min_:
            min_ = yt.min()
            print(min_)
            input()
