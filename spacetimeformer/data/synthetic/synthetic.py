import os
import pandas as pd
import numpy as np

from spacetimeformer.data import CSVTimeSeries


class SyntheticData(CSVTimeSeries):
    def __init__(self, data_path: str):
        super().__init__(
            data_path=data_path,
            target_cols=[f"y{i}" for i in range(20)],
            ignore_cols=[],
            remove_target_from_context_cols=[],
            time_col_name="Datetime",
            time_features=["year", "month", "day", "hour", "minute"],
        )
