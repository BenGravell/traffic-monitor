from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import warnings

import pandas as pd


class ParquetAble(ABC):
    """Class for objects writable as parquet."""

    @abstractmethod
    def to_parquet(self, path: Path | str) -> None:
        """Write to parquet file at given path."""


class DataFrameAble(ParquetAble):
    """Class for objects convertible to a pandas dataframe."""

    @abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""

    def to_parquet(self, path: Path | str) -> None:
        self.to_dataframe().to_parquet(path)


class DataLogger(ABC):
    """Base class for data loggers."""

    @abstractmethod
    def store(self, data: Any, idx: int) -> None:
        """Store a piece of data associated to the given index number."""

    @abstractmethod
    def dump(self, path: Path | str) -> None:
        """Dump the currently stored log data to the given path."""


class NullDataLogger(DataLogger):
    """Logger that does not log any data."""

    def store(self, data: Any, idx: int) -> None:
        pass

    def dump(self, path: Path | str) -> None:
        pass


class ParquetDataLogger(DataLogger):
    """Logger to log parquet-able data."""

    def __init__(self) -> None:
        self.dataframes: dict[int, pd.DataFrame] = {}

    def store(self, data: DataFrameAble, idx: int) -> None:
        self.dataframes[idx] = data.to_dataframe()

    def dump(self, path: Path | str) -> None:
        idxs = sorted(self.dataframes.keys())
        # Suppress the FutureWarning from pd.concat().
        # See https://github.com/pandas-dev/pandas/issues/39122
        # and https://github.com/pandas-dev/pandas/issues/58304
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            cat_df = pd.concat([self.dataframes[idx] for idx in idxs], keys=idxs)
        cat_df = cat_df.convert_dtypes()
        cat_df.index = cat_df.index.set_names(["outer_idx", "inner_idx"])
        cat_df.to_parquet(path)
