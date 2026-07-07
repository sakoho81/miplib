import numpy as np
import pandas as pd

from miplib.data.core.dictionary import FixedDictionary


class _FourierCorrelationDataCollectionIterator:
    """Independent iterator for FourierCorrelationDataCollection."""

    def __init__(self, data: "FourierCorrelationDataCollection"):
        self._items = list(data._data.items())
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._items):
            raise StopIteration
        item = self._items[self._index]
        self._index += 1
        return item


class FourierCorrelationDataCollection:
    """Container for directional Fourier correlation data keyed by angle."""

    def __init__(self) -> None:
        self._data: dict[str, FourierCorrelationData] = {}

    def __setitem__(self, key: int, value: "FourierCorrelationData") -> None:
        if not isinstance(key, (int, np.integer)):
            raise TypeError(f"Key must be an integer, got {type(key).__name__}")
        if not isinstance(value, FourierCorrelationData):
            raise TypeError(
                f"Value must be FourierCorrelationData, got {type(value).__name__}"
            )
        self._data[str(key)] = value

    def __getitem__(self, key: int) -> "FourierCorrelationData":
        return self._data[str(key)]

    def __iter__(self):
        return _FourierCorrelationDataCollectionIterator(self)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def items(self) -> list[tuple[str, "FourierCorrelationData"]]:
        return list(self._data.items())

    def nitems(self) -> int:
        return len(self._data)

    def as_dataframe(self, include_results: bool = False) -> pd.DataFrame:
        """Return all correlation data as a Pandas DataFrame."""
        df = pd.DataFrame(columns=["Correlation", "Frequency", "nPoints", "Angle"])
        for key, dataset in self._data.items():
            df_temp = dataset.as_dataframe(include_results=include_results)
            angle = np.full(len(df_temp), int(key), dtype=np.int64)
            df_temp["Angle"] = angle
            df = pd.concat([df, df_temp], ignore_index=True)
        df["Angle"] = df["Angle"].astype("category")
        return df


class FourierCorrelationData:
    """Single FRC/FSC result with correlation and resolution metadata."""

    def __init__(self, data: dict | None = None) -> None:
        correlation_keys = (
            "correlation frequency points-x-bin curve-fit curve-fit-coefficients"
        )
        resolution_keys = (
            "threshold criterion resolution-point "
            "resolution-threshold-coefficients resolution spacing"
        )
        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

        if data is not None:
            if not isinstance(data, dict):
                raise TypeError(
                    f"data must be a dict or None, got {type(data).__name__}"
                )
            for key, value in data.items():
                if key in self.resolution.keys:
                    self.resolution[key] = value
                elif key in self.correlation.keys:
                    self.correlation[key] = value
                else:
                    raise ValueError(f"Unknown key in data: {key!r}")

    def as_dataframe(self, include_results: bool = False) -> pd.DataFrame:
        """Return correlation data as a DataFrame."""
        if include_results is False:
            to_df: dict = {
                "Correlation": self.correlation["correlation"],
                "Frequency": self.correlation["frequency"],
                "nPoints": self.correlation["points-x-bin"],
            }
        else:
            resolution = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution"],
                dtype=np.float32,
            )
            resolution_point_x = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution-point"][0],
                dtype=np.float32,
            )
            resolution_point_y = np.full(
                self.correlation["correlation"].shape,
                self.resolution["resolution-point"][1],
                dtype=np.float32,
            )
            threshold = (self.resolution["threshold"],)
            to_df = {
                "Correlation": self.correlation["correlation"],
                "Frequency": self.correlation["frequency"],
                "nPoints": self.correlation["points-x-bin"],
                "Resolution": resolution,
                "Resolution_X": resolution_point_x,
                "Resolution_Y": resolution_point_y,
                "Threshold": threshold,
            }
        return pd.DataFrame(to_df)
