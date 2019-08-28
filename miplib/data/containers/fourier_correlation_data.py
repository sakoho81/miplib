
from miplib.data.core.dictionary import FixedDictionary

import pandas as pd
import numpy as np


class FourierCorrelationDataCollection(object):
    """
    A container for the directional Fourier correlation data
    """
    def __init__(self):
        self._data = dict()

        self.iter_index = 0

    def __setitem__(self, key, value):
        assert isinstance(key, (int, np.integer))
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, key):
        return self._data[str(key)]

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = list(self._data.items())[self.iter_index]
        except IndexError:
            self.iter_index = 0
            raise StopIteration

        self.iter_index += 1
        return item

    def __len__(self):
        return len(self._data)

    def clear(self):
        self._data.clear()

    def items(self):
        return list(self._data.items())

    def nitems(self):
        return len(self._data)

    def as_dataframe(self, include_results=False):
        """
        Convert a FourierCorrelationDataCollection object into a Pandas
        dataframe. Only returns the raw Fourier correlation data,
        not the processed results.

        :return: A dataframe with columns: Angle (categorical), Correlation (Y),
                 Frequency (X) and nPoints (number of points in each bin)
        """
        df = pd.DataFrame(columns=['Correlation', 'Frequency', 'nPoints', 'Angle'])

        for key, dataset in self._data.items():
            df_temp = dataset.as_dataframe(include_results=include_results)

            angle = np.full(len(df_temp), int(key), dtype=np.int64)
            df_temp['Angle'] = angle

            df = pd.concat([df, df_temp], ignore_index=True)

        df['Angle'] = df['Angle'].astype('category')
        return df


class FourierCorrelationData(object):
    """
    A datatype for FRC data

    """
    #todo: the dictionary format here is a bit clumsy. Maybe change to a simpler structure

    def __init__(self, data=None):

        correlation_keys = "correlation frequency points-x-bin curve-fit " \
                           "curve-fit-coefficients"
        resolution_keys = "threshold criterion resolution-point " \
                          "resolution-threshold-coefficients resolution spacing"

        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

        if data is not None:
            assert isinstance(data, dict)

            for key, value in data.items():
                if key in self.resolution.keys:
                    self.resolution[key] = value
                elif key in self.correlation.keys:
                    self.correlation[key] = value
                else:
                    raise ValueError("Unknown key found in the initialization data")

    def as_dataframe(self, include_results=False):
        """
        Convert a FourierCorrelationData object into a Pandas
        dataframe. Only returns the raw Fourier correlation data,
        not the processed results.

        :return: A dataframe with columns: Correlation (Y), Frequency (X) and
                 nPoints (number of points in each bin)
        """
        if include_results is False:
            to_df = {
                'Correlation': self.correlation["correlation"],
                'Frequency': self.correlation["frequency"],
                'nPoints': self.correlation["points-x-bin"],
            }
        else:
            resolution = np.full(self.correlation["correlation"].shape,
                                 self.resolution["resolution"],
                                 dtype=np.float32)
            resolution_point_x = np.full(self.correlation["correlation"].shape,
                                         self.resolution["resolution-point"][0],
                                         dtype=np.float32)
            resolution_point_y = np.full(self.correlation["correlation"].shape,
                                         self.resolution["resolution-point"][1],
                                         dtype=np.float32)
            threshold = self.resolution["threshold"],

            to_df = {
                'Correlation': self.correlation["correlation"],
                'Frequency': self.correlation["frequency"],
                'nPoints': self.correlation["points-x-bin"],
                'Resolution': resolution,
                'Resolution_X': resolution_point_x,
                'Resolution_Y': resolution_point_y,
                'Threshold': threshold

            }

        return pd.DataFrame(to_df)
