
from supertomo.data.core.dictionary import FixedDictionary


class FourierCorrelationDataCollection(object):
    """
    A container for the directional Fourier shell correlation data
    """
    def __init__(self):
        self._data = dict()

        self.iter_index = 0

    def __setitem__(self, key, value):
        assert isinstance(key, int)
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, key):
        return self._data[str(key)]

    def __iter__(self):
        return self

    def next(self):
        try:
            item = self._data.items()[self.iter_index]
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
        return self._data.items()


class FourierCorrelationData(object):
    """
    A datatype for FRC data
    """
    def __init__(self, data=None):

        correlation_keys = "correlation frequency points-x-bin curve-fit " \
                           "curve-fit-coefficients"
        resolution_keys = "threshold criterion resolution-point " \
                          "resolution-threshold-coefficients resolution"

        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

        if data is not None:
            assert isinstance(data, dict)

            for key, value in data.iteritems():
                if key in self.resolution.keys:
                    self.resolution[key] = value
                elif key in self.correlation.keys:
                    self.correlation[key] = value
                else:
                    raise ValueError("Unknown key found in the initialization data")



