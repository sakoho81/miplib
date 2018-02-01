
from supertomo.data.core.dictionary import FixedDictionary


class FourierCorrelationDataCollection(object):
    """
    A container for the directional Fourier shell correlation data
    """
    def __init__(self):
        self._data = dict()

    def __setitem__(self, key, value):
        assert isinstance(key, int)
        assert isinstance(value, FourierCorrelationData)

        self._data[str(key)] = value

    def __getitem__(self, item):
        return self._data[item]

    @property
    def contents(self):
        return self._data.keys(), self._data.values()


class FourierCorrelationData(object):
    """
    A datatype for FRC data
    """
    def __init__(self):

        correlation_keys = "correlation frequency points-x-bin curve-fit curve-fit-eq"
        resolution_keys = "threshold criterion resolution-point threshold-eq resolution"

        self.resolution = FixedDictionary(resolution_keys.split())
        self.correlation = FixedDictionary(correlation_keys.split())

