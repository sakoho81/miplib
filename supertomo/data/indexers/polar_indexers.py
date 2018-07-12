
import numpy as np


class SimplePolarIndexer(object):
    def __init__(self, shape):
        assert isinstance(shape, tuple) or \
               isinstance(shape, list) or \
               isinstance(shape, np.ndarray)
        assert 1 < len(shape) < 4

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)

        meshgrid = np.meshgrid(*axes)
        self.r = np.sqrt(sum(map(lambda axis: axis**2, meshgrid)))

    def __getitem__(self, item):
        return self.r == item


class PolarLowPassIndexer(SimplePolarIndexer):
    def __getitem__(self, item):
        return self.r < item


class PolarHighPassIndexer(SimplePolarIndexer):
    def __getitem__(self, item):
        return self.r > item



