# coding=utf-8
from math import floor

import numpy as np

class FourierRingIterator(object):
    """
    A Fourier ring iterator class for 2D images. Calculates a 2D polar coordinate
    centered at the geometric center of the image shape.
    """
    def __init__(self, shape, d_bin):
        """
        :param shape: the volume shape
        :param d_bin: thickness of the ring in pixels
        """

        assert len(shape) == 2

        # Get bin size
        self.d_bin = d_bin
        self.ring_start = 0
        self._nbins = floor(shape[0] / (2 * self.d_bin))
        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        x, y = np.meshgrid(*axes)

        # Create OP vector array
        self.r = np.sqrt(x ** 2 + y ** 2)
        # Current ring index
        self.current_ring = self.ring_start

        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        self._radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def radii(self): return self._radii

    @property
    def nbins(self): return self._nbins

    def get_points_on_ring(self, ring_start, ring_stop):

        arr_inf = self.r >= ring_start
        arr_sup = self.r < ring_stop

        return arr_inf*arr_sup

    def __iter__(self):
        return self

    def next(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(self.current_ring * self.d_bin,
                                           (self.current_ring + 1) * self.d_bin)
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring-1


