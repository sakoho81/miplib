import numpy as np

import miplib.processing.converters as converters
from miplib.data.coordinates.polar import SimplePolarIndexer


class FourierRingIterator:
    """A Fourier ring iterator for 2D images.

    Computes a polar coordinate grid centered at the geometric center
    and iterates over concentric rings of thickness *d_bin*.
    """

    def __init__(self, shape, d_bin):
        if len(shape) != 2:
            raise ValueError(f"shape must be 2D, got shape {shape}")

        self.d_bin = d_bin
        self._nbins = int(np.floor(shape[0] / (2 * self.d_bin)))

        indexer = SimplePolarIndexer(shape)
        self.r = indexer.r
        y, x = indexer.meshgrid
        self.meshgrid = (y, x)

        self.current_ring = 0
        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        self._radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def radii(self):
        return self._radii

    @property
    def nbins(self):
        return self._nbins

    def get_points_on_ring(self, ring_start, ring_stop):
        arr_inf = self.r >= ring_start
        arr_sup = self.r < ring_stop
        return arr_inf * arr_sup

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring - 1


class SectionedFourierRingIterator(FourierRingIterator):
    """A Fourier ring iterator with angular sectoring.

    Each ring is restricted to an angular cone of width *d_angle* (degrees),
    centered at a settable rotation angle.
    """

    def __init__(self, shape, d_bin, d_angle):
        FourierRingIterator.__init__(self, shape, d_bin)

        self.d_angle = converters.degrees_to_radians(d_angle)

        y, x = self.meshgrid
        self.phi = np.arctan2(y, x) + np.pi
        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi

        self._angle = 0
        self.angle_sector = self.get_angle_sector(0, self.d_angle)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        angle = converters.degrees_to_radians(value)
        self._angle = angle
        self.angle_sector = self.get_angle_sector(angle, angle + self.d_angle)

    def get_angle_sector(self, phi_min, phi_max):
        """Return a boolean mask for the angular sector [phi_min, phi_max).

        Includes the 180-degree counterpart for Fourier-space symmetry.
        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __getitem__(self, limits):
        (ring_start, ring_stop, angle_min, angle_max) = limits
        ring = self.get_points_on_ring(ring_start, ring_stop)
        cone = self.get_angle_sector(angle_min, angle_max)
        return np.where(ring * cone)

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(
                self.current_ring * self.d_bin, (self.current_ring + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring * self.angle_sector), self.current_ring - 1
