# coding=utf-8
from math import floor

import numpy as np
import miplib.processing.converters as converters


class FourierRingIterator(object):
    """
    A Fourier ring iterator class for 2D images. Calculates a 2D polar coordinate
    centered at the geometric center of the data shape.
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
        self._nbins = int(floor(shape[0] / (2 * self.d_bin)))
        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        y, x = np.meshgrid(*axes)
        self.meshgrid = (y, x)

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

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(self.current_ring * self.d_bin,
                                           (self.current_ring + 1) * self.d_bin)
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring-1


class SectionedFourierRingIterator(FourierRingIterator):
    """
    An iterator for 2D images. Includes the option use only a specific rotated section of
    the fourier ring for FRC calculation.
    """
    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        FourierRingIterator.__init__(self, shape, d_bin)

        self.d_angle = converters.degrees_to_radians(d_angle)

        y, x = self.meshgrid

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, x) + np.pi

        self.phi += self.d_angle/2
        self.phi[self.phi >= 2*np.pi] -= 2*np.pi

        self._angle = 0
        
        self.angle_sector = self.get_angle_sector(0, d_bin)

    @property
    def angle(self):
        return self._angle
    
    @angle.setter
    def angle(self, value):
        angle = converters.degrees_to_radians(value)
        self._angle = angle
        self.angle_sector = self.get_angle_sector(angle, angle + self.d_angle)
        
    def get_angle_sector(self, phi_min, phi_max):
        """
        Use this to extract
        a section from a sphere that is defined by start and stop angles.

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __getitem__(self, limits):
        """
        Get a single conical section of a 2D ring.

        :param limits:  a list of parameters (ring_start, ring_stop, angle_min, angle_ma)
        that are required to define a single section of a fourier ring.
         """
        (ring_start, ring_stop, angle_min, angle_max) = limits
        ring = self.get_points_on_ring(ring_start, ring_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(ring*cone)

    def __next__(self):
        if self.current_ring < self._nbins:
            ring = self.get_points_on_ring(self.current_ring * self.d_bin,
                                           (self.current_ring + 1) * self.d_bin)
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring*self.angle_sector), self.current_ring-1
