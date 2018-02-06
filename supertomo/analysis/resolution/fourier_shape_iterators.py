import numpy as np
import supertomo.processing.ndarray as ops_array
from math import floor
from abc import ABCMeta, abstractmethod


class IteratorBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, shape, d_bin, **kwargs):
        pass
    @abstractmethod
    def __iter__(self):
        pass
    @abstractmethod
    def next(self):
        pass


class FourierRingIterator(IteratorBase):
    def __init__(self, shape, d_bin, **kwargs):

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

    @property
    def nbins(self): return self._nbins

    def __get_points_on_ring(self, ring_start, ring_stop):

        arr_inf = self.r >= ring_start
        arr_sup = self.r < ring_stop

        return arr_inf*arr_sup

    def __iter__(self):
        return self

    def next(self):
        if self.current_ring < self._nbins:
            ring = self.__get_points_on_ring(self.current_ring * self.d_bin,
                                             (self.current_ring + 1) * self.d_bin)
        else:
            raise StopIteration

        self.current_ring += 1
        return np.where(ring), self.current_ring-1


class FourierShellIterator(IteratorBase):

    def __init__(self, shape, d_bin, **kwargs):
        """
        Generate a coordinate system for spherical indexing of 3D Fourier domain
        images and iterate through it.

        :param shape: Shape of the image
        :param kwargs: Should contain at least the terms "d_theta" and
                      "d_bin", for angluar and radial bin sizes.

        """
        # Check that all the necessary inputs are present
        if "d_theta" in kwargs:
            self.d_rotation = kwargs["d_theta"]
            self.rotation_axis = "theta"

        elif "d_phi" in kwargs:
            self.d_rotation = kwargs["d_phi"]
            self.rotation_axis = "phi"
        else:
            raise ValueError("You must specify a value for rotation angle increments")

        self.d_bin = d_bin

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        z, x, y = np.meshgrid(*axes)

        # Create OP vector array
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, x)
        self.theta = np.arccos(ops_array.safe_divide(z, self.r))

        self.rotation_start = 0
        self.rotation_stop = 360 / self.d_rotation - 1

        self.shell_start = 0
        self.shell_stop = floor(shape[0] / (2 * self.d_bin)) - 1

        self.current_rotation = self.rotation_start
        self.current_shell = self.shell_start

    def __get_inclination_sector(self, theta_min, theta_max):

        arr_inf = self.theta >= theta_min
        arr_sup = self.theta < theta_max

        return arr_inf*arr_sup

    def __get_azimuthal_sector(self, phi_min, phi_max):
        arr_inf = self.phi >= phi_min
        arr_sup = self.theta < phi_max

        return arr_inf * arr_sup

    def __get_points_on_shell(self, shell_start, shell_stop):

        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop

        return arr_inf*arr_sup

    def __iter__(self):
        return self

    def next(self):
        try:
            shell = self.__get_points_on_shell(self.current_shell * self.d_bin,
                                               (self.current_shell + 1) * self.d_bin)
            if self.rotation_axis == "theta":
                cone = self.__get_inclination_sector(self.current_rotation * self.d_rotation,
                                                 (self.current_rotation + 1) * self.d_rotation)
            else:
                cone = self.__get_azimuthal_sector(self.current_rotation * self.d_rotation,
                                                     (self.current_rotation + 1) * self.d_rotation)
        except IndexError:
            raise StopIteration

        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if self.current_rotation >= self.rotation_stop:
            self.current_rotation = 0
            self.current_shell += 1
        else:
            self.current_rotation += 1

        return np.where(shell*cone), shell_idx, rotation_idx

