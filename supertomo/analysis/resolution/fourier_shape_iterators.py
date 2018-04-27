from math import floor

import numpy as np

import supertomo.processing.converters as converters


class FourierRingIterator(object):
    def __init__(self, shape, d_bin):

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


class FourierShellIterator(object):
    """
    An iterator for 3D images. Includes the option section a single shell into rotational
    sections.
    """
    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the image
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        self.d_angle = converters.degrees_to_radians(d_angle)
        self.d_bin = d_bin

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        z, y, x = np.meshgrid(*axes)

        # Create OP vector array
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, z) + np.pi

        self.phi += self.d_angle/2
        self.phi[self.phi >= 2*np.pi] -= 2*np.pi

        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1

        self.shell_start = 0
        self.shell_stop = floor(shape[0] / (2 * self.d_bin)) - 1

        self.current_rotation = self.rotation_start
        self.current_shell = self.shell_start

        self.freq_nyq = int(np.floor(shape[0] / 2.0))

        self.angles = np.arange(0, 360, d_angle, dtype=int)
        self.radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def steps(self):
        return self.radii, self.angles

    @property
    def nyquist(self):
        return self.freq_nyq

    def get_angle_sector(self, phi_min, phi_max):
        """
        Assuming a classical spherical coordinate system the azimutahl
        angle is the angle between the x- and y- axes. Use this to extract
        a conical section from a sphere that is defined by start and stop azimuth
        angles.

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __get_points_on_shell(self, shell_start, shell_stop):

        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop

        return arr_inf*arr_sup

    def __getitem__(self, (shell_start, shell_stop, angle_min, angle_max)):
        """
        Get a single conical section of a 3D shell.

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell
        :param angle_min:   The start of the cone (degrees 0-360)
        :param angle_max:   The end of the cone
        :return:            Returns the coordinates of the points that are located inside
                            the portion of a shell that intersects with the points on the
                            cone.
        """

        angle_min = converters.degrees_to_radians(angle_min)
        angle_max = converters.degrees_to_radians(angle_max)

        shell = self.__get_points_on_shell(shell_start, shell_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(shell*cone)

    def __iter__(self):
        return self

    def next(self):

        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.__get_points_on_shell(self.current_shell * self.d_bin,
                                              (self.current_shell + 1) * self.d_bin)

            cone = self.get_angle_sector(self.current_rotation * self.d_angle,
                                          (self.current_rotation + 1) * self.d_angle)
        else:
            raise StopIteration

        if rotation_idx >= self.rotation_stop:
            self.current_rotation = 0
            self.current_shell += 1
        else:
            self.current_rotation += 1

        return np.where(shell*cone), shell_idx, rotation_idx


class HollowFourierShellIterator(FourierShellIterator):

    def __init__(self,  shape, d_bin, d_angle, d_extract_angle=5):

        FourierShellIterator.__init__(self, shape, d_bin, d_angle)

        self.d_extract_angle = converters.degrees_to_radians(d_extract_angle)

    def get_angle_sector(self, phi_min, phi_max):
        """
        Assuming a classical spherical coordinate system the azimutahl
        angle is the angle between the x- and y- axes. Use this to extract
        a conical section from a sphere that is defined by start and stop azimuth
        angles.

        In the hollow implementation a small slice in the center of the section is
        removed to avoid the effect of resampling when calculating the resolution
        along the lowest resolution axis (z), on images with very isotropic resolution
        (e.g. STED).

        :param phi_min: the angle at which to start the section, in radians
        :param phi_max: the angle at which to stop the section, in radians
        :return:

        """
        # Calculate angular sector
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        full_section = arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

        # Calculate part of the section to exclude
        sector_center = phi_min + (phi_max-phi_min)/2
        phi_min_ext = sector_center - self.d_extract_angle
        phi_max_ext = sector_center + self.d_extract_angle

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return full_section - extract_section
