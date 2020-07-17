# coding=utf-8
from math import floor

import numpy as np

import miplib.processing.converters as converters
import miplib.processing.ndarray as nputils
import miplib.processing.itk as itkutils


class FourierShellIterator(object):
    """
    A Simple Fourier Shell Iterator. Basically the same as a Fourier Ring Iterator,
    but for 3D.
    """

    def __init__(self, shape, d_bin):
        self.d_bin = d_bin

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        z, y, x = np.meshgrid(*axes)
        self.meshgrid = (z, y, x)

        # Create OP vector array
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        self.shell_start = 0
        self.shell_stop = int(floor(shape[0] / (2 * self.d_bin))) - 1

        self.current_shell = self.shell_start

        self.freq_nyq = int(np.floor(shape[0] / 2.0))

        self.radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def steps(self):
        return self.radii

    @property
    def nyquist(self):
        return self.freq_nyq

    def get_points_on_shell(self, shell_start, shell_stop):

        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop

        return arr_inf*arr_sup

    def __getitem__(self, limits):
        """
        Get a points on a Fourier shell specified by the start and stop coordinates

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell

        :return:            Returns the coordinates of the points that are located on
                            the specified shell
        """
        (shell_start, shell_stop) = limits
        shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(shell)

    def __iter__(self):
        return self

    def __next__(self):

        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(self.current_shell * self.d_bin,
                                             (self.current_shell + 1) * self.d_bin)
        else:
            raise StopIteration

        self.current_shell += 1

        return np.where(shell), shell_idx


class ConicalFourierShellIterator(FourierShellIterator):
    """
    An iterator for 3D images. Includes the option section a single shell into rotational
    sections.
    """
    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        FourierShellIterator.__init__(self, shape, d_bin)

        self.d_angle = converters.degrees_to_radians(d_angle)

        z, y, x = self.meshgrid

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, z) + np.pi

        self.phi += self.d_angle/2
        self.phi[self.phi >= 2*np.pi] -= 2*np.pi

        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start

        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        return self.radii, self.angles

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

    def __getitem__(self, limits):
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
        (shell_start, shell_stop, angle_min, angle_max) = limits
        angle_min = converters.degrees_to_radians(angle_min)
        angle_max = converters.degrees_to_radians(angle_max)

        shell = self.get_points_on_shell(shell_start, shell_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(shell*cone)

    def __next__(self):

        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(self.current_shell * self.d_bin,
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


class HollowConicalFourierShellIterator(ConicalFourierShellIterator):
    """
    A conical Fourier shell iterator with the added possibility to remove
    a central section of the cone, to better deal with interpolation artefacts etc.
    """

    def __init__(self,  shape, d_bin, d_angle, d_extract_angle=5):

        ConicalFourierShellIterator.__init__(self, shape, d_bin, d_angle)

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

        return np.logical_xor(full_section, extract_section)


class AxialExcludeHollowConicalFourierShellIterator(HollowConicalFourierShellIterator):
    """
    A conical Fourier shell iterator with the added possibility to remove
    a central section of the cone, to better deal with interpolation artefacts etc.
    """

    def __init__(self,  shape, d_bin, d_angle, d_extract_angle=5):

        HollowConicalFourierShellIterator.__init__(self, shape, d_bin, d_angle)


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

        axis_pos = converters.degrees_to_radians(90) + self.d_angle/2
        axis_neg = converters.degrees_to_radians(270) + self.d_angle/2

        if phi_min <= axis_pos <= phi_max:
            phi_min_ext = axis_pos - self.d_extract_angle
            phi_max_ext = axis_pos + self.d_extract_angle

        elif phi_min <= axis_neg <= phi_max:

            # Calculate part of the section to exclude
            phi_min_ext = axis_neg - self.d_extract_angle
            phi_max_ext = axis_neg + self.d_extract_angle

        else:
            return full_section

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return np.logical_xor(full_section, extract_section)


class RotatingFourierShellIterator(FourierShellIterator):
    """
    A 3D Fourier Ring Iterator -- not a Fourier Shell Iterator, but rather
    single planes are extracted from a 3D shape by rotating the XY plane,
    as in:

    Nieuwenhuizen, Rpj, K. A. Lidke, and Mark Bates. 2013.
    “Measuring Image Resolution in Optical Nanoscopy.” Nature
    advance on (April). https://doi.org/10.1038/nmeth.2448.

    Here the shell iteration is still in 3D (for compatilibility with the others
    which doesn't make much sense in terms of calculation effort, but it should make
    it possible to

    """

    def __init__(self, shape, d_bin, d_angle):
        """
        :param shape: Shape of the data
        :param d_bin: The radius increment size (pixels)
        :param d_angle: The angle increment size (degrees)
        """

        assert len(shape) == 3, "This iterator assumes a 3D shape"

        FourierShellIterator.__init__(self, shape, d_bin)

        plane = nputils.expand_to_shape(np.ones((1, shape[1], shape[2])), shape)

        self.plane = itkutils.convert_from_numpy(
            plane,
            (1, 1, 1))

        self.rotated_plane = plane > 0

        self.rotation_start = 0
        self.rotation_stop = 360 / d_angle - 1
        self.current_rotation = self.rotation_start

        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self):
        return self.radii, self.angles

    def __getitem__(self, limits):
        """
        Get a single conical section of a 3D shell.

        :param shell_start: The start of the shell (0 ... Nyquist)
        :param shell_stop:  The end of the shell
        :param angle:
        """
        (shell_start, shell_stop, angle) = limits
        rotated_plane = itkutils.convert_from_itk_image(
            itkutils.rotate_image(self.plane, angle))

        points_on_plane = rotated_plane > 0
        points_on_shell = self.get_points_on_shell(shell_start, shell_stop)

        return np.where(points_on_plane * points_on_shell)

    def __next__(self):

        rotation_idx = self.current_rotation + 1
        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(self.current_shell * self.d_bin,
                                             (self.current_shell + 1) * self.d_bin)
            self.current_shell += 1

        elif rotation_idx <= self.rotation_stop:

            rotated_plane = itkutils.convert_from_itk_image(
                itkutils.rotate_image(self.plane, self.angles[rotation_idx],
                                      interpolation='linear'))

            self.rotated_plane = rotated_plane > 0
            self.current_shell = 0
            shell_idx = 0
            self.current_rotation += 1

            shell = self.get_points_on_shell(self.current_shell * self.d_bin,
                                             (self.current_shell + 1) * self.d_bin)

        else:
            raise StopIteration

        return np.where(shell * self.rotated_plane), shell_idx, self.current_rotation


