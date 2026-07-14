import numpy as np

import miplib.processing.converters as converters
import miplib.processing.itk as itkutils
import miplib.processing.ndarray as nputils
from miplib.data.coordinates.polar import SimplePolarIndexer


class FourierShellIterator:
    """A 3D Fourier shell iterator.

    Computes a spherical coordinate grid centered at the geometric center
    and iterates over concentric shells of thickness *d_bin*.
    """

    def __init__(self, shape: tuple[int, int, int], d_bin: int | float) -> None:
        if len(shape) != 3:
            raise ValueError(f"shape must be 3D, got shape {shape}")

        self.d_bin = d_bin

        indexer = SimplePolarIndexer(shape)
        self.r = indexer.r
        z, y, x = indexer.meshgrid
        self.meshgrid = (z, y, x)

        self.shell_stop = int(np.floor(shape[0] / (2 * self.d_bin))) - 1
        self.current_shell = 0

        self.freq_nyq = int(np.floor(shape[0] / 2.0))
        self.radii = np.arange(0, self.freq_nyq, self.d_bin)

    @property
    def steps(self) -> np.ndarray:
        return self.radii

    @property
    def nyquist(self) -> int:
        return self.freq_nyq

    def get_points_on_shell(self, shell_start: float, shell_stop: float) -> np.ndarray:
        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop
        return arr_inf * arr_sup

    def __getitem__(self, limits: tuple[float, float]) -> tuple[np.ndarray, ...]:
        """Return point indices for a shell defined by *(shell_start, shell_stop)*."""
        (shell_start, shell_stop) = limits
        shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(shell)

    def __iter__(self) -> "FourierShellIterator":
        return self

    def __next__(self) -> tuple[tuple[np.ndarray, ...], int]:
        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
        else:
            raise StopIteration

        self.current_shell += 1
        return np.where(shell), shell_idx


class SectionedFourierShellIterator(FourierShellIterator):
    """A sectioned Fourier shell iterator.

    Divides each shell into angular sectors of width *d_angle* (degrees),
    iterating over all (shell, rotation) pairs. Iteration order is shell-outer:
    for each shell, iterate all rotations, then advance shell.
    """

    def __init__(
        self, shape: tuple[int, int, int], d_bin: int | float, d_angle: float
    ) -> None:
        FourierShellIterator.__init__(self, shape, d_bin)

        self.d_angle = converters.degrees_to_radians(d_angle)

        z, y, x = self.meshgrid
        self.phi = np.arctan2(y, z) + np.pi
        self.phi += self.d_angle / 2
        self.phi[self.phi >= 2 * np.pi] -= 2 * np.pi

        self.rotation_start = 0
        self.rotation_stop = int(360 / d_angle) - 1
        self.current_rotation = self.rotation_start

        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        return self.radii, self.angles

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        """Return a boolean mask for the azimuthal sector [phi_min, phi_max).

        The azimuth is the angle in the YZ plane (rotation around X-axis).
        Includes the 180-degree counterpart for Fourier-space symmetry.
        """
        arr_inf = self.phi >= phi_min
        arr_sup = self.phi < phi_max

        arr_inf_neg = self.phi >= phi_min + np.pi
        arr_sup_neg = self.phi < phi_max + np.pi

        return arr_inf * arr_sup + arr_inf_neg * arr_sup_neg

    def __getitem__(
        self,
        limits: tuple[float, float, float, float],  # type: ignore[override]
    ) -> tuple[np.ndarray, ...]:
        (shell_start, shell_stop, angle_min, angle_max) = limits
        angle_min = converters.degrees_to_radians(angle_min)
        angle_max = converters.degrees_to_radians(angle_max)

        shell = self.get_points_on_shell(shell_start, shell_stop)
        cone = self.get_angle_sector(angle_min, angle_max)

        return np.where(shell * cone)

    def __next__(self) -> tuple[tuple[np.ndarray, ...], int, int]:  # type: ignore[override]
        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if rotation_idx <= self.rotation_stop and shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
            cone = self.get_angle_sector(
                self.current_rotation * self.d_angle,
                (self.current_rotation + 1) * self.d_angle,
            )
        else:
            raise StopIteration

        if rotation_idx >= self.rotation_stop:
            self.current_rotation = 0
            self.current_shell += 1
        else:
            self.current_rotation += 1

        return np.where(shell * cone), shell_idx, rotation_idx


class HollowSectionedFourierShellIterator(SectionedFourierShellIterator):
    """A sectioned iterator that hollows out the center of each angular sector.

    Removes a central slice of width *d_extract_angle* (degrees) to reduce
    interpolation artifacts from the lowest-resolution axis.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        d_bin: int | float,
        d_angle: float,
        d_extract_angle: float = 5,
    ) -> None:
        SectionedFourierShellIterator.__init__(self, shape, d_bin, d_angle)

        self.d_extract_angle = converters.degrees_to_radians(d_extract_angle)

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        full_section = SectionedFourierShellIterator.get_angle_sector(
            self, phi_min, phi_max
        )

        sector_center = phi_min + (phi_max - phi_min) / 2
        phi_min_ext = sector_center - self.d_extract_angle
        phi_max_ext = sector_center + self.d_extract_angle

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return full_section & ~extract_section


class AxialExcludeSectionedFourierShellIterator(HollowSectionedFourierShellIterator):
    """A hollow iterator that only removes the center slice for axial sectors.

    The axial directions (90° and 270°, perpendicular to the optical axis Z)
    are the ones affected by low-resolution interpolation artifacts.
    Non-axial sectors are left intact.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        d_bin: int | float,
        d_angle: float,
        d_extract_angle: float = 5,
    ) -> None:
        HollowSectionedFourierShellIterator.__init__(
            self, shape, d_bin, d_angle, d_extract_angle
        )

    def get_angle_sector(self, phi_min: float, phi_max: float) -> np.ndarray:
        full_section = SectionedFourierShellIterator.get_angle_sector(
            self, phi_min, phi_max
        )

        axis_pos = converters.degrees_to_radians(90) + self.d_angle / 2
        axis_neg = converters.degrees_to_radians(270) + self.d_angle / 2

        if phi_min <= axis_pos <= phi_max:
            phi_min_ext = axis_pos - self.d_extract_angle
            phi_max_ext = axis_pos + self.d_extract_angle
        elif phi_min <= axis_neg <= phi_max:
            phi_min_ext = axis_neg - self.d_extract_angle
            phi_max_ext = axis_neg + self.d_extract_angle
        else:
            return full_section

        arr_inf_ext = self.phi >= phi_min_ext
        arr_sup_ext = self.phi < phi_max_ext

        arr_inf_neg_ext = self.phi >= phi_min_ext + np.pi
        arr_sup_neg_ext = self.phi < phi_max_ext + np.pi

        extract_section = arr_inf_ext * arr_sup_ext + arr_inf_neg_ext * arr_sup_neg_ext

        return full_section & ~extract_section


class RotatingFourierShellIterator(FourierShellIterator):
    """A 3D iterator using a rotating-plane method.

    A single XY plane is extracted from the 3D volume and rotated through
    a set of angles. At each rotation, the plane is intersected with every
    Fourier shell. Based on Nieuwenhuizen et al. (2013).
    """

    def __init__(
        self, shape: tuple[int, int, int], d_bin: int | float, d_angle: float
    ) -> None:
        if len(shape) != 3:
            raise ValueError(f"shape must be 3D, got shape {shape}")

        FourierShellIterator.__init__(self, shape, d_bin)

        plane = nputils.expand_to_shape(np.ones((1, shape[1], shape[2])), shape)
        self.plane = itkutils.convert_from_numpy(plane, (1, 1, 1))
        self.rotated_plane = plane > 0

        self.rotation_start = 0
        self.rotation_stop = int(360 / d_angle) - 1
        self.current_rotation = self.rotation_start
        self.angles = np.arange(0, 360, d_angle, dtype=int)

    @property
    def steps(self) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[override]
        return self.radii, self.angles

    def __getitem__(
        self,
        limits: tuple[float, float, float],  # type: ignore[override]
    ) -> tuple[np.ndarray, ...]:
        (shell_start, shell_stop, angle) = limits

        rotated_plane = itkutils.convert_from_itk_image(
            itkutils.rotate_image(self.plane, angle)
        )
        points_on_plane = rotated_plane > 0
        points_on_shell = self.get_points_on_shell(shell_start, shell_stop)
        return np.where(points_on_plane * points_on_shell)

    def __next__(self) -> tuple[tuple[np.ndarray, ...], int, int]:  # type: ignore[override]
        rotation_idx = self.current_rotation
        shell_idx = self.current_shell

        if shell_idx <= self.shell_stop:
            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
            self.current_shell += 1
        elif rotation_idx <= self.rotation_stop:
            rotated_plane = itkutils.convert_from_itk_image(
                itkutils.rotate_image(
                    self.plane, self.angles[rotation_idx], interpolation="linear"
                )
            )
            self.rotated_plane = rotated_plane > 0
            self.current_shell = 0
            shell_idx = 0
            self.current_rotation += 1

            shell = self.get_points_on_shell(
                self.current_shell * self.d_bin, (self.current_shell + 1) * self.d_bin
            )
        else:
            raise StopIteration

        return np.where(shell * self.rotated_plane), shell_idx, self.current_rotation
