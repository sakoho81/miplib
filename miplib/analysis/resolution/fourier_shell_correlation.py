"""
Sami Koho 01/2018

Sectioned Fourier Shell Correlation for complex resolution analysis
in 3D images.
"""

import numpy as np
from scipy.ndimage.interpolation import rotate

import miplib.data.containers.fourier_correlation_data as containers
import miplib.data.iterators.fourier_shell_iterators as iterators
import miplib.processing.image as imops
import miplib.processing.ndarray as ndarray
from miplib.data.containers.image import Image
from miplib.processing import fftutils, windowing

from . import analysis as fsc_analysis
from .fourier_ring_correlation import FRCOptions, _cutoff_correction


def calculate_fourier_plane_correlation(
    image1: Image,
    image2: Image,
    options: FRCOptions | None = None,
    *,
    z_correction: float = 1.0,
):
    if options is None:
        options = FRCOptions()
    steps = np.arange(0, 360, options.d_angle)
    data = containers.FourierCorrelationDataCollection()

    for _idx, step in enumerate(steps):
        im1_rot = fftutils.fft(rotate(image1, step, reshape=False), window=None)
        im2_rot = fftutils.fft(rotate(image2, step, reshape=False), window=None)

        numerator = np.sum(im1_rot * np.conjugate(im2_rot), axis=(0, 2))
        denominator = np.sum(
            np.sqrt(np.abs(im1_rot) ** 2 * np.abs(im2_rot) ** 2), axis=(0, 2)
        )

        correlation = ndarray.safe_divide(numerator, denominator)

        zero = correlation.size / 2
        correlation = correlation[zero:]

        result = containers.FourierCorrelationData()
        result.correlation["correlation"] = correlation
        result.correlation["frequency"] = np.linspace(0, 1.0, num=correlation.size)
        result.correlation["points-x-bin"] = np.ones(correlation.size) * (
            im2_rot.shape[2] * im2_rot.shape[0]
        )

        data[int(step)] = result

    analyzer = fsc_analysis.FourierCorrelationAnalysis(data, image1.spacing[0], options)
    return analyzer.execute(z_correction=z_correction)


def calculate_one_image_sectioned_fsc(
    image: Image,
    options: FRCOptions | None = None,
    *,
    z_correction: float = 1.0,
):
    if options is None:
        options = FRCOptions()

    image1, image2 = imops.checkerboard_split(image)

    image1 = Image(windowing.apply_hamming_window(image1), image1.spacing)
    image2 = Image(windowing.apply_hamming_window(image2), image2.spacing)

    iterator = iterators.AxialExcludeSectionedFourierShellIterator(
        image1.shape, options.d_bin, options.d_angle, options.d_extract_angle
    )
    fsc_task = DirectionalFSC(image1, image2, iterator)

    data = fsc_task.execute()

    analyzer = fsc_analysis.FourierCorrelationAnalysis(data, image1.spacing[0], options)
    result = analyzer.execute(z_correction=z_correction)

    for _angle, dataset in result:
        point_data = dataset.resolution["resolution-point"]
        if point_data is None:
            continue

        cut_off_correction = _cutoff_correction(point_data[1])
        dataset.resolution["spacing"] /= cut_off_correction
        dataset.resolution["resolution"] /= cut_off_correction

    return result


def calculate_two_image_sectioned_fsc(
    image1: Image,
    image2: Image,
    options: FRCOptions | None = None,
    *,
    z_correction: float = 1.0,
):
    if options is None:
        options = FRCOptions()

    image1 = Image(windowing.apply_hamming_window(image1), image1.spacing)
    image2 = Image(windowing.apply_hamming_window(image2), image2.spacing)

    iterator = iterators.AxialExcludeSectionedFourierShellIterator(
        image1.shape, options.d_bin, options.d_angle, options.d_extract_angle
    )
    fsc_task = DirectionalFSC(image1, image2, iterator)
    data = fsc_task.execute()

    analyzer = fsc_analysis.FourierCorrelationAnalysis(data, image1.spacing[0], options)
    return analyzer.execute(z_correction=z_correction)


class DirectionalFSC:
    def __init__(self, image1, image2, iterator, normalize_power=False):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("You should provide a stack for FSC analysis")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        # Create an Iterator
        self.iterator = iterator

        # FFT transforms of the input images
        self.fft_image1 = fftutils.fft(image1, window=None)
        self.fft_image2 = fftutils.fft(image2, window=None)

        if normalize_power:
            pixels = image1.shape[0] ** 3
            self.fft_image1 /= np.array(pixels * np.mean(image1))
            self.fft_image2 /= np.array(pixels * np.mean(image2))

        self._result = None

        self.pixel_size = image1.spacing[0]

    @property
    def result(self):
        if self._result is None:
            return self.execute()
        else:
            return self._result

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        data_structure = containers.FourierCorrelationDataCollection()
        radii, angles = self.iterator.steps
        freq_nyq = self.iterator.nyquist
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        # Iterate through the sphere and calculate initial values
        for ind_ring, shell_idx, rotation_idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
            c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)

            points[rotation_idx, shell_idx] = len(subset1)

        # Finish up FRC calculation for every rotation angle and sav
        # results to the data structure.
        for i in range(angles.size):
            # Calculate FRC for every orientation
            spatial_freq = radii.astype(np.float32) / freq_nyq
            n_points = np.array(points[i])
            frc = ndarray.safe_divide(c1[i], np.sqrt(c2[i] * c3[i]))

            result = containers.FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            # Save result to the structure and move to next
            # angle
            data_structure[angles[i]] = result

        return data_structure
