from __future__ import annotations

import numpy as np

import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.processing.image as imops
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)
from miplib.data.containers.image import Image
from miplib.processing import fftutils, windowing

from . import analysis as fsc_analysis
from .common import (
    FRCOptions,
    _cutoff_correction,
    accumulate_fourier_correlation,
    build_correlation_curve,
)


def calculate_single_image_frc(
    image: Image,
    options: FRCOptions | None = None,
    *,
    average: bool = True,
    z_correction: float = 1.0,
) -> FourierCorrelationData:
    if options is None:
        options = FRCOptions()

    frc_data = FourierCorrelationDataCollection()

    if options.use_hamming:
        spacing = image.spacing
        image = Image(windowing.apply_hamming_window(image), spacing)

    image1, image2 = imops.checkerboard_split(image)
    image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)

    iterator = iterators.FourierRingIterator(image1.shape, options.d_bin)
    frc_task = FRC(image1, image2, iterator)
    frc_data[0] = frc_task.execute()

    if average:
        image1, image2 = imops.reverse_checkerboard_split(image)
        image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)
        iterator = iterators.FourierRingIterator(image1.shape, options.d_bin)
        frc_task = FRC(image1, image2, iterator)

        frc_data[0].correlation["correlation"] *= 0.5
        frc_data[0].correlation["correlation"] += (
            0.5 * frc_task.execute().correlation["correlation"]
        )

    analyzer = fsc_analysis.FourierCorrelationAnalysis(
        frc_data, image1.spacing[0], options
    )

    result = analyzer.execute(z_correction=z_correction)[0]

    point_data = result.resolution["resolution-point"]
    if point_data is None:
        return result

    cut_off_correction = _cutoff_correction(point_data[1])
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction

    return result


def calculate_two_image_frc(
    image1: Image,
    image2: Image,
    options: FRCOptions | None = None,
    *,
    z_correction: float = 1.0,
) -> FourierCorrelationData:
    if options is None:
        options = FRCOptions()

    frc_data = FourierCorrelationDataCollection()
    spacing = image1.spacing

    if options.use_hamming:
        image1 = Image(windowing.apply_hamming_window(image1), spacing)
        image2 = Image(windowing.apply_hamming_window(image2), spacing)

    iterator = iterators.FourierRingIterator(image1.shape, options.d_bin)
    frc_task = FRC(image1, image2, iterator)
    frc_data[0] = frc_task.execute()

    analyzer = fsc_analysis.FourierCorrelationAnalysis(
        frc_data, image1.spacing[0], options
    )

    return analyzer.execute(z_correction=z_correction)[0]


def calculate_single_image_sectioned_frc(
    image: Image,
    rotation: float,
    *,
    options: FRCOptions | None = None,
    orthogonal: bool = True,
):
    if options is None:
        options = FRCOptions()

    frc_data = FourierCorrelationDataCollection()

    if options.use_hamming:
        spacing = image.spacing
        image = Image(windowing.apply_hamming_window(image), spacing)

    def frc_helper(image1, image2, rotation):
        iterator = iterators.SectionedFourierRingIterator(
            image1.shape, options.d_bin, options.d_angle
        )
        iterator.angle = rotation
        frc_task = FRC(image1, image2, iterator)
        return frc_task.execute()

    image1, image2 = imops.checkerboard_split(image)
    image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)

    image1_r, image2_r = imops.reverse_checkerboard_split(image)
    image1_r, image2_r = imops.zero_pad_to_matching_shape(image1_r, image2_r)

    pair_1 = frc_helper(image1, image2, rotation)
    pair_2 = frc_helper(image1_r, image2_r, rotation)

    pair_1.correlation["correlation"] *= 0.5
    pair_1.correlation["correlation"] += 0.5 * pair_2.correlation["correlation"]

    if orthogonal:
        pair_1_o = frc_helper(image1, image2, rotation + 90)
        pair_2_o = frc_helper(image1_r, image2_r, rotation + 90)

        pair_1_o.correlation["correlation"] *= 0.5
        pair_1_o.correlation["correlation"] += 0.5 * pair_2_o.correlation["correlation"]

        pair_1.correlation["correlation"] += 0.5 * pair_1_o.correlation["correlation"]

    frc_data[0] = pair_1

    analyzer = fsc_analysis.FourierCorrelationAnalysis(
        frc_data, image1.spacing[0], options
    )

    result = analyzer.execute()[0]

    point_data = result.resolution["resolution-point"]
    if point_data is None:
        return result

    log_correction = _cutoff_correction(point_data[1])
    result.resolution["spacing"] /= log_correction
    result.resolution["resolution"] /= log_correction

    return result


class FRC:
    def __init__(self, image1, image2, iterator):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape or tuple(image1.spacing) != tuple(
            image2.spacing
        ):
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.pixel_size = image1.spacing[0]

        # Expand to square
        image1 = imops.zero_pad_to_cube(image1)
        image2 = imops.zero_pad_to_cube(image2)

        self.iterator = iterator
        # Calculate power spectra for the input images.
        self.fft_image1 = fftutils.fft(image1, window=None)
        self.fft_image2 = fftutils.fft(image2, window=None)

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))

    def execute(self):
        c1, c2, c3, n_points = accumulate_fourier_correlation(
            self.fft_image1, self.fft_image2, self.iterator
        )

        spatial_freq, frc = build_correlation_curve(
            c1, c2, c3, self.iterator.radii, self.freq_nyq
        )

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set
