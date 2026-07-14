from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.processing.image as imops
import miplib.processing.ndarray as arrayutils
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)
from miplib.data.containers.image import Image
from miplib.processing import fftutils, windowing

from . import analysis as fsc_analysis


@dataclass
class FRCOptions:
    d_bin: float = 1.0
    disable_hamming: bool = False
    d_angle: float = 45.0
    d_extract_angle: float = 5.0
    frc_curve_fit_degree: int = 8
    frc_curve_fit_type: str = "spline"
    resolution_threshold_criterion: str = "fixed"
    resolution_threshold_value: float = 1.0 / 7
    resolution_snr_value: float = 0.25


def namespace_to_frc_options(ns: object) -> FRCOptions:
    """Convert an argparse.Namespace or any object to FRCOptions.

    Extracts only the fields that match FRCOptions field names.
    This bridges the old CLI layer (argparse flags) to the new dataclass API.
    """
    return FRCOptions(
        d_bin=getattr(ns, "d_bin", 1.0),
        disable_hamming=getattr(ns, "disable_hamming", False),
        d_angle=getattr(ns, "d_angle", 45.0),
        d_extract_angle=getattr(ns, "d_extract_angle", 5.0),
        frc_curve_fit_degree=getattr(ns, "frc_curve_fit_degree", 8),
        frc_curve_fit_type=getattr(ns, "frc_curve_fit_type", "spline"),
        resolution_threshold_criterion=getattr(
            ns, "resolution_threshold_criterion", "fixed"
        ),
        resolution_threshold_value=getattr(ns, "resolution_threshold_value", 1.0 / 7),
        resolution_snr_value=getattr(ns, "resolution_snr_value", 0.25),
    )


def create_fourier_iterator(
    shape: tuple[int, ...], d_bin: float = 1.0
) -> iterators.FourierRingIterator:
    """Create a Fourier iterator appropriate for the given image dimensionality.

    For 2D shapes returns a FourierRingIterator, for 3D returns a FourierShellIterator.
    """
    if len(shape) == 2:
        return iterators.FourierRingIterator(shape, d_bin)
    raise ValueError(f"Unsupported dimensionality: {len(shape)}D")


def accumulate_fourier_correlation(
    fft1: np.ndarray,
    fft2: np.ndarray,
    iterator: iterators.FourierRingIterator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Iterate over Fourier rings/shells and accumulate c1, c2, c3, n_points.

    Args:
        fft1: FFT of first image.
        fft2: FFT of second image.
        iterator: A Fourier ring/shell iterator yielding (indices, bin_idx) tuples.

    Returns:
        (c1, c2, c3, n_points) arrays, one per radial bin.
    """
    radii = iterator.radii
    c1 = np.zeros(radii.shape, dtype=np.float32)
    c2 = np.zeros(radii.shape, dtype=np.float32)
    c3 = np.zeros(radii.shape, dtype=np.float32)
    n_points = np.zeros(radii.shape, dtype=np.float32)

    for indices, idx in iterator:
        subset1 = fft1[indices]
        subset2 = fft2[indices]
        c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
        c2[idx] = np.sum(np.abs(subset1) ** 2)
        c3[idx] = np.sum(np.abs(subset2) ** 2)
        n_points[idx] = len(subset1)

    return c1, c2, c3, n_points


def build_correlation_curve(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    radii: np.ndarray,
    nyquist: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the FRC curve from accumulated correlation values.

    Args:
        c1: Sum of F1 * conj(F2) per bin.
        c2: Sum of |F1|^2 per bin.
        c3: Sum of |F2|^2 per bin.
        radii: Radial distances for each bin.
        nyquist: Nyquist frequency for normalization.

    Returns:
        (spatial_freq, correlation) arrays.
    """
    spatial_freq = radii.astype(np.float32) / nyquist
    frc = arrayutils.safe_divide(np.abs(c1), np.sqrt(c2 * c3))
    return spatial_freq, frc


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

    if not options.disable_hamming:
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

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    analyzer = fsc_analysis.FourierCorrelationAnalysis(
        frc_data, image1.spacing[0], options
    )

    result = analyzer.execute(z_correction=z_correction)[0]
    point = result.resolution["resolution-point"][1]

    cut_off_correction = func(point, *params)
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

    if not options.disable_hamming:
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
    rotation: float = 45,
    *,
    options: FRCOptions | None = None,
    orthogonal: bool = True,
):
    if options is None:
        options = FRCOptions()

    frc_data = FourierCorrelationDataCollection()

    if not options.disable_hamming:
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

    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d

    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    analyzer = fsc_analysis.FourierCorrelationAnalysis(
        frc_data, image1.spacing[0], options
    )

    result = analyzer.execute()[0]
    point = result.resolution["resolution-point"][1]

    log_correction = func(point, *params)
    result.resolution["spacing"] /= log_correction
    result.resolution["resolution"] /= log_correction

    return result


class FRC:
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

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
        """
        Calculate the FRC
        :return: Returns the FRC results.

        """
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
