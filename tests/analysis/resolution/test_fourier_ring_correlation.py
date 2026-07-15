import argparse

import numpy as np
import numpy.testing as npt
import pytest

from miplib.analysis.resolution.analysis import FitType, ResolutionCriterion
from miplib.analysis.resolution.fourier_ring_correlation import (
    FRCOptions,
    _cutoff_correction,
    accumulate_fourier_correlation,
    build_correlation_curve,
    create_fourier_iterator,
    namespace_to_frc_options,
)

# ---------------------------------------------------------------------------
# FRCOptions
# ---------------------------------------------------------------------------


def test_frc_options_defaults():
    opts = FRCOptions()
    assert opts.d_bin == 1.0
    assert opts.use_hamming is True
    assert opts.frc_curve_fit_degree == 8
    assert opts.frc_curve_fit_type == FitType.SPLINE
    assert opts.resolution_threshold_criterion == ResolutionCriterion.FIXED
    assert opts.resolution_threshold_value == pytest.approx(1.0 / 7)
    assert opts.resolution_snr_value == 0.25
    assert opts.d_angle == 45.0
    assert opts.d_extract_angle == 5.0


def test_frc_options_explicit():
    opts = FRCOptions(
        d_bin=2.0,
        use_hamming=False,
        frc_curve_fit_type=FitType.POLYNOMIAL,
        resolution_threshold_criterion=ResolutionCriterion.HALF_BIT,
    )
    assert opts.d_bin == 2.0
    assert opts.use_hamming is False
    assert opts.frc_curve_fit_type == FitType.POLYNOMIAL
    assert opts.resolution_threshold_criterion == ResolutionCriterion.HALF_BIT
    assert opts.frc_curve_fit_degree == 8


@pytest.mark.parametrize("d_bin", [0.5, 1.0, 2.0])
def test_frc_options_d_bin(d_bin):
    opts = FRCOptions(d_bin=d_bin)
    assert opts.d_bin == d_bin


# ---------------------------------------------------------------------------
# namespace_to_frc_options
# ---------------------------------------------------------------------------


def test_namespace_to_frc_options_full():
    ns = argparse.Namespace(
        d_bin=3.0,
        use_hamming=False,
        frc_curve_fit_degree=12,
        frc_curve_fit_type="polynomial",
        resolution_threshold_criterion="half-bit",
        resolution_threshold_value=0.5,
        resolution_snr_value=0.5,
        d_angle=90.0,
        d_extract_angle=10.0,
    )
    opts = namespace_to_frc_options(ns)
    assert opts.d_bin == 3.0
    assert opts.use_hamming is False
    assert opts.frc_curve_fit_degree == 12
    assert opts.frc_curve_fit_type == FitType.POLYNOMIAL
    assert opts.resolution_threshold_criterion == ResolutionCriterion.HALF_BIT
    assert opts.resolution_threshold_value == 0.5
    assert opts.resolution_snr_value == 0.5
    assert opts.d_angle == 90.0
    assert opts.d_extract_angle == 10.0


def test_namespace_to_frc_options_partial():
    ns = argparse.Namespace(d_bin=5.0)
    opts = namespace_to_frc_options(ns)
    assert opts.d_bin == 5.0
    assert opts.resolution_threshold_criterion == ResolutionCriterion.FIXED


def test_namespace_to_frc_options_unknown_attrs_ignored():
    ns = argparse.Namespace(d_bin=2.0, unknown_attr=42)
    opts = namespace_to_frc_options(ns)
    assert opts.d_bin == 2.0


def test_namespace_to_frc_options_empty():
    ns = argparse.Namespace()
    opts = namespace_to_frc_options(ns)
    assert opts.d_bin == 1.0


# ---------------------------------------------------------------------------
# cutoff_correction
# ---------------------------------------------------------------------------


def test_cutoff_correction_returns_finite():
    result = _cutoff_correction(0.5)
    assert np.isfinite(result)
    assert result > 0


# ---------------------------------------------------------------------------
# create_fourier_iterator
# ---------------------------------------------------------------------------


def test_create_fourier_iterator_2d():
    it = create_fourier_iterator((64, 64), d_bin=1.0)
    from miplib.data.iterators.fourier_ring_iterators import FourierRingIterator

    assert isinstance(it, FourierRingIterator)


@pytest.mark.parametrize("bad_shape", [(32,), (32, 32, 32), (16, 16, 16, 16)])
def test_create_fourier_iterator_rejects_non_2d(bad_shape):
    with pytest.raises(ValueError, match="Unsupported dimensionality"):
        create_fourier_iterator(bad_shape)


# ---------------------------------------------------------------------------
# accumulate_fourier_correlation
# ---------------------------------------------------------------------------


def test_accumulate_identical_ffts():
    shape = (32, 32)
    it = create_fourier_iterator(shape, d_bin=2.0)
    fft = np.ones(shape, dtype=np.complex64)
    c1, c2, c3, _ = accumulate_fourier_correlation(fft, fft, it)
    npt.assert_allclose(c1, c2, atol=1e-6)
    npt.assert_allclose(c1, c3, atol=1e-6)


def test_accumulate_n_points_positive():
    shape = (16, 16)
    it = create_fourier_iterator(shape, d_bin=1.0)
    fft = np.ones(shape, dtype=np.complex64)
    _, _, _, n_points = accumulate_fourier_correlation(fft, fft, it)
    assert np.all(n_points > 0)


def test_accumulate_preserves_dtype():
    shape = (16, 16)
    it = create_fourier_iterator(shape, d_bin=1.0)
    fft = np.random.randn(*shape).astype(np.complex64)
    c1, c2, c3, _ = accumulate_fourier_correlation(fft, fft, it)
    assert c1.dtype == np.float32
    assert c2.dtype == np.float32
    assert c3.dtype == np.float32


def test_accumulate_output_length():
    shape = (64, 64)
    it = create_fourier_iterator(shape, d_bin=2.0)
    fft = np.ones(shape, dtype=np.complex64)
    c1, c2, c3, n_points = accumulate_fourier_correlation(fft, fft, it)
    n_bins = len(it.radii)
    assert len(c1) == n_bins
    assert len(c2) == n_bins
    assert len(c3) == n_bins
    assert len(n_points) == n_bins


# ---------------------------------------------------------------------------
# build_correlation_curve
# ---------------------------------------------------------------------------


def test_correlation_curve_identical_inputs():
    radii = np.arange(10, dtype=np.float32)
    c1 = c2 = c3 = np.ones(10, dtype=np.float32)
    _, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
    npt.assert_allclose(corr, 1.0)


def test_correlation_curve_zero_c1():
    radii = np.arange(10, dtype=np.float32)
    c1 = np.zeros(10, dtype=np.float32)
    c2 = c3 = np.ones(10, dtype=np.float32)
    _, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
    npt.assert_allclose(corr, 0.0, atol=1e-6)


def test_correlation_curve_bounded():
    radii = np.arange(10, dtype=np.float32)
    c1 = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.0], dtype=np.float32)
    c2 = c3 = np.ones(10, dtype=np.float32)
    _, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
    assert np.all(corr >= 0.0)
    assert np.all(corr <= 1.0 + 1e-6)


def test_correlation_curve_frequency_normalization():
    radii = np.arange(10, dtype=np.float32)
    c1 = c2 = c3 = np.ones(10, dtype=np.float32)
    freq, _ = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
    npt.assert_allclose(freq, radii / 10.0)
    assert freq[0] == 0.0
    assert freq[-1] == 0.9


def test_correlation_curve_handles_inf():
    radii = np.arange(5, dtype=np.float32)
    c1 = np.ones(5, dtype=np.float32)
    c2 = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    c3 = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    _, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=5.0)
    assert np.isfinite(corr).all()
    assert corr[0] == 0.0


# ---------------------------------------------------------------------------
# calculate_single_image_frc (integration)
# ---------------------------------------------------------------------------


def test_single_image_frc_returns_valid_result(frc_options):
    from skimage import data as skdata

    from miplib.analysis.resolution.fourier_ring_correlation import (
        calculate_single_image_frc,
    )
    from miplib.data.containers.image import Image
    from miplib.processing.image import noisy

    im = Image(skdata.camera().astype(np.float64), spacing=(1.0, 1.0))
    im = noisy(im, "gauss")  # independent noise so FRC curve decays
    result = calculate_single_image_frc(im, frc_options)
    resolution = result.resolution["resolution"]
    assert np.isfinite(resolution)
    assert resolution > 0


@pytest.mark.integration
def test_real_ism_image_frc_resolution(frc_options):
    """FRC on a real 2PE-ISM dendrite image — expect ~0.5 µm resolution."""
    from pathlib import Path

    from skimage import io

    from miplib.analysis.resolution.fourier_ring_correlation import (
        calculate_single_image_frc,
    )
    from miplib.data.containers.image import Image

    path = Path(__file__).parent.parent.parent / "testdata" / "ism_dendrite.tiff"
    if not path.is_file():
        pytest.skip(f"Test image not found: {path}")

    data = io.imread(str(path)).astype(np.float64)
    spacing = (0.1, 0.1)  # PhysicalSizeX/Y from OME-XML metadata
    im = Image(data, spacing=spacing)

    result = calculate_single_image_frc(im, frc_options)
    resolution = result.resolution["resolution"]
    assert np.isfinite(resolution)
    assert 0.534 < resolution < 0.590, f"Expected ~0.56 µm, got {resolution:.3f} µm"
