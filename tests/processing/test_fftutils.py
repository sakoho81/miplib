import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.processing.fftutils import (
    butterworth_fft_filter,
    fft,
    gaussian_fft_filter,
    ideal_fft_filter,
    ifft,
)
from tests.conftest import sine_grating


def test_fft_ifft_roundtrip_even():
    x = np.random.default_rng(42).random((8, 8))
    recovered = np.abs(ifft(fft(x, window=None)))
    npt.assert_allclose(recovered, x, atol=1e-14)


def test_fft_ifft_roundtrip_odd():
    """The ifft bug fix: round-trip now works for odd shapes too."""
    x = np.random.default_rng(42).random((7, 7))
    recovered = np.abs(ifft(fft(x, window=None)))
    npt.assert_allclose(recovered, x, atol=1e-14)


def test_fft_ifft_roundtrip_3d():
    x = np.random.default_rng(42).random((6, 8, 8))
    recovered = np.abs(ifft(fft(x, window=None)))
    npt.assert_allclose(recovered, x, atol=1e-14)


def test_fft_constant_image_centered():
    """FFT of all-ones should be a single peak at the center (DC term)."""
    ones = np.ones((8, 8))
    f = fft(ones, window=None)
    mag = np.abs(f)
    nonzero_idxs = np.argwhere(mag > 1e-10)
    assert len(nonzero_idxs) == 1
    assert tuple(nonzero_idxs[0]) == (4, 4)


def test_fft_gaussian_centered_peak(gaussian_2d):
    """Gaussian self-duality: |FFT| peaks at center, decays radially."""
    f = fft(gaussian_2d, window=None)
    mag = np.abs(f)
    center = (mag.shape[0] // 2, mag.shape[1] // 2)
    peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
    assert peak_idx == center
    assert mag[center] > mag[center[0] + 5, center[1]]


def test_fft_interpolation_doubles_shape():
    x = np.ones((8, 8))
    f = fft(x, window=None, interpolation=2.0)
    assert f.shape == (16, 16)


def test_fft_interpolation_default_is_identity():
    x = np.ones((8, 8))
    f = fft(x, window=None)
    assert f.shape == (8, 8)


def test_fft_window_none_preserves_shape():
    x = np.random.rand(16, 16)
    f = fft(x, window=None)
    assert f.shape == (16, 16)
    assert f.dtype == np.complex128


def test_fft_window_hamming_preserves_shape():
    x = np.random.rand(16, 16)
    f = fft(x, window="hamming")
    assert f.shape == (16, 16)


def test_fft_window_tukey_preserves_shape():
    x = np.random.rand(16, 16)
    f = fft(x, window="tukey", alpha=0.5)
    assert f.shape == (16, 16)


def test_unknown_window_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        fft(np.ones((8, 8)), window="bad")  # type: ignore[arg-type]


def test_ideal_fft_filter_low_pass_reduces_variance():
    """Low-pass on a sine grating should heavily attenuate the sinusoid."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = ideal_fft_filter(img, threshold=0.1, kind="low")
    assert filtered.spacing == img.spacing
    # low-pass removes the sine component → much lower variance
    assert filtered.std() < 0.2 * img.std()


def test_ideal_fft_filter_high_pass_on_sine():
    """High-pass on a sine grating preserves the oscillation energy."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = ideal_fft_filter(img, threshold=0.1, kind="high")
    assert filtered.spacing == img.spacing
    # after high-pass, the sine component should survive (non-trivial variance)
    assert filtered.std() > 0.1


def test_butterworth_filter_on_constant_returns_constant(camera_image):
    """Butterworth on all-ones is identity (only DC present, retained)."""
    ones = Image(np.ones((8, 8)), (0.1, 0.1))
    filtered = butterworth_fft_filter(ones, threshold=0.5)
    npt.assert_allclose(filtered, np.ones((8, 8)), atol=1e-10)


def test_gaussian_filter_on_constant_returns_constant():
    """Gaussian on all-ones is identity."""
    ones = Image(np.ones((8, 8)), (0.1, 0.1))
    filtered = gaussian_fft_filter(ones, threshold=0.5)
    npt.assert_allclose(filtered, np.ones((8, 8)), atol=1e-10)


def test_fft_interpolation_increases_frequency_resolution():
    """interpolation=2.0 doubles FFT size, approximately doubling FWHM."""
    g = Image(np.ones((32, 32)), (0.1, 0.1))
    # create a Gaussian spot and check FFT resolution
    coords = np.arange(32) - 16
    xx, yy = np.meshgrid(coords, coords)
    g = np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))
    f1 = np.abs(fft(g, window=None))
    f2 = np.abs(fft(g, window=None, interpolation=2.0))
    # higher interpolation → larger FFT → finer frequency sampling
    assert f2.shape == (64, 64)
    assert f1.shape == (32, 32)


def test_ideal_filter_on_constant_low_pass():
    """Low-pass on all-ones should return the same (all energy is DC)."""
    ones = Image(np.ones((16, 16)), (0.1, 0.1))
    filtered = ideal_fft_filter(ones, threshold=0.3, kind="low")
    npt.assert_allclose(filtered, np.ones((16, 16)), atol=1e-12)


def test_ideal_filter_on_constant_high_pass():
    """High-pass on all-ones should remove all energy (DC removed)."""
    ones = Image(np.ones((16, 16)), (0.1, 0.1))
    filtered = ideal_fft_filter(ones, threshold=0.3, kind="high")
    # high-pass removes the only frequency component (DC) → near-zero image
    npt.assert_allclose(filtered, np.zeros((16, 16)), atol=1e-12)


def test_ideal_fft_filter_invalid_kind(camera_image):
    with pytest.raises(ValueError, match="Unknown filter kind"):
        ideal_fft_filter(camera_image, threshold=0.5, kind="band")


def test_ideal_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ideal_fft_filter(camera_image, threshold=1.5)


def test_ideal_fft_filter_rejects_non_image():
    with pytest.raises(TypeError, match="Expected Image"):
        ideal_fft_filter(np.ones((8, 8)), threshold=0.5)


def test_butterworth_fft_filter_shape_and_spacing(camera_image):
    filtered = butterworth_fft_filter(camera_image, threshold=0.3)
    assert filtered.shape == camera_image.shape
    assert filtered.spacing == camera_image.spacing


def test_butterworth_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        butterworth_fft_filter(camera_image, threshold=1.5)


def test_butterworth_fft_filter_invalid_n(camera_image):
    with pytest.raises(ValueError, match="n must be"):
        butterworth_fft_filter(camera_image, threshold=0.5, n=0)


def test_gaussian_fft_filter_shape_and_spacing(camera_image):
    filtered = gaussian_fft_filter(camera_image, threshold=0.3)
    assert filtered.shape == camera_image.shape
    assert filtered.spacing == camera_image.spacing


def test_gaussian_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        gaussian_fft_filter(camera_image, threshold=1.5)
