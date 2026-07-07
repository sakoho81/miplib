import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.coordinates.polar import SimplePolarIndexer
from miplib.processing.fftutils import (
    butterworth_fft_filter,
    fft,
    gaussian_fft_filter,
    ideal_fft_filter,
    ifft,
)


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


def test_fft_unknown_window_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        fft(np.ones((8, 8)), window="bad")


def test_ideal_fft_filter_low_pass_removes_energy(camera_image):
    filtered = ideal_fft_filter(camera_image, threshold=0.1, kind="low")
    assert filtered.shape == camera_image.shape
    assert filtered.spacing == camera_image.spacing
    # low-pass should smooth the image — check it differs from original
    assert not np.allclose(filtered[:100, :100], camera_image[:100, :100])


def test_ideal_fft_filter_high_pass_removes_dc(camera_image):
    filtered = ideal_fft_filter(camera_image, threshold=0.05, kind="high")
    assert filtered.shape == camera_image.shape
    assert filtered.spacing == camera_image.spacing
    # high-pass should have lower mean than original (DC attenuated)
    assert np.abs(filtered.mean()) < np.abs(camera_image.mean())


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


def test_simple_polar_indexer_in_frequency_grid(camera_image):
    """The polar indexer r has the same shape as the image."""
    idx = SimplePolarIndexer(camera_image.shape)
    assert idx.r.shape == camera_image.shape
