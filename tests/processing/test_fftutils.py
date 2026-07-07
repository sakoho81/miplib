import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.processing import windowing
from miplib.processing.fftutils import (
    butterworth_fft_filter,
    fft,
    gaussian_fft_filter,
    ideal_fft_filter,
    ifft,
)
from tests.conftest import sine_grating, step_edge

# --- Round-trip (fundamental) ---


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


# --- Centering / self-duality ---


def test_fft_constant_image_centered():
    """FFT of all-ones: single peak at center (DC term after fftshift)."""
    ones = np.ones((8, 8))
    f = fft(ones, window=None)
    mag = np.abs(f)
    nonzero = np.argwhere(mag > 1e-10)
    assert len(nonzero) == 1
    assert tuple(nonzero[0]) == (4, 4)


def test_fft_gaussian_centered_peak(gaussian_2d):
    """Gaussian self-duality: |FFT| peaks at center, decays radially."""
    f = fft(gaussian_2d, window=None)
    mag = np.abs(f)
    center = (mag.shape[0] // 2, mag.shape[1] // 2)
    peak_idx = np.unravel_index(np.argmax(mag), mag.shape)
    assert peak_idx == center
    assert mag[center] > mag[center[0] + 5, center[1]]


# --- Interpolation ---


def test_fft_interpolation_default_is_identity():
    x = np.ones((8, 8))
    f = fft(x, window=None)
    assert f.shape == (8, 8)


def test_fft_interpolation_consolidated():
    """interpolation=2.0 doubles FFT output shape and increases
    frequency-domain FWHM (Gaussian self-duality)."""
    coords = np.arange(32) - 16
    xx, yy = np.meshgrid(coords, coords)
    g = np.exp(-(xx**2 + yy**2) / (2 * 3.0**2))
    f1 = np.abs(fft(g, window=None))
    f2 = np.abs(fft(g, window=None, interpolation=2.0))
    assert f2.shape == (64, 64)
    assert f1.shape == (32, 32)

    def fwhm_1d(profile):
        half = profile.max() / 2
        above = profile > half
        edges = np.diff(above.astype(int))
        left = np.where(edges == 1)[0][0]
        right = np.where(edges == -1)[0][-1]
        return right - left

    c1, c2 = f1.shape[0] // 2, f2.shape[0] // 2
    w1 = fwhm_1d(f1[c1, :])
    w2 = fwhm_1d(f2[c2, :])
    assert 1.8 < w2 / w1 < 3.0  # approximately double (discrete grid)


# --- Windowing ---


def test_fft_window_none_preserves_shape():
    x = np.random.rand(16, 16)
    f = fft(x, window=None)
    assert f.shape == (16, 16)
    assert f.dtype == np.complex128


def test_window_hamming_equivalence():
    """fft(window='hamming') equals external hamming application."""
    x = np.random.rand(16, 16)
    f_wrapped = fft(x, window="hamming")
    f_external = fft(windowing.apply_hamming_window(x), window=None)
    npt.assert_allclose(f_wrapped, f_external, atol=1e-12)


def test_window_tukey_alpha_depends_on_alpha():
    """Different alpha values produce different FFT outputs."""
    x = np.ones((32, 32))
    f_lo = fft(x, window="tukey", alpha=0.1)
    f_hi = fft(x, window="tukey", alpha=0.9)
    assert not np.allclose(f_lo, f_hi)


def test_window_reduces_spectral_leakage():
    """Windowing reduces sidelobe energy for a non-bin-aligned sine."""
    grating = sine_grating((64, 64), frequency=2.7)
    f_none = fft(grating, window=None)
    f_hamming = fft(grating, window="hamming")
    f_tukey = fft(grating, window="tukey", alpha=0.5)

    def sidelobe_energy(f):
        mag = np.abs(f)
        peak = np.unravel_index(np.argmax(mag), mag.shape)
        # mask ±3 bins around the peak to exclude the main lobe
        r = 3
        i0, j0 = peak[0], peak[1]
        mag_clipped = mag.copy()
        mag_clipped[i0 - r : i0 + r + 1, j0 - r : j0 + r + 1] = 0
        # also mask the symmetric peak on the other side
        i1 = mag.shape[0] - 1 - i0
        mag_clipped[i1 - r : i1 + r + 1, j0 - r : j0 + r + 1] = 0
        return mag_clipped.sum()

    assert sidelobe_energy(f_none) > sidelobe_energy(f_hamming)
    assert sidelobe_energy(f_none) > sidelobe_energy(f_tukey)


def test_unknown_window_raises():
    with pytest.raises(ValueError, match="Unknown window"):
        fft(np.ones((8, 8)), window="bad")  # type: ignore[arg-type]


# --- Step-edge ringing (broadband transient response) ---


def test_step_edge_ideal_filter_different_from_input():
    """Ideal filter changes a step edge (broadband → ringing/overshoot)."""
    edge = step_edge((64, 64), axis=0)
    img = Image(edge, (0.1, 0.1))
    filtered = ideal_fft_filter(img, threshold=0.15, kind="low")
    assert filtered.shape == img.shape
    assert filtered.spacing == img.spacing
    # ideal filter produces overshoot (ringing) beyond [0, 1]
    assert filtered.max() > 1.0


def test_step_edge_gaussian_smooth():
    """Gaussian filter produces a smooth transition without severe overshoot."""
    edge = step_edge((64, 64), axis=0)
    img = Image(edge, (0.1, 0.1))
    filtered = gaussian_fft_filter(img, threshold=0.15)
    # Gaussian is smooth: negligible overshoot, close to step edge bounds
    assert abs(filtered.max() - 1.0) < 0.02
    assert abs(filtered.min() - 0.0) < 0.02


def test_step_edge_butterworth_ringing_increases_with_order():
    """Higher-order Butterworth approaches ideal → sharper transition."""
    edge = step_edge((64, 64), axis=0)
    img = Image(edge, (0.1, 0.1))
    f_lo = butterworth_fft_filter(img, threshold=0.15, n=1)
    f_hi = butterworth_fft_filter(img, threshold=0.15, n=8)
    # higher order → larger gradients near the edge → higher max value
    # (both differ from original — filter is applied)
    assert not np.allclose(f_lo[:32, :], edge[:32, :])
    assert not np.allclose(f_hi[:32, :], edge[:32, :])
    # higher order → more ringing → greater overshoot
    assert f_hi.max() >= f_lo.max()


# --- Non-constant signals are changed (filter applied) ---


def test_ideal_filter_low_pass_affects_nonconstant():
    """Low-pass filter alters a non-constant image (high frequencies removed)."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = ideal_fft_filter(img, threshold=0.1, kind="low")
    assert filtered.shape == img.shape
    assert filtered.spacing == img.spacing
    assert not np.allclose(filtered[:10, :10], grating[:10, :10])


def test_ideal_filter_high_pass_affects_nonconstant():
    """High-pass filter alters a non-constant image."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = ideal_fft_filter(img, threshold=0.3, kind="high")
    assert filtered.shape == img.shape
    assert filtered.spacing == img.spacing
    assert not np.allclose(filtered[:10, :10], grating[:10, :10])


def test_butterworth_filter_affects_nonconstant():
    """Butterworth filter alters a non-constant image."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = butterworth_fft_filter(img, threshold=0.15)
    assert filtered.shape == img.shape
    assert filtered.spacing == img.spacing
    assert not np.allclose(filtered[:10, :10], grating[:10, :10])


def test_gaussian_filter_affects_nonconstant():
    """Gaussian filter alters a non-constant image."""
    grating = sine_grating((64, 64), frequency=0.3)
    img = Image(grating, (0.1, 0.1))
    filtered = gaussian_fft_filter(img, threshold=0.15)
    assert filtered.shape == img.shape
    assert filtered.spacing == img.spacing
    assert not np.allclose(filtered[:10, :10], grating[:10, :10])


# --- Constant-image DC identity (complementary edge case) ---


def test_ideal_filter_on_constant_low_pass():
    """Low-pass on all-ones returns all-ones (only DC present)."""
    ones = Image(np.ones((16, 16)), (0.1, 0.1))
    filtered = ideal_fft_filter(ones, threshold=0.3, kind="low")
    npt.assert_allclose(filtered, np.ones((16, 16)), atol=1e-12)


def test_ideal_filter_on_constant_high_pass():
    """High-pass on all-ones returns zeros (DC removed)."""
    ones = Image(np.ones((16, 16)), (0.1, 0.1))
    filtered = ideal_fft_filter(ones, threshold=0.3, kind="high")
    npt.assert_allclose(filtered, np.zeros((16, 16)), atol=1e-12)


def test_butterworth_filter_on_constant_returns_constant():
    """Butterworth on all-ones is identity (only DC present)."""
    ones = Image(np.ones((8, 8)), (0.1, 0.1))
    filtered = butterworth_fft_filter(ones, threshold=0.5)
    npt.assert_allclose(filtered, np.ones((8, 8)), atol=1e-10)


def test_gaussian_filter_on_constant_returns_constant():
    """Gaussian on all-ones is identity."""
    ones = Image(np.ones((8, 8)), (0.1, 0.1))
    filtered = gaussian_fft_filter(ones, threshold=0.5)
    npt.assert_allclose(filtered, np.ones((8, 8)), atol=1e-10)


# --- Input validation (error paths) ---


def test_ideal_fft_filter_invalid_kind(camera_image):
    with pytest.raises(ValueError, match="Unknown filter kind"):
        ideal_fft_filter(camera_image, threshold=0.5, kind="band")


def test_ideal_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ideal_fft_filter(camera_image, threshold=1.5)


def test_ideal_fft_filter_rejects_non_image():
    with pytest.raises(TypeError, match="Expected Image"):
        ideal_fft_filter(np.ones((8, 8)), threshold=0.5)


def test_butterworth_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        butterworth_fft_filter(camera_image, threshold=1.5)


def test_butterworth_fft_filter_invalid_n(camera_image):
    with pytest.raises(ValueError, match="n must be"):
        butterworth_fft_filter(camera_image, threshold=0.5, n=0)


def test_gaussian_fft_filter_invalid_threshold(camera_image):
    with pytest.raises(ValueError, match="between 0 and 1"):
        gaussian_fft_filter(camera_image, threshold=1.5)
