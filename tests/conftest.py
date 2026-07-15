import numpy as np
import pytest
from skimage import data as skdata

from miplib.analysis.resolution.common import FRCOptions
from miplib.data.containers.image import Image
from miplib.psf.psfgen import PsfFromFwhm


def gaussian_spot(shape, sigma=2.0):
    """n-dimensional Gaussian spot centered in the array."""
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    return np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)


def sine_grating(shape, frequency=0.1):
    """Sinusoidal grating along the first axis, broadcast across remaining dims."""
    x = np.arange(shape[0])
    grating = np.sin(2 * np.pi * frequency * x)
    return np.broadcast_to(
        grating.reshape(shape[0], *[1] * (len(shape) - 1)), shape
    ).astype(np.float64)


def impulse(shape):
    """Single 1.0 value at center, zeros elsewhere."""
    arr = np.zeros(shape, dtype=np.float64)
    arr[tuple(s // 2 for s in shape)] = 1.0
    return arr


def step_edge(shape, axis=0):
    """0 on first half, 1 on second half along `axis` — broadband content."""
    arr = np.zeros(shape, dtype=np.float64)
    idx = tuple(
        slice(s // 2, None) if i == axis else slice(None) for i, s in enumerate(shape)
    )
    arr[idx] = 1.0
    return arr


def bin_aligned_sine(shape, n_cycles, axis=0):
    """Sine with integer number of cycles along `axis` — zero spectral leakage."""
    n = shape[axis]
    sine_1d = np.sin(2 * np.pi * n_cycles * np.arange(n) / n).astype(np.float64)
    shape_1d = [1] * len(shape)
    shape_1d[axis] = n
    return np.broadcast_to(sine_1d.reshape(shape_1d), shape)


def two_frequency_signal(shape, low_cycles, high_cycles, axis=0):
    """Sum of two bin-aligned sine waves — for passband/stopband verification."""
    return bin_aligned_sine(shape, low_cycles, axis) + bin_aligned_sine(
        shape, high_cycles, axis
    )


def checkerboard_pattern(shape):
    """2D 0/1 checkerboard where (x + y) % 2 == 0 positions are 1.

    Splitting this pattern via a forward checkerboard subsample produces
    constant-1 halves; the reverse split produces constant-0 halves.
    """
    y, x = np.mgrid[: shape[0], : shape[1]]
    return np.where((x + y) % 2 == 0, 1.0, 0.0).astype(np.float64)


@pytest.fixture
def image_2d():
    """16x16 float64 Image with isotropic spacing."""
    return Image(np.ones((16, 16), dtype=np.float64), spacing=(0.1, 0.1))


@pytest.fixture
def image_3d():
    """8x16x16 float64 Image with anisotropic spacing."""
    return Image(np.ones((8, 16, 16), dtype=np.float64), spacing=(0.2, 0.1, 0.1))


@pytest.fixture
def gaussian_2d():
    """32x32 Gaussian spot (sigma=3, centered), isotropic spacing."""
    return Image(gaussian_spot((32, 32), sigma=3.0), spacing=(0.1, 0.1))


@pytest.fixture
def camera_image():
    """Classic 512x512 camera test image as Image with unit spacing."""
    return Image(skdata.camera().astype(np.float64), spacing=(1.0, 1.0))


@pytest.fixture
def shepp_logan():
    """Shepp-Logan phantom — known frequency content, unit spacing."""
    return Image(skdata.shepp_logan_phantom(), spacing=(1.0, 1.0))


@pytest.fixture
def blobs_3d():
    """64x64x64 synthetic 3D blobs, fixed seed for reproducibility."""
    rng = np.random.default_rng(42)
    blobs = skdata.binary_blobs(
        length=64, n_dim=3, volume_fraction=0.5, rng=rng
    ).astype(np.float64)
    return Image(blobs, spacing=(0.2, 0.1, 0.1))


@pytest.fixture
def noisy_blobs_3d(blobs_3d):
    """blobs_3d with additive Gaussian noise via 'noisy' — ensures FSC curve decays."""
    from miplib.processing.image import noisy

    return noisy(blobs_3d, "gauss")


@pytest.fixture
def gaussian_field_pair_3d():
    """Two 64x64x64 images of the same Gaussian-smoothed random field with
    independent additive Gaussian noise — simulates two acquisitions of the
    same object for two-image FSC.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(42)
    smooth = gaussian_filter(rng.standard_normal((64, 64, 64)), sigma=4.0).astype(
        np.float64
    )
    im1 = Image(
        smooth + 0.3 * rng.standard_normal((64, 64, 64)), spacing=(0.2, 0.1, 0.1)
    )
    im2 = Image(
        smooth + 0.3 * rng.standard_normal((64, 64, 64)), spacing=(0.2, 0.1, 0.1)
    )
    return im1, im2


@pytest.fixture
def psf_gaussian_2d():
    """2D Gaussian PSF from FWHM (2 µm, 128×128, 4 µm FOV).

    Spacing = 4/128 µm/px. Sigma = FWHM / (2*sqrt(2*ln(2))) ≈ 0.425 * FWHM.
    Useful for deconvolution and resolution testing with known PSF parameters.
    """
    return PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128), dims=(4.0, 4.0)).xy()


@pytest.fixture
def frc_options():
    """Default FRCOptions for FRC/FSC testing."""
    return FRCOptions()
