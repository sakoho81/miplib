import numpy as np
import pytest
from skimage import data as skdata

from miplib.data.containers.image import Image


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
