from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.signal

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.wiener import (
    _CUDA_AVAILABLE,
    wiener_deconvolution,
)


def _gaussian_spot_img(shape, sigma=2.0):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    data = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    return Image(data, spacing=(1.0, 1.0))


def test_wiener_improves_blur():
    """Wiener deconvolution should be substantially closer to the original than the blur."""
    shape = (64, 64)
    original = _gaussian_spot_img(shape, sigma=3.0)
    psf = _gaussian_spot_img(shape, sigma=1.5)

    psf_norm = psf.view(np.ndarray) / psf.max()
    blurred = Image(
        scipy.signal.fftconvolve(original, psf_norm, mode="same"),
        spacing=original.spacing,
    )

    blur_mse = np.mean((blurred - original) ** 2)

    recovered = wiener_deconvolution(blurred, psf, nsr=1e-6, backend="cpu")
    recovered_mse = np.mean((recovered - original) ** 2)

    assert recovered_mse < blur_mse * 0.1, (
        f"Wiener MSE={recovered_mse:.6f} should be < 10% of blur MSE={blur_mse:.6f}"
    )


def test_wiener_preserves_flatness():
    """Wiener of a constant image should still be constant (flat)."""
    shape = (32, 32)
    const = Image(np.ones(shape, dtype=np.float64), spacing=(1.0, 1.0))
    psf = _gaussian_spot_img(shape, sigma=2.0)

    result = wiener_deconvolution(const, psf, nsr=1e-10, backend="cpu")
    assert np.std(result) < 1e-10


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_wiener_cuda_matches_cpu():
    shape = (32, 32)
    original = _gaussian_spot_img(shape, sigma=3.0)
    psf = _gaussian_spot_img(shape, sigma=1.5)

    psf_norm = psf.view(np.ndarray) / psf.max()
    blurred = Image(
        scipy.signal.fftconvolve(original, psf_norm, mode="same"),
        spacing=original.spacing,
    )

    cpu_result = wiener_deconvolution(blurred, psf, nsr=1e-6, backend="cpu")
    cuda_result = wiener_deconvolution(blurred, psf, nsr=1e-6, backend="cuda")

    npt.assert_allclose(cpu_result, cuda_result, rtol=1e-3)


def test_wiener_invalid_backend():
    img = Image(np.ones((8, 8), dtype=np.float64), spacing=(1.0, 1.0))
    psf = _gaussian_spot_img((8, 8))
    with pytest.raises(ValueError, match="Unknown backend"):
        wiener_deconvolution(img, psf, backend="invalid")
