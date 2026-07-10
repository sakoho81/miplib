from __future__ import annotations

import logging

import numpy as np

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import resolve_fft
from miplib.processing.image import (
    remove_zero_padding,
    zero_pad_to_matching_shape,
    zero_pad_to_shape,
    zoom_to_spacing,
)

logger = logging.getLogger(__name__)


def wiener_deconvolution(
    image: Image,
    psf: Image,
    nsr: float = 30,
    add_pad: int = 0,
    *,
    backend: str = "cpu",
) -> Image:
    """Linear Wiener deconvolution.

    Parameters
    ----------
    image : Image
        Blurred / observed image.
    psf : Image
        Point-spread function.
    nsr : float
        Noise-to-signal ratio (higher = more regularization, lower = sharper).
    add_pad : int
        Extra zero-padding voxels beyond the PSF support.
    backend : {"cpu", "cuda"}
        Computation backend.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError(f"Unknown backend: {backend!r}")

    psf = zoom_to_spacing(psf, image.spacing)
    image, psf = zero_pad_to_matching_shape(image, psf)
    if add_pad > 0:
        pad_shape = tuple(s + add_pad * 2 for s in image.shape)
        image = zero_pad_to_shape(image, pad_shape)
        psf = zero_pad_to_shape(psf, pad_shape)

    psf_arr = psf.view(np.ndarray) / psf.max()
    image_arr = image.view(np.ndarray)

    fft_backend = resolve_fft(backend)

    psf_f = fft_backend.fftn(fft_backend.fftshift(psf_arr))
    wiener = _wiener_filter(psf_f, nsr)
    image_f = fft_backend.ifftn(fft_backend.fftn(image_arr) * wiener)

    result = Image(image_f.astype(image_arr.dtype), image.spacing)
    if add_pad > 0:
        result = remove_zero_padding(result, image.shape)
    return result


def _wiener_filter(
    psf_f: np.ndarray,
    nsr: float,
) -> np.ndarray:
    """Frequency-domain Wiener filter — works on numpy or cupy arrays."""
    psf_abs_sq = np.abs(psf_f) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        result = psf_abs_sq / (psf_abs_sq + nsr) / psf_f
        result[result == np.inf] = 0.0
        return np.nan_to_num(result)
