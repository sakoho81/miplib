from __future__ import annotations

import logging

import numpy as np
import numpy.fft as fft

from miplib.data.containers.image import Image
from miplib.processing.image import (
    remove_zero_padding,
    zero_pad_to_matching_shape,
    zero_pad_to_shape,
    zoom_to_spacing,
)
from miplib.processing.ndarray import safe_divide

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx.scipy.fftpack as cufft

    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


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

    if backend == "cuda":
        if not _CUDA_AVAILABLE:
            raise RuntimeError("cupy is not installed — cannot use CUDA backend")
        cpx_dtype = cp.complex64 if image_arr.dtype == np.float32 else cp.complex128
        image_f: np.ndarray = cp.asnumpy(
            cp.abs(
                cufft.ifftn(
                    cufft.fftn(cp.asarray(image_arr, dtype=cpx_dtype))
                    * _cuda_wiener_filter(cp.asarray(psf_arr, dtype=cpx_dtype), nsr)
                )
            )
        ).real
    else:
        psf_f = fft.fftn(fft.fftshift(psf_arr))
        wiener = _cpu_wiener_filter(psf_f, nsr)
        image_f = fft.ifftn(fft.fftn(image_arr) * wiener).real

    result = Image(image_f.astype(image_arr.dtype), image.spacing)
    if add_pad > 0:
        result = remove_zero_padding(result, image.shape)
    return result


def _cpu_wiener_filter(
    psf_f: np.ndarray,
    nsr: float,
) -> np.ndarray:
    psf_abs_sq = np.abs(psf_f) ** 2
    return safe_divide(psf_abs_sq, psf_abs_sq + nsr) / psf_f


def _cuda_wiener_filter(
    psf_dev: "cp.ndarray",
    nsr: float,
) -> "cp.ndarray":
    psf_f = cufft.fftn(psf_dev)
    psf_abs_sq = cp.abs(psf_f) ** 2
    wiener = psf_abs_sq / (psf_abs_sq + nsr) / psf_f
    return wiener
