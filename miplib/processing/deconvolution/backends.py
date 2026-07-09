from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx.scipy.fftpack as cufft

    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


def convolve_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolve *a* and *b* via scipy FFT (mode='same')."""
    return scipy.signal.fftconvolve(a, b, mode="same")


def make_cuda_psf_convolve(
    psf: np.ndarray, shape: tuple[int, ...]
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a closure that convolves data with the pre-computed FFT of *psf*.

    The PSF FFT is computed once on the GPU and reused on every call.
    Raises RuntimeError if cupy is not available.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "cupy is not installed — cannot create CUDA convolve backend"
        )

    complex_dtype = cp.complex64 if psf.dtype == np.float32 else cp.complex128
    psf_dev = cp.asarray(psf, dtype=complex_dtype)
    psf_fft = cufft.fftn(psf_dev, shape, overwrite_x=True)

    def convolve(data: np.ndarray) -> np.ndarray:
        data_dev = cp.asarray(data, dtype=complex_dtype)
        data_fft = cufft.fftn(data_dev, shape, overwrite_x=True)
        data_fft *= psf_fft
        result_dev = cufft.ifftn(data_fft, shape, overwrite_x=True)
        return cp.asnumpy(cp.abs(result_dev))

    return convolve


def make_cuda_convolve(
    shape: tuple[int, ...], dtype: np.dtype = np.dtype(np.float32)
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Return a general CUDA FFT convolve function for the given shape and dtype."""
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "cupy is not installed — cannot create CUDA convolve backend"
        )

    complex_dtype = cp.complex64 if dtype == np.float32 else cp.complex128

    def convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_dev = cp.asarray(a, dtype=complex_dtype)
        b_dev = cp.asarray(b, dtype=complex_dtype)
        a_fft = cufft.fftn(a_dev, shape, overwrite_x=True)
        b_fft = cufft.fftn(b_dev, shape, overwrite_x=True)
        a_fft *= b_fft
        result_dev = cufft.ifftn(a_fft, shape, overwrite_x=True)
        return cp.asnumpy(cp.abs(result_dev))

    return convolve
