from __future__ import annotations

from typing import Callable

import numpy as np

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import convolve_cpu
from miplib.processing.image import zoom_to_spacing
from miplib.processing.ndarray import reverse_array


def prepare_psf(
    psf: Image,
    target_spacing: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Resample *psf* to match *target_spacing*, normalize to unit sum, and
    return the mirror-reversed (adjoint) PSF.

    Returns (psf_normalized, adj_psf).
    """
    psf_resampled = zoom_to_spacing(psf, target_spacing)
    psf_resampled /= psf_resampled.sum()
    adj_psf = reverse_array(psf_resampled)
    return psf_resampled.view(np.ndarray), adj_psf.view(np.ndarray)


def prepare_psfs(
    psfs: list[Image],
    target_spacing: tuple[float, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Resample and normalize a list of PSFs."""
    norms = []
    adjs = []
    for psf in psfs:
        n, a = prepare_psf(psf, target_spacing)
        norms.append(n)
        adjs.append(a)
    return norms, adjs


def compute_virtual_psfs(
    psfs: list[np.ndarray],
    adj_psfs: list[np.ndarray],
    *,
    convolve: Callable[[np.ndarray, np.ndarray], np.ndarray] = convolve_cpu,
) -> list[np.ndarray]:
    """Replace adjoint PSFs with virtual PSFs per Preibisch et al. 2014.

    For each view *i*, the virtual adjoint PSF incorporates cross-view PSF
    information to improve multi-view deconvolution convergence.

    ``len(psfs)`` and ``len(adj_psfs)`` must be equal and >= 2.
    """
    n_views = len(psfs)
    if n_views < 2:
        raise ValueError("computing virtual PSFs requires at least 2 views")
    if len(psfs) != len(adj_psfs):
        raise ValueError("psfs and adj_psfs must have the same length")

    virtual = []
    for i in range(n_views):
        vp = np.ones_like(adj_psfs[i])
        for j in range(n_views):
            if j == i:
                continue
            cache = convolve(adj_psfs[i], psfs[j])
            cache = convolve(cache, adj_psfs[j])
            vp *= cache
        vp *= adj_psfs[i]
        vp /= vp.sum()
        virtual.append(vp)

    return virtual
