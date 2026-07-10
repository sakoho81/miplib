from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.signal

from miplib.processing.ndarray import nroot, safe_divide
from miplib.processing.ops_ext import div_unit_grad, update_estimate_poisson

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import cupyx.scipy.fftpack as cufft

    _CUDA_AVAILABLE = True
except ImportError:
    _CUDA_AVAILABLE = False


@dataclass
class ViewData:
    """Per-iteration data shared between backends."""

    source: object  # duck-typed DataSource
    psfs: list[np.ndarray]
    adj_psfs: list[np.ndarray]
    weights: list[float]
    backgrounds: list[float]

    @property
    def n_views(self) -> int:
        return len(self.psfs)


# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------


class CPUBackend:
    """In-memory RL compute kernel (one block at a time)."""

    def __init__(self, view_data: ViewData) -> None:
        self._vd = view_data

    def compute_block(
        self,
        est_block: np.ndarray,
        img_blocks: list[np.ndarray],
        options,  # RLOptions
    ) -> tuple[np.ndarray, float, float, float, float]:
        """One RL update for a single block across all views.

        Returns ``(updated_block, e, s, u, n)``.
        """
        vd = self._vd

        if options.fusion_mode == "summative":
            correction: np.ndarray = np.zeros_like(est_block)
            for img, psf, adj, w, bg in zip(
                img_blocks, vd.psfs, vd.adj_psfs, vd.weights, vd.backgrounds
            ):
                forward = _convolve(est_block, psf) * w + bg
                ratio = safe_divide(img, forward)
                correction += _convolve(ratio, adj)
            correction /= vd.n_views
        else:
            correction = np.ones_like(est_block)
            for img, psf, adj, w, bg in zip(
                img_blocks, vd.psfs, vd.adj_psfs, vd.weights, vd.backgrounds
            ):
                forward = _convolve(est_block, psf) * w + bg
                ratio = safe_divide(img, forward)
                correction *= _convolve(ratio, adj)
            correction = np.asarray(nroot(correction, vd.n_views))

        if options.tv_lambda > 0:
            correction += options.tv_lambda * div_unit_grad(
                est_block, (1.0,) * est_block.ndim
            )

        new_block = est_block.copy()
        e, s, u, n = update_estimate_poisson(new_block, correction, options.epsilon)
        return new_block, e, s, u, n


# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------


class CUDABackend:
    """GPU-accelerated RL compute kernel.

    PSF FFTs are pre-computed at construction time and reused on every call.
    """

    def __init__(self, view_data: ViewData, shape: tuple[int, ...]) -> None:
        if not _CUDA_AVAILABLE:
            raise RuntimeError("cupy is not installed")
        self._vd = view_data
        self._shape = shape

        cpx = cp.complex64 if view_data.psfs[0].dtype == np.float32 else cp.complex128
        self._psfs_fft = [
            cufft.fftn(cp.asarray(p, dtype=cpx), shape, overwrite_x=True)
            for p in view_data.psfs
        ]
        self._adjs_fft = [
            cufft.fftn(cp.asarray(a, dtype=cpx), shape, overwrite_x=True)
            for a in view_data.adj_psfs
        ]

    def compute_block(
        self,
        est_block: np.ndarray,
        img_blocks: list[np.ndarray],
        options,  # RLOptions
    ) -> tuple[np.ndarray, float, float, float, float]:
        vd = self._vd
        cpx = cp.complex64 if est_block.dtype == np.float32 else cp.complex128

        if options.fusion_mode == "summative":
            correction: np.ndarray = np.zeros_like(est_block)
            for img, psf_f, adj_f, w, bg in zip(
                img_blocks, self._psfs_fft, self._adjs_fft, vd.weights, vd.backgrounds
            ):
                forward = _cuda_convolve(est_block, psf_f, self._shape, cpx) * w + bg
                ratio = _safe_divide(img, forward)
                correction += _cuda_convolve(ratio, adj_f, self._shape, cpx)
            correction /= vd.n_views
        else:
            correction = np.ones_like(est_block)
            for img, psf_f, adj_f, w, bg in zip(
                img_blocks, self._psfs_fft, self._adjs_fft, vd.weights, vd.backgrounds
            ):
                forward = _cuda_convolve(est_block, psf_f, self._shape, cpx) * w + bg
                ratio = _safe_divide(img, forward)
                correction *= _cuda_convolve(ratio, adj_f, self._shape, cpx)
            correction = np.asarray(nroot(correction, vd.n_views))

        if options.tv_lambda > 0:
            correction += options.tv_lambda * div_unit_grad(
                est_block, (1.0,) * est_block.ndim
            )

        new_block = est_block.copy()
        e, s, u, n = update_estimate_poisson(new_block, correction, options.epsilon)
        return new_block, e, s, u, n


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def resolve_backend(
    name: str,
    view_data: ViewData,
    shape: tuple[int, ...],
) -> CPUBackend | CUDABackend:
    """Resolve a backend from a string name."""
    if name == "cuda":
        return CUDABackend(view_data, shape)
    if name == "cpu":
        return CPUBackend(view_data)
    raise ValueError(f"Unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return scipy.signal.fftconvolve(a, b, mode="same")


def _cuda_convolve(
    a: np.ndarray, b_fft: "cp.ndarray", shape: tuple[int, ...], cpx
) -> np.ndarray:
    a_dev = cp.asarray(a, dtype=cpx)
    a_fft = cufft.fftn(a_dev, shape, overwrite_x=True)
    a_fft *= b_fft
    result_dev = cufft.ifftn(a_fft, shape, overwrite_x=True)
    return cp.asnumpy(cp.abs(result_dev))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, 0.0)
