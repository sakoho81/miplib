from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.signal

from miplib.data.adapters.image_data import DataSource
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

    source: DataSource
    psfs: list[np.ndarray]
    adj_psfs: list[np.ndarray]
    weights: list[float]
    backgrounds: list[float]

    @property
    def n_views(self) -> int:
        return len(self.psfs)


# ---------------------------------------------------------------------------
# base backend
# ---------------------------------------------------------------------------


class _BaseBackend:
    def __init__(self, view_data: ViewData) -> None:
        self._vd = view_data

    def _forward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError

    def _backward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _accumulate(
        est_block: np.ndarray,
        img_blocks: list[np.ndarray],
        vd: ViewData,
        fwd_fn,
        bwd_fn,
        fusion_mode,
    ) -> np.ndarray:
        """Accumulate the RL correction across all views."""
        if fusion_mode.value == "summative":
            correction: np.ndarray = np.zeros_like(est_block)
            for idx, (img, w, bg) in enumerate(
                zip(img_blocks, vd.weights, vd.backgrounds)
            ):
                forward = fwd_fn(est_block, idx) * w + bg
                ratio = safe_divide(img, forward)
                correction += bwd_fn(ratio, idx)
            correction /= vd.n_views
        else:
            correction = np.ones_like(est_block)
            for idx, (img, w, bg) in enumerate(
                zip(img_blocks, vd.weights, vd.backgrounds)
            ):
                forward = fwd_fn(est_block, idx) * w + bg
                ratio = safe_divide(img, forward)
                correction *= bwd_fn(ratio, idx)
            correction = np.asarray(nroot(correction, vd.n_views))
        return correction

    @staticmethod
    def _apply_update(
        est_block: np.ndarray,
        correction: np.ndarray,
        options,  # RLOptions
    ) -> tuple[np.ndarray, float, float, float, float]:
        """Apply TV regularization and Poisson update."""
        if options.tv_lambda > 0:
            correction += options.tv_lambda * div_unit_grad(
                est_block, (1.0,) * est_block.ndim
            )
        new_block = est_block.copy()
        e, s, u, n = update_estimate_poisson(new_block, correction, options.epsilon)
        return new_block, e, s, u, n

    def compute_block(
        self,
        est_block: np.ndarray,
        img_blocks: list[np.ndarray],
        options,  # RLOptions
    ) -> tuple[np.ndarray, float, float, float, float]:
        correction = self._accumulate(
            est_block,
            img_blocks,
            self._vd,
            self._forward_convolve,
            self._backward_convolve,
            options.fusion_mode,
        )
        return self._apply_update(est_block, correction, options)


# ---------------------------------------------------------------------------
# CPU backend
# ---------------------------------------------------------------------------


class CPUBackend(_BaseBackend):
    """In-memory RL compute kernel (one block at a time)."""

    def _forward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        return _convolve(data, self._vd.psfs[idx])

    def _backward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        return _convolve(data, self._vd.adj_psfs[idx])


# ---------------------------------------------------------------------------
# CUDA backend
# ---------------------------------------------------------------------------


class CUDABackend(_BaseBackend):
    """GPU-accelerated RL compute kernel.

    PSF FFTs are pre-computed at construction time and reused on every call.
    """

    def __init__(self, view_data: ViewData, shape: tuple[int, ...]) -> None:
        if not _CUDA_AVAILABLE:
            raise RuntimeError("cupy is not installed")
        super().__init__(view_data)
        self._shape = shape

        cpx = cp.complex64 if view_data.psfs[0].dtype == np.float32 else cp.complex128
        self._cpx = cpx
        self._psfs_fft = [
            cufft.fftn(cp.asarray(p, dtype=cpx), shape, overwrite_x=True)
            for p in view_data.psfs
        ]
        self._adjs_fft = [
            cufft.fftn(cp.asarray(a, dtype=cpx), shape, overwrite_x=True)
            for a in view_data.adj_psfs
        ]

    def _forward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        return _cuda_convolve(data, self._psfs_fft[idx], self._shape, self._cpx)

    def _backward_convolve(self, data: np.ndarray, idx: int) -> np.ndarray:
        return _cuda_convolve(data, self._adjs_fft[idx], self._shape, self._cpx)


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


def convolve_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolve *a* and *b* via scipy FFT (mode='same')."""
    return _convolve(a, b)


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return scipy.signal.fftconvolve(a, b, mode="same")


def _cuda_convolve(
    a: np.ndarray,
    b_fft: "cp.ndarray",
    shape: tuple[int, ...],
    cpx,
) -> np.ndarray:
    a_dev = cp.asarray(a, dtype=cpx)
    a_fft = cufft.fftn(a_dev, shape, overwrite_x=True)
    a_fft *= b_fft
    result_dev = cufft.ifftn(a_fft, shape, overwrite_x=True)
    return cp.asnumpy(cp.abs(result_dev))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den != 0, num / den, 0.0)
