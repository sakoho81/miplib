from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from miplib.data.containers.image import Image
from miplib.processing import ops_ext
from miplib.processing.deconvolution.backends import (
    _CUDA_AVAILABLE,
    convolve_cpu,
    make_cuda_psf_convolve,
)
from miplib.processing.deconvolution.blocks import (
    calculate_block_layout,
    extract_padded_block,
)
from miplib.processing.deconvolution.kernels import rl_multi_view
from miplib.processing.deconvolution.psf_utils import (
    compute_virtual_psfs,
    prepare_psfs,
)
from miplib.processing.deconvolution.tracker import RLConvergenceTracker
from miplib.processing.ndarray import nroot

logger = logging.getLogger(__name__)


class RLDeconvolver:
    """Richardson-Lucy deconvolution and multi-view fusion.

    Single-view deconvolution is the special case with one view
    (weight=1, background=0). Multi-view fusion uses multiple views
    with per-view weights and backgrounds.
    """

    def __init__(
        self,
        images: list[Image],
        psfs: list[Image],
        *,
        weights: list[float] | None = None,
        backgrounds: list[float] | None = None,
        fusion_mode: str = "summative",
        backend: str = "cpu",
        n_blocks: int = 1,
        block_pad: int = 0,
        epsilon: float = 1e-7,
        tv_lambda: float = 0.0,
        max_iterations: int = 100,
        stop_tau: float = 1e-4,
        first_estimate: str = "image",
        estimate_constant: float = 1.0,
        virtual_psf: bool = False,
        memmap_estimates: bool = False,
        verbose: bool = False,
    ) -> None:
        if backend not in ("cpu", "cuda"):
            raise ValueError(f"Unknown backend: {backend!r}")
        if backend == "cuda" and not _CUDA_AVAILABLE:
            raise RuntimeError("cupy is not installed")

        n_views = len(images)
        if len(psfs) != n_views:
            raise ValueError("images and psfs must have the same length")

        self._n_views = n_views
        self._fusion_mode = fusion_mode
        self._backend = backend
        self._epsilon = epsilon
        self._tv_lambda = tv_lambda
        self._max_iterations = max_iterations
        self._stop_tau = stop_tau
        self._verbose = verbose
        self._image_spacing = images[0].spacing

        if weights is None:
            weights = [1.0] * n_views
        if backgrounds is None:
            backgrounds = [0.0] * n_views

        self._norms, self._adjs = prepare_psfs(psfs, self._image_spacing)
        if virtual_psf and n_views >= 2:
            self._adjs = compute_virtual_psfs(self._norms, self._adjs)

        self._images: list[np.ndarray] = [img.view(np.ndarray) for img in images]
        self._weights = weights
        self._backgrounds = backgrounds

        self._shape = self._images[0].shape
        self._blocks = calculate_block_layout(self._shape, n_blocks, pad=block_pad)
        self._block_pad = block_pad

        self._tmpdir: tempfile.TemporaryDirectory | None = None
        if memmap_estimates:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._estimate = np.memmap(
                Path(self._tmpdir.name) / "estimate.dat",
                dtype=np.float32,
                mode="w+",
                shape=self._shape,
            )
            self._estimate_new = np.memmap(
                Path(self._tmpdir.name) / "estimate_new.dat",
                dtype=np.float32,
                mode="w+",
                shape=self._shape,
            )
            self._estimate[:] = 0
            self._estimate_new[:] = 0
        else:
            self._estimate = np.zeros(self._shape, dtype=np.float32)
            self._estimate_new = np.zeros(self._shape, dtype=np.float32)

        self._prev_estimate: np.ndarray | None = None
        self._init_estimate(first_estimate, estimate_constant)
        self._setup_backend()

        self.tracker = RLConvergenceTracker()
        self._iteration: int = 0
        self._converged: bool = False

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------

    def _init_estimate(self, first_estimate: str, constant: float) -> None:
        if first_estimate == "constant":
            self._estimate[:] = constant
        elif first_estimate == "image":
            self._estimate[:] = self._images[0]
        elif first_estimate == "image_mean":
            self._estimate[:] = self._images[0].mean()
        elif first_estimate == "average":
            self._estimate[:] = np.mean(self._images, axis=0)
        elif first_estimate == "sum":
            self._estimate[:] = np.sum(self._images, axis=0)
        else:
            raise ValueError(f"Unknown first_estimate: {first_estimate!r}")

    def _setup_backend(self) -> None:
        if self._backend == "cuda":
            self._psf_convolves = [
                make_cuda_psf_convolve(p, self._shape) for p in self._norms
            ]
            self._adj_convolves = [
                make_cuda_psf_convolve(a, self._shape) for a in self._adjs
            ]
            self._convolve_fn = None
        else:
            self._convolve_fn = convolve_cpu
            self._psf_convolves = None
            self._adj_convolves = None

    # ------------------------------------------------------------------
    # iteration protocol
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """Execute one RL iteration. Returns True if converged."""
        if self._converged:
            return True
        if self._iteration >= self._max_iterations:
            return True

        self._prev_estimate = self._estimate.copy()
        t0 = time.perf_counter()

        if self._backend == "cuda":
            assert self._psf_convolves is not None
            assert self._adj_convolves is not None
            self._estimate_new, e, s, u, n = self._compute_step_cuda()
        else:
            self._estimate_new, e, s, u, n = self._compute_step_cpu()

        self._estimate[:] = self._estimate_new

        elapsed = time.perf_counter() - t0
        total_photons = self._estimate.sum()
        leak = (total_photons - self._prev_estimate.sum()) / (total_photons + 1e-30)
        tau1 = _compute_tau1(self._estimate, self._prev_estimate)

        self.tracker.add(elapsed, tau1, leak, e, s, u, n)
        self._iteration += 1

        if self._verbose:
            logger.info(
                "iter %02d  tau1=%.4f  leak=%.4e  (e=%.0f s=%.0f u=%.0f n=%.0f)",
                self._iteration,
                tau1,
                leak,
                e,
                s,
                u,
                n,
            )

        self._converged = (
            self._iteration >= self._max_iterations
            or self.tracker.has_converged(self._stop_tau)
        )
        return self._converged

    def __iter__(self) -> "RLDeconvolver":
        return self

    def __next__(self) -> Image:
        if self._converged:
            raise StopIteration
        self.step()
        return self.result

    def __len__(self) -> int:
        return self._max_iterations - self._iteration

    def _compute_step_cpu(
        self,
    ) -> tuple[np.ndarray, float, float, float, float]:
        assert self._convolve_fn is not None
        estimate_new = np.zeros_like(self._estimate)
        e_total = s_total = u_total = n_total = 0.0

        for block in self._blocks:
            est_block = extract_padded_block(self._estimate, block)
            img_blocks = [extract_padded_block(img, block) for img in self._images]

            new_block, e, s, u, n = rl_multi_view(
                est_block,
                img_blocks,
                self._norms,
                self._adjs,
                weights=self._weights,
                backgrounds=self._backgrounds,
                fusion_mode=self._fusion_mode,
                convolve=self._convolve_fn,
                tv_lambda=self._tv_lambda,
                epsilon=self._epsilon,
            )

            p = block.pad
            inner = tuple(slice(p, p + s) for s in block.inner_size)
            estimate_new[block.inner_slice] = new_block[inner]
            e_total += e
            s_total += s
            u_total += u
            n_total += n

        return estimate_new, e_total, s_total, u_total, n_total

    def _compute_step_cuda(
        self,
    ) -> tuple[np.ndarray, float, float, float, float]:
        assert self._psf_convolves is not None
        assert self._adj_convolves is not None
        estimate_new = np.zeros_like(self._estimate)
        e_total = s_total = u_total = n_total = 0.0

        for block in self._blocks:
            est_block = extract_padded_block(self._estimate, block)
            img_blocks = [extract_padded_block(img, block) for img in self._images]

            if self._fusion_mode == "summative":
                correction = np.zeros_like(est_block)
                for img, psf_conv, adj_conv, w, bg in zip(
                    img_blocks,
                    self._psf_convolves,
                    self._adj_convolves,
                    self._weights,
                    self._backgrounds,
                ):
                    forward = psf_conv(est_block) * w + bg
                    ratio = _safe_divide(img, forward)
                    correction += adj_conv(ratio)
                correction /= self._n_views
            else:
                correction = np.ones_like(est_block)
                for img, psf_conv, adj_conv, w, bg in zip(
                    img_blocks,
                    self._psf_convolves,
                    self._adj_convolves,
                    self._weights,
                    self._backgrounds,
                ):
                    forward = psf_conv(est_block) * w + bg
                    ratio = _safe_divide(img, forward)
                    correction *= adj_conv(ratio)
                correction = nroot(correction, self._n_views)

            if self._tv_lambda > 0:
                correction += self._tv_lambda * ops_ext.div_unit_grad(
                    est_block, (1.0,) * est_block.ndim
                )

            new_block = est_block.copy()
            e, s, u, n = ops_ext.update_estimate_poisson(
                new_block, correction, self._epsilon
            )

            p = block.pad
            inner = tuple(slice(p, p + s) for s in block.inner_size)
            estimate_new[block.inner_slice] = new_block[inner]
            e_total += e
            s_total += s
            u_total += u
            n_total += n

        return estimate_new, e_total, s_total, u_total, n_total

    def run(self) -> Image:
        """Run all remaining iterations. Returns final result."""
        for _ in self:
            pass
        return self.result

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    @property
    def result(self) -> Image:
        return Image(self._estimate.copy(), list(self._image_spacing))

    def close(self) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _compute_tau1(current: np.ndarray, previous: np.ndarray) -> float:
    diff = np.abs(current - previous)
    denom = np.abs(previous)
    denom[denom == 0] = 1e-30
    return float(np.max(diff / denom))


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        quotient = np.where(den != 0, num / den, 0.0)
    return quotient
