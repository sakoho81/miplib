from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import CPUBackend, CUDABackend
from miplib.processing.deconvolution.blocks import extract_padded_block, iter_blocks
from miplib.processing.deconvolution.tracker import ConvergenceTracker
from miplib.processing.deconvolution.types import FusionMode

logger = logging.getLogger(__name__)


@dataclass
class RLOptions:
    """Algorithm parameters for Richardson-Lucy deconvolution / fusion."""

    fusion_mode: FusionMode = FusionMode.SUMMATIVE
    n_blocks: int = 1
    block_pad: int = 0
    epsilon: float = 1e-7
    tv_lambda: float = 0.0
    max_iterations: int = 100
    stop_tau: float = 1e-4
    tracker_type: str = "tau1"

    def __post_init__(self):
        if not isinstance(self.fusion_mode, FusionMode):
            self.fusion_mode = FusionMode(self.fusion_mode)
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.epsilon < 0 or self.epsilon > 0.5:
            raise ValueError("epsilon must be between 0 and 0.5")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")


class RLDeconvolver:
    """Richardson-Lucy deconvolution and multi-view fusion."""

    def __init__(
        self,
        backend: CPUBackend | CUDABackend,
        estimate: np.ndarray,
        *,
        options: RLOptions | None = None,
    ) -> None:
        self._backend = backend
        self._options = options or RLOptions()

        vd = backend._vd
        if estimate.shape != vd.source.shape:
            raise ValueError(
                f"estimate shape {estimate.shape} does not match "
                f"source shape {vd.source.shape}"
            )
        self._estimate = estimate
        self._estimate_new = np.zeros_like(estimate)
        self._shape = vd.source.shape
        self.tracker = ConvergenceTracker(
            tracker_type=self._options.tracker_type,
        )
        self._iteration: int = 0
        self._converged: bool = False

    # ------------------------------------------------------------------
    # iteration protocol
    # ------------------------------------------------------------------

    def step(self) -> bool:
        """Execute one RL iteration. Returns True if converged."""
        if self._converged:
            return True
        if self._iteration >= self._options.max_iterations:
            return True

        prev = self._estimate.copy()
        t0 = time.perf_counter()
        self._estimate_new.fill(0)

        vd = self._backend._vd
        opts = self._options
        e_tot = s_tot = u_tot = n_tot = 0.0

        for block in iter_blocks(self._shape, opts.n_blocks, pad=opts.block_pad):
            est_block = extract_padded_block(self._estimate, block)
            img_blocks = [
                vd.source.get_image_block(v, block) for v in range(vd.n_views)
            ]

            new_block, e, s, u, n = self._backend.compute_block(
                est_block, img_blocks, opts
            )

            self._estimate_new[block.inner_slice] = new_block[block.inner_slice_local]
            e_tot += e
            s_tot += s
            u_tot += u
            n_tot += n

        self._estimate[:] = self._estimate_new

        elapsed = time.perf_counter() - t0
        self.__record_step(prev, elapsed, e_tot, s_tot, u_tot, n_tot)
        self._iteration += 1

        self._converged = (
            self._iteration >= self._options.max_iterations
            or self.tracker.has_converged(
                self._options.stop_tau,
                Image(self._estimate, list(vd.source.spacing)),
            )
        )
        return self._converged

    def __iter__(self) -> "RLDeconvolver":
        return self

    def __next__(self) -> Image:
        if self._converged:
            raise StopIteration
        self.step()
        vd = self._backend._vd
        return Image(self._estimate, list(vd.source.spacing))

    def __len__(self) -> int:
        return self._options.max_iterations - self._iteration

    def run(self) -> Image:
        for _ in self:
            pass
        return self.result()

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    def result(self) -> Image:
        vd = self._backend._vd
        return Image(self._estimate.copy(), list(vd.source.spacing))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def __compute_tau1(current: np.ndarray, previous: np.ndarray) -> float:
        diff = np.abs(current - previous)
        denom = np.abs(previous)
        denom[denom == 0] = 1e-30
        return float(np.max(diff / denom))

    def __record_step(
        self,
        previous: np.ndarray,
        elapsed: float,
        e: float,
        s: float,
        u: float,
        n: float,
    ) -> None:
        total = self._estimate.sum()
        leak = (total - previous.sum()) / (total + 1e-30)
        tau1 = self.__compute_tau1(self._estimate, previous)
        self.tracker.add(elapsed, tau1, leak, e, s, u, n)
