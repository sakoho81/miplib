from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import CPUBackend, CUDABackend
from miplib.processing.deconvolution.blocks import extract_padded_block, iter_blocks
from miplib.processing.deconvolution.tracker import RLConvergenceTracker

logger = logging.getLogger(__name__)


@dataclass
class RLOptions:
    """Algorithm parameters for Richardson-Lucy deconvolution / fusion."""

    fusion_mode: str = "summative"
    n_blocks: int = 1
    block_pad: int = 0
    epsilon: float = 1e-7
    tv_lambda: float = 0.0
    max_iterations: int = 100
    stop_tau: float = 1e-4

    def __post_init__(self):
        if self.fusion_mode not in ("summative", "multiplicative"):
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode!r}")
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if self.epsilon < 0 or self.epsilon > 0.5:
            raise ValueError("epsilon must be between 0 and 0.5")
        if self.max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {self.max_iterations}")


class RLDeconvolver:
    """Richardson-Lucy deconvolution and multi-view fusion.

    Takes a backend instance (``CPUBackend`` or ``CUDABackend``) and an
    optional pre-allocated estimate array.  The backend owns all per-view
    data through ``ViewData``.
    """

    def __init__(
        self,
        backend: CPUBackend | CUDABackend,
        *,
        estimate: np.ndarray | None = None,
        options: RLOptions | None = None,
    ) -> None:
        self._backend = backend
        self._options = options or RLOptions()

        vd = backend._vd
        if estimate is None:
            self._estimate = np.zeros(vd.source.shape, dtype=np.float32)
        else:
            if estimate.shape != vd.source.shape:
                raise ValueError(
                    f"estimate shape {estimate.shape} does not match "
                    f"source shape {vd.source.shape}"
                )
            self._estimate = estimate

        self._estimate_new = np.zeros_like(self._estimate)
        self._shape = vd.source.shape
        self.tracker = RLConvergenceTracker()
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

            p = block.pad
            ndim = len(self._shape)
            write_slice = tuple(
                slice(p, new_block.shape[ax] - p if p > 0 else None)
                for ax in range(ndim)
            )
            self._estimate_new[block.inner_slice] = new_block[write_slice]
            e_tot += e
            s_tot += s
            u_tot += u
            n_tot += n

        self._estimate[:] = self._estimate_new

        elapsed = time.perf_counter() - t0
        _record_step(
            self.tracker, elapsed, self._estimate, prev, e_tot, s_tot, u_tot, n_tot
        )
        self._iteration += 1

        self._converged = (
            self._iteration >= self._options.max_iterations
            or self.tracker.has_converged(self._options.stop_tau)
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
        return self._options.max_iterations - self._iteration

    def run(self) -> Image:
        for _ in self:
            pass
        return self.result

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    @property
    def result(self) -> Image:
        vd = self._backend._vd
        return Image(self._estimate.copy(), list(vd.source.spacing))


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _record_step(
    tracker: RLConvergenceTracker,
    elapsed: float,
    current: np.ndarray,
    previous: np.ndarray,
    e: float,
    s: float,
    u: float,
    n: float,
) -> None:
    total = current.sum()
    leak = (total - previous.sum()) / (total + 1e-30)
    tau1 = _compute_tau1(current, previous)
    tracker.add(elapsed, tau1, leak, e, s, u, n)


def _compute_tau1(current: np.ndarray, previous: np.ndarray) -> float:
    diff = np.abs(current - previous)
    denom = np.abs(previous)
    denom[denom == 0] = 1e-30
    return float(np.max(diff / denom))
