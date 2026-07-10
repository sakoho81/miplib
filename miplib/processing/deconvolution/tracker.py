from __future__ import annotations

import argparse
import logging

import pandas as pd

from miplib.data.containers.image import Image

logger = logging.getLogger(__name__)


class Tau1ConvergenceTracker:
    """Tau1-based convergence: stops when the maximum pixel-wise relative
    change between iterations falls below *stop_tau*."""

    COLUMNS = ("t", "tau1", "leak", "e", "s", "u", "n")

    def __init__(self) -> None:
        self._rows: list[tuple[float, ...]] = []
        self._current_tau1: float = float("inf")

    def add(
        self,
        t: float,
        tau1: float,
        leak: float,
        e: float,
        s: float,
        u: float,
        n: float,
    ) -> None:
        self._rows.append((t, tau1, leak, e, s, u, n))
        self._current_tau1 = tau1

    @property
    def current_tau1(self) -> float:
        return self._current_tau1

    def has_converged(self, tau_threshold: float) -> bool:
        return self._current_tau1 <= tau_threshold

    def check_estimate(self, estimate: Image) -> bool:
        """Optional per-iteration check against the current estimate.
        The Tau1 tracker does not inspect the estimate — always False."""
        return False

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=list(self.COLUMNS))


class FRCConvergenceTracker(Tau1ConvergenceTracker):
    """Tau1 + FRC-based convergence.

    On every *frc_check_frequency* iterations, computes single-image FRC
    resolution and stops when the resolution change between consecutive FRC
    checks drops below *stagnation_threshold*.
    """

    def __init__(
        self,
        *,
        frc_check_frequency: int = 10,
        stagnation_threshold: float = 0.001,
    ) -> None:
        super().__init__()
        self._freq = frc_check_frequency
        self._threshold = stagnation_threshold
        self._prev_resolution: float = float("inf")
        self._iteration: int = 0

    def check_estimate(self, estimate: Image) -> bool:
        self._iteration += 1
        if self._iteration % self._freq != 0:
            return False

        from miplib.analysis.resolution.fourier_ring_correlation import (
            calculate_single_image_frc,
        )

        args = _make_frc_args()
        result = calculate_single_image_frc(estimate, args)
        resolution = result.resolution["resolution"]
        diff = abs(self._prev_resolution - resolution)
        logger.debug(
            "FRC check: resolution=%.4f  prev=%.4f  diff=%.4f",
            resolution,
            self._prev_resolution,
            diff,
        )
        self._prev_resolution = resolution
        return diff <= self._threshold


def _make_frc_args() -> argparse.Namespace:
    return argparse.Namespace(
        d_bin=1,
        disable_hamming=False,
        frc_curve_fit_degree=8,
        frc_curve_fit_type="spline",
        resol_square=False,
        resolution_threshold_criterion="fixed",
        resolution_threshold_value=1.0 / 7,
        resolution_point_sigma=0.01,
        resolution_threshold_curve_fit_degree=3,
        d_angle=20,
        d_extract_angle=5.0,
        hollow_iterator=False,
        min_filter=False,
        resolution_snr_value=0.25,
        verbose=False,
    )
