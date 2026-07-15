from __future__ import annotations

import logging

import pandas as pd

from miplib.data.containers.image import Image

logger = logging.getLogger(__name__)


class ConvergenceTracker:
    """Tracks RL convergence via tau1, optionally combined with FRC.

    Parameters
    ----------
    tracker_type : "tau1" | "frc"
    frc_check_frequency : int
        FRC resolution is recomputed every N iterations.
    frc_stagnation_threshold : float
        Stop when successive FRC resolution values differ by less than this.
    """

    COLUMNS = ("t", "tau1", "leak", "e", "s", "u", "n")

    def __init__(
        self,
        *,
        tracker_type: str = "tau1",
        frc_check_frequency: int = 10,
        frc_stagnation_threshold: float = 0.001,
    ) -> None:
        self._rows: list[tuple[float, ...]] = []
        self._current_tau1: float = float("inf")
        self._frc_enabled = tracker_type == "frc"
        self._freq = frc_check_frequency
        self._threshold = frc_stagnation_threshold
        self._prev_resolution: float = float("inf")
        self._frct_iteration: int = 0

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

    def has_converged(
        self, tau_threshold: float, estimate: Image | None = None
    ) -> bool:
        """Return True if tau1 < *tau_threshold* or (if FRC-enabled)
        the FRC resolution has stagnated."""
        if self._current_tau1 <= tau_threshold:
            return True
        if self._frc_enabled and estimate is not None:
            return self._check_frc(estimate)
        return False

    def _check_frc(self, estimate: Image) -> bool:
        self._frct_iteration += 1
        if self._frct_iteration % self._freq != 0:
            return False

        from miplib.analysis.resolution.fourier_ring_correlation import (
            FRCOptions,
            calculate_single_image_frc,
        )

        options = FRCOptions()
        result = calculate_single_image_frc(estimate, options)
        resolution = result.resolution["resolution"]
        if resolution is None:
            return False
        diff = abs(self._prev_resolution - resolution)
        diff = abs(self._prev_resolution - resolution)
        logger.debug(
            "FRC check: resolution=%.4f  prev=%.4f  diff=%.4f",
            resolution,
            self._prev_resolution,
            diff,
        )
        self._prev_resolution = resolution
        return diff <= self._threshold

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=list(self.COLUMNS))
