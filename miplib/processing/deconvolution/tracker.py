from __future__ import annotations

import pandas as pd


class RLConvergenceTracker:
    """Accumulates and analyses Richardson-Lucy convergence statistics."""

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

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows, columns=list(self.COLUMNS))
