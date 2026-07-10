from __future__ import annotations

import numpy as np
import pytest

from miplib.processing.deconvolution.tracker import Tau1ConvergenceTracker


def test_add_and_retrieve():
    tracker = Tau1ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.5, leak=0.01, e=1.0, s=0.0, u=0.0, n=0.0)
    tracker.add(t=1.0, tau1=0.3, leak=0.005, e=2.0, s=1.0, u=0.5, n=0.1)

    assert tracker.current_tau1 == pytest.approx(0.3)


def test_has_converged():
    tracker = Tau1ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.1, leak=0.0, e=0.0, s=0.0, u=0.0, n=0.0)
    assert tracker.has_converged(0.2)
    assert tracker.has_converged(0.1)
    assert not tracker.has_converged(0.05)


def test_has_converged_no_data():
    tracker = Tau1ConvergenceTracker()
    assert not tracker.has_converged(0.0)


def test_to_dataframe_columns():
    tracker = Tau1ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.5, leak=0.01, e=1.0, s=2.0, u=3.0, n=4.0)
    tracker.add(t=1.0, tau1=0.3, leak=0.005, e=5.0, s=6.0, u=7.0, n=8.0)

    df = tracker.to_dataframe()
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]
    assert len(df) == 2
    assert df.iloc[0]["tau1"] == pytest.approx(0.5)
    assert df.iloc[1]["n"] == pytest.approx(8.0)


def test_time_is_monotonic():
    tracker = Tau1ConvergenceTracker()
    times = [0.0, 0.5, 1.2, 2.1]
    for t in times:
        tracker.add(t=t, tau1=1.0 / (t + 1), leak=0.0, e=0.0, s=0.0, u=0.0, n=0.0)

    df = tracker.to_dataframe()
    assert np.all(np.diff(df["t"]) > 0)
