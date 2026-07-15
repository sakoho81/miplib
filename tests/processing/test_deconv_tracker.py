from __future__ import annotations

import numpy as np
import pytest

from miplib.processing.deconvolution.tracker import ConvergenceTracker


def test_add_and_retrieve():
    tracker = ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.5, leak=0.01, e=1.0, s=0.0, u=0.0, n=0.0)
    tracker.add(t=1.0, tau1=0.3, leak=0.005, e=2.0, s=1.0, u=0.5, n=0.1)

    assert tracker.current_tau1 == pytest.approx(0.3)


def test_has_converged():
    tracker = ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.1, leak=0.0, e=0.0, s=0.0, u=0.0, n=0.0)
    assert tracker.has_converged(0.2)
    assert tracker.has_converged(0.1)
    assert not tracker.has_converged(0.05)


def test_has_converged_no_data():
    tracker = ConvergenceTracker()
    assert not tracker.has_converged(0.0)


def test_to_dataframe_columns():
    tracker = ConvergenceTracker()
    tracker.add(t=0.0, tau1=0.5, leak=0.01, e=1.0, s=2.0, u=3.0, n=4.0)
    tracker.add(t=1.0, tau1=0.3, leak=0.005, e=5.0, s=6.0, u=7.0, n=8.0)

    df = tracker.to_dataframe()
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]
    assert len(df) == 2
    assert df.iloc[0]["tau1"] == pytest.approx(0.5)
    assert df.iloc[1]["n"] == pytest.approx(8.0)


def test_time_is_monotonic():
    tracker = ConvergenceTracker()
    times = [0.0, 0.5, 1.2, 2.1]
    for t in times:
        tracker.add(t=t, tau1=1.0 / (t + 1), leak=0.0, e=0.0, s=0.0, u=0.0, n=0.0)

    df = tracker.to_dataframe()
    assert np.all(np.diff(df["t"]) > 0)


def test_has_converged_ignores_estimate_in_tau1_mode():
    tracker = ConvergenceTracker(tracker_type="tau1")
    tracker.add(t=0.0, tau1=0.1, leak=0.0, e=0.0, s=0.0, u=0.0, n=0.0)

    # With estimate but tau1 not met: still False
    assert not tracker.has_converged(0.05, estimate=None)
    # With tau1 met: True regardless of estimate
    assert tracker.has_converged(0.2, estimate=None)


def test_frc_tracker_does_not_crash():
    """Frc tracker survives when FRC curve never crosses the resolution threshold.

    Clean images produce checkerboard halves with correlated noise, so
    the FRC correlation may never drop below the fixed threshold of 1/7.
    The tracker must not crash — it should return False (not converged).
    """
    import skimage.data

    from miplib.data.containers.image import Image

    tracker = ConvergenceTracker(tracker_type="frc", frc_check_frequency=1)

    coins = skimage.data.coins().astype(np.float64)
    img = Image(coins, spacing=(1.0, 1.0))
    result = tracker.has_converged(tau_threshold=1e-8, estimate=img)
    assert result is False  # Clean image FRC never crosses threshold


@pytest.mark.integration
def test_frc_tracker_on_real_ism_image():
    """FRC tracker resolves and converges on the same image."""
    from pathlib import Path

    from skimage import io

    from miplib.data.containers.image import Image

    path = Path(__file__).parent.parent / "testdata" / "ism_dendrite.tiff"
    if not path.is_file():
        pytest.skip(f"Test image not found: {path}")

    data = io.imread(str(path)).astype(np.float64)
    img = Image(data, spacing=(0.1, 0.1))

    tracker = ConvergenceTracker(
        tracker_type="frc",
        frc_check_frequency=1,
        frc_stagnation_threshold=0.001,
    )

    # First call: resolution found, but prev_resolution=inf so not converged
    result = tracker.has_converged(tau_threshold=1e-8, estimate=img)
    assert not result
    assert tracker._prev_resolution == pytest.approx(0.56, abs=0.1)

    # Second call with same image: resolution unchanged → converged
    result = tracker.has_converged(tau_threshold=1e-8, estimate=img)
    assert result
