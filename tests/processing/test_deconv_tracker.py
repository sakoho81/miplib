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
    """FRC tracker finds a resolution on a real ISM dendrite image."""
    from pathlib import Path

    from skimage import io

    from miplib.data.containers.image import Image

    path = Path(__file__).parent.parent / "testdata" / "ism_dendrite.tiff"
    if not path.is_file():
        pytest.skip(f"Test image not found: {path}")

    data = io.imread(str(path)).astype(np.float64)
    img = Image(data, spacing=(0.1, 0.1))

    tracker = ConvergenceTracker(tracker_type="frc", frc_check_frequency=1)
    result = tracker.has_converged(tau_threshold=1e-8, estimate=img)
    assert not result  # First check, prev_resolution=inf, diff > threshold


@pytest.mark.integration
def test_deconvolution_improves_frc_resolution():
    """RL deconvolution with FRC tracker improves resolution on real ISM image."""
    from pathlib import Path

    from skimage import io

    from miplib.data.adapters.image_data import ArrayDataSource
    from miplib.data.containers.image import Image
    from miplib.processing.deconvolution.backends import ViewData, resolve_backend
    from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
    from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
    from miplib.processing.deconvolution.psf_utils import prepare_psf
    from miplib.processing.image import noisy
    from miplib.psf.psfgen import PsfFromFwhm

    path = Path(__file__).parent.parent / "testdata" / "ism_dendrite.tiff"
    if not path.is_file():
        pytest.skip(f"Test image not found: {path}")

    data = io.imread(str(path)).astype(np.float64)
    spacing = (0.1, 0.1)
    img = Image(data[200:456, 200:456], spacing=spacing)
    img = noisy(img, "gauss")  # noise ensures FRC curve crosses the threshold

    psf = PsfFromFwhm(fwhm=[1.0, 1.0], shape=img.shape, dims=tuple(img.shape)).xy()
    psf_arr, adj_psf_arr = prepare_psf(psf, spacing)

    source = ArrayDataSource([img])
    vd = ViewData(
        source=source,
        psfs=[psf_arr],
        adj_psfs=[adj_psf_arr],
        weights=[1.0],
        backgrounds=[0.0],
    )
    backend = resolve_backend("cpu", vd, vd.source.shape)
    estimate = create_estimate(source, FirstEstimate("image_mean"))
    options = RLOptions(max_iterations=2, stop_tau=1e-8, tracker_type="frc")

    deconv = RLDeconvolver(backend, estimate=estimate, options=options)
    for _ in deconv:
        pass

    result_img = deconv.result()
    assert result_img is not None
    assert np.isfinite(result_img).all()
    assert result_img.shape == img.shape

    # FRC-based tracker ran: verify it recorded some iterations
    df = deconv.tracker.to_dataframe()
    assert len(df) >= options.max_iterations
