from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.deconvolver import RLDeconvolver


def _gaussian_psf_img(shape, sigma=2.0):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    data = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    return Image(data, spacing=(1.0, 1.0))


def _blur(img, psf):
    return scipy.signal.fftconvolve(img, psf, mode="same")


@pytest.fixture
def small_obj():
    shape = (32, 32)
    rng = np.random.default_rng(42)
    obj = rng.normal(loc=10, scale=2, size=shape).astype(np.float64)
    obj = np.abs(obj)
    return obj


@pytest.fixture
def small_img(small_obj):
    psf_data = _gaussian_psf_img((32, 32), sigma=1.5).view(np.ndarray)
    psf_data /= psf_data.sum()
    blurred = _blur(small_obj, psf_data)
    return Image(blurred, spacing=(1.0, 1.0))


@pytest.fixture
def small_psf():
    return _gaussian_psf_img((32, 32), sigma=1.5)


def test_manual_step_loop(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=5,
        stop_tau=0.0,
        verbose=False,
    )

    for _ in range(5):
        converged = deconv.step()
    assert converged

    result = deconv.result
    assert result.shape == small_img.shape
    assert result.min() >= 0
    assert result.max() > 0


def test_iteration_protocol(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=5,
        verbose=False,
    )

    results = list(deconv)
    assert len(results) == 5
    assert deconv.result.shape == small_img.shape


def test_for_loop_syntax(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=3,
        stop_tau=1.0,
        verbose=False,
    )

    count = 0
    for _ in deconv:
        count += 1

    assert 1 <= count <= 3


def test_run_convenience(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=5,
        stop_tau=1.0,
        verbose=False,
    )
    result = deconv.run()
    assert result.shape == small_img.shape
    assert len(deconv.tracker.to_dataframe()) >= 1


def test_result_before_iteration(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=5,
        verbose=False,
    )
    result = deconv.result
    assert result.shape == small_img.shape


def test_multi_view_identity(small_obj, small_psf):
    psf_data = small_psf.view(np.ndarray) / small_psf.sum()
    blurred = _blur(small_obj, psf_data)
    img = Image(blurred, spacing=(1.0, 1.0))

    deconv1 = RLDeconvolver(
        [img],
        [small_psf],
        max_iterations=3,
        stop_tau=0.0,
        verbose=False,
    )
    deconv3 = RLDeconvolver(
        [img, img, img],
        [small_psf, small_psf, small_psf],
        max_iterations=3,
        stop_tau=0.0,
        verbose=False,
    )

    for _ in deconv1:
        pass
    for _ in deconv3:
        pass

    r1 = deconv1.result
    r3 = deconv3.result
    np.testing.assert_allclose(r1, r3, rtol=1e-5)


def test_progress_tracking(small_img, small_psf):
    deconv = RLDeconvolver(
        [small_img],
        [small_psf],
        max_iterations=3,
        stop_tau=0.0,
        verbose=False,
    )

    for _ in deconv:
        pass

    df = deconv.tracker.to_dataframe()
    assert len(df) == 3
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]


def test_first_estimates(small_img, small_psf):
    for fe in ("constant", "image", "image_mean"):
        deconv = RLDeconvolver(
            [small_img],
            [small_psf],
            first_estimate=fe,
            max_iterations=1,
            stop_tau=0.0,
            verbose=False,
        )
        deconv.step()
        assert deconv.result.min() >= 0


def test_invalid_backend():
    img = Image(np.ones((8, 8), dtype=np.float64), spacing=(1.0, 1.0))
    psf = _gaussian_psf_img((8, 8))
    with pytest.raises(ValueError, match="Unknown backend"):
        RLDeconvolver([img], [psf], backend="invalid")


def test_mismatched_lengths():
    img = Image(np.ones((8, 8), dtype=np.float64), spacing=(1.0, 1.0))
    psf = _gaussian_psf_img((8, 8))
    with pytest.raises(ValueError, match="same length"):
        RLDeconvolver([img, img], [psf])
