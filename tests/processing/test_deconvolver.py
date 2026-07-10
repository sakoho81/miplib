from __future__ import annotations

import numpy as np
import pytest
import scipy.signal

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import CPUBackend, ViewData
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate


def _gaussian_psf_img(shape, sigma=2.0):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    data = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    return Image(data, spacing=(1.0, 1.0))


def _blur(img, psf):
    return scipy.signal.fftconvolve(img, psf, mode="same")


def _psf_data(shape, sigma=1.5):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    p = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    p /= p.sum()
    return p


def _view_data(n_views=1, shape=(32, 32), seed=42):
    rng = np.random.default_rng(seed)
    images = []
    psfs = []
    adjs = []
    for i in range(n_views):
        obj = np.abs(rng.normal(loc=10, scale=2, size=shape).astype(np.float64))
        psf = _psf_data(shape, sigma=1.5 + i * 0.5)
        blurred = _blur(obj, psf)
        images.append(Image(blurred, spacing=(1.0, 1.0)))
        psfs.append(psf)
        adjs.append(psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))])
    source = ArrayDataSource(images)
    return ViewData(
        source=source,
        psfs=psfs,
        adj_psfs=adjs,
        weights=[1.0] * n_views,
        backgrounds=[0.0] * n_views,
    )


def test_manual_step_loop():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5, stop_tau=0.0)
    )

    for _ in range(5):
        converged = deconv.step()
    assert converged

    result = deconv.result
    assert result.shape == vd.source.shape
    assert result.min() >= 0
    assert result.max() > 0


def test_iteration_protocol():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5)
    )

    results = list(deconv)
    assert len(results) == 5


def test_for_loop_syntax():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=3, stop_tau=1.0)
    )

    count = 0
    for _ in deconv:
        count += 1
    assert 1 <= count <= 3


def test_run_convenience():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5, stop_tau=1.0)
    )
    result = deconv.run()
    assert result.shape == vd.source.shape
    assert len(deconv.tracker.to_dataframe()) >= 1


def test_result_before_iteration():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5)
    )
    result = deconv.result
    assert result.shape == vd.source.shape


def test_multi_view_identity():
    shape = (32, 32)
    rng = np.random.default_rng(42)
    obj = np.abs(rng.normal(loc=10, scale=2, size=shape).astype(np.float64))
    psf = _psf_data(shape, sigma=1.5)
    adj = psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))]
    blurred = _blur(obj, psf)
    img = Image(blurred, spacing=(1.0, 1.0))

    vd1 = ViewData(
        source=ArrayDataSource([img]),
        psfs=[psf],
        adj_psfs=[adj],
        weights=[1.0],
        backgrounds=[0.0],
    )
    vd3 = ViewData(
        source=ArrayDataSource([img, img, img]),
        psfs=[psf, psf, psf],
        adj_psfs=[adj, adj, adj],
        weights=[1.0, 1.0, 1.0],
        backgrounds=[0.0, 0.0, 0.0],
    )

    backend1 = CPUBackend(vd1)
    backend3 = CPUBackend(vd3)
    opts = RLOptions(max_iterations=3, stop_tau=0.0)

    est1 = create_estimate(vd1.source, FirstEstimate.IMAGE_MEAN)
    est3 = create_estimate(vd3.source, FirstEstimate.IMAGE_MEAN)
    deconv1 = RLDeconvolver(backend1, estimate=est1, options=opts)
    deconv3 = RLDeconvolver(backend3, estimate=est3, options=opts)

    for _ in deconv1:
        pass
    for _ in deconv3:
        pass

    np.testing.assert_allclose(deconv1.result, deconv3.result, rtol=1e-5)


def test_progress_tracking():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=3, stop_tau=0.0)
    )

    for _ in deconv:
        pass

    df = deconv.tracker.to_dataframe()
    assert len(df) == 3
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]


def test_external_estimate_array():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = np.zeros(vd.source.shape, dtype=np.float32)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=1)
    )
    deconv.step()


def test_estimate_shape_mismatch():
    vd = _view_data(n_views=1)
    backend = CPUBackend(vd)
    wrong = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="estimate shape"):
        RLDeconvolver(backend, estimate=wrong)


def test_rloptions_validation():
    with pytest.raises(ValueError, match="fusion_mode"):
        RLOptions(fusion_mode="invalid")
    with pytest.raises(ValueError, match="n_blocks"):
        RLOptions(n_blocks=0)
    with pytest.raises(ValueError, match="epsilon"):
        RLOptions(epsilon=-0.1)
    with pytest.raises(ValueError, match="max_iterations"):
        RLOptions(max_iterations=0)
