from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import (
    _CUDA_AVAILABLE,
    CPUBackend,
    CUDABackend,
    ViewData,
    resolve_backend,
)
from miplib.processing.deconvolution.blocks import extract_padded_block, iter_blocks
from miplib.processing.deconvolution.deconvolver import FusionMode
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate


def _make_view_data(n_views=1, shape=(32, 32)):
    rng = np.random.default_rng(42)

    def _psf(sigma=1.5):
        coords = [np.arange(s) - s // 2 for s in shape]
        grid = np.meshgrid(*coords, indexing="ij")
        r_sq = sum(g**2 for g in grid)
        p = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
        p /= p.sum()
        return p

    def _blur(img, psf):
        import scipy.signal

        return scipy.signal.fftconvolve(img, psf, mode="same")

    images = []
    psfs = []
    adj_psfs = []

    for i in range(n_views):
        obj = rng.normal(loc=10, scale=2, size=shape).astype(np.float64)
        obj = np.abs(obj)
        psf = _psf(sigma=1.5 + i * 0.5)
        blurred = _blur(obj, psf)
        images.append(Image(blurred, spacing=(1.0, 1.0)))
        psfs.append(psf)
        adj_psfs.append(psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))])

    source = ArrayDataSource(images)
    return ViewData(
        source=source,
        psfs=psfs,
        adj_psfs=adj_psfs,
        weights=[1.0] * n_views,
        backgrounds=[0.0] * n_views,
    )


class FakeOptions:
    fusion_mode = FusionMode.SUMMATIVE
    epsilon = 0.1
    tv_lambda = 0.0
    n_blocks = 1
    block_pad = 0
    max_iterations = 10
    stop_tau = 0.0
    first_estimate = FirstEstimate.IMAGE_MEAN
    estimate_constant = 1.0
    virtual_psf = False
    backend = "cpu"


def test_view_data_n_views():
    vd = _make_view_data(n_views=3)
    assert vd.n_views == 3


def test_cpu_backend_compute_block_single():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    opts = FakeOptions()

    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]

    new_block, e, s, u, n = backend.compute_block(est_block, img_blocks, opts)
    assert new_block.shape == est_block.shape
    assert new_block.min() >= 0
    assert e >= 0


def test_cpu_backend_compute_block_multi():
    vd = _make_view_data(n_views=3)
    backend = CPUBackend(vd)
    opts = FakeOptions()

    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]

    new_block, e, s, u, n = backend.compute_block(est_block, img_blocks, opts)
    assert new_block.shape == est_block.shape


def test_cpu_backend_multiplicative():
    vd = _make_view_data(n_views=2)
    backend = CPUBackend(vd)
    opts = FakeOptions()
    opts.fusion_mode = FusionMode.MULTIPLICATIVE

    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]

    new_block, *_ = backend.compute_block(est_block, img_blocks, opts)
    assert new_block.shape == est_block.shape


def test_resolve_backend_cpu():
    vd = _make_view_data(n_views=1)
    backend = resolve_backend("cpu", vd, vd.source.shape)
    assert isinstance(backend, CPUBackend)


def test_resolve_backend_invalid():
    vd = _make_view_data(n_views=1)
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("invalid", vd, vd.source.shape)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_resolve_backend_cuda():
    vd = _make_view_data(n_views=1)
    backend = resolve_backend("cuda", vd, vd.source.shape)
    assert isinstance(backend, CUDABackend)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_cuda_backend_compute_block():
    vd = _make_view_data(n_views=1)
    backend = CUDABackend(vd, vd.source.shape)
    opts = FakeOptions()

    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]

    new_block, *_ = backend.compute_block(est_block, img_blocks, opts)
    assert new_block.shape == est_block.shape
    assert new_block.min() >= 0


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_cuda_matches_cpu():
    vd = _make_view_data(n_views=1)
    cpu = CPUBackend(vd)
    cuda = CUDABackend(vd, vd.source.shape)
    opts = FakeOptions()

    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]

    cpu_result, *_ = cpu.compute_block(est_block, img_blocks, opts)
    cuda_result, *_ = cuda.compute_block(est_block, img_blocks, opts)

    npt.assert_allclose(cpu_result, cuda_result, rtol=1e-3)
