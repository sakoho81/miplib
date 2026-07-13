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
    convolve_cpu,
    resolve_backend,
)
from miplib.processing.deconvolution.blocks import extract_padded_block, iter_blocks
from miplib.processing.deconvolution.deconvolver import FusionMode
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psf
from tests.conftest import gaussian_spot


def _make_view_data(n_views=1, shape=(32, 32)):
    rng = np.random.default_rng(42)

    images: list[Image] = []
    psfs: list[np.ndarray] = []
    adj_psfs: list[np.ndarray] = []

    for i in range(n_views):
        obj = np.abs(rng.normal(loc=10, scale=2, size=shape).astype(np.float64))
        psf_img = Image(gaussian_spot(shape, sigma=1.5 + i * 0.5), spacing=(1.0, 1.0))
        psf, adj = prepare_psf(psf_img, psf_img.spacing)
        blurred = convolve_cpu(obj, psf)
        images.append(Image(blurred, spacing=(1.0, 1.0)))
        psfs.append(psf)
        adj_psfs.append(adj)

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


def _run_block(backend, vd, opts):
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    block = next(iter_blocks(estimate.shape, n_blocks=1, pad=0))
    est_block = extract_padded_block(estimate, block)
    img_blocks = [vd.source.get_image_block(v, block) for v in range(vd.n_views)]
    return backend.compute_block(est_block, img_blocks, opts)


def test_cpu_backend_compute_block_single():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    opts = FakeOptions()

    new_block, e, s, u, n = _run_block(backend, vd, opts)
    assert new_block.shape == (32, 32)
    assert new_block.min() >= 0
    assert e >= 0
    # Deconvolution should produce output similar in structure to input
    assert np.abs(new_block.sum() - vd.source.get_full_image(0).sum()) < 1000


def test_cpu_backend_compute_block_multi():
    vd = _make_view_data(n_views=3)
    backend = CPUBackend(vd)
    opts = FakeOptions()

    new_block, e, s, u, n = _run_block(backend, vd, opts)
    assert new_block.shape == (32, 32)
    assert e >= 0
    assert s >= 0


def test_cpu_backend_multiplicative():
    vd = _make_view_data(n_views=2)
    backend = CPUBackend(vd)
    opts = FakeOptions()
    opts.fusion_mode = FusionMode.MULTIPLICATIVE

    new_block, *_ = _run_block(backend, vd, opts)
    assert new_block.shape == (32, 32)


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

    new_block, *_ = _run_block(backend, vd, opts)
    assert new_block.shape == (32, 32)
    assert new_block.min() >= 0


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_cuda_matches_cpu():
    vd = _make_view_data(n_views=1)
    cpu = CPUBackend(vd)
    cuda = CUDABackend(vd, vd.source.shape)
    opts = FakeOptions()

    cpu_result, *_ = _run_block(cpu, vd, opts)
    cuda_result, *_ = _run_block(cuda, vd, opts)

    npt.assert_allclose(cpu_result, cuda_result, rtol=1e-3)
