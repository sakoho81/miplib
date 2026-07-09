from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.signal

from miplib.processing.deconvolution.backends import (
    _CUDA_AVAILABLE,
    convolve_cpu,
    make_cuda_convolve,
    make_cuda_psf_convolve,
)


def _random_array(shape, dtype=np.float64):
    rng = np.random.default_rng(123)
    return rng.normal(size=shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_convolve_cpu_matches_scipy(dtype):
    a = _random_array((32, 32), dtype)
    b = _random_array((32, 32), dtype)

    result = convolve_cpu(a, b)
    expected = scipy.signal.fftconvolve(a, b, mode="same")

    npt.assert_allclose(result, expected, rtol=1e-5)


def test_convolve_cpu_known_result():
    a = np.array([[1.0]], dtype=np.float64)
    b = np.array([[2.0]], dtype=np.float64)
    assert convolve_cpu(a, b)[0, 0] == pytest.approx(2.0)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
@pytest.mark.parametrize("dtype", [np.float32])
def test_cuda_convolve_matches_cpu(dtype):
    shape = (32, 32)
    a = _random_array(shape, dtype)
    b = _random_array(shape, dtype)

    cpu_result = convolve_cpu(a, b)
    cuda_fn = make_cuda_convolve(shape, dtype=np.dtype(dtype))
    cuda_result = cuda_fn(a, b)

    npt.assert_allclose(cpu_result, cuda_result, rtol=1e-3)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
@pytest.mark.parametrize("dtype", [np.float32])
def test_cuda_psf_convolve_matches_cpu(dtype):
    shape = (32, 32)
    a = _random_array(shape, dtype)
    b = _random_array(shape, dtype)

    cpu_result = convolve_cpu(a, b)
    cuda_fn = make_cuda_psf_convolve(b, shape)
    cuda_result = cuda_fn(a)

    npt.assert_allclose(cpu_result, cuda_result, rtol=1e-3)


@pytest.mark.skipif(not _CUDA_AVAILABLE, reason="cupy not installed")
def test_cuda_psf_convolve_reuses_fft():
    shape = (32, 32)
    psf = _random_array(shape, np.float32)
    data1 = _random_array(shape, np.float32)
    data2 = _random_array(shape, np.float32)

    fn = make_cuda_psf_convolve(psf, shape)
    result1 = fn(data1)
    result2 = fn(data2)

    assert result1.shape == shape
    assert result2.shape == shape
    assert not np.allclose(result1, result2)


def test_missing_cupy_raises():
    if _CUDA_AVAILABLE:
        pytest.skip("cupy is available")

    with pytest.raises(RuntimeError, match="cupy is not installed"):
        make_cuda_psf_convolve(np.ones((8, 8), dtype=np.float32), (8, 8))
