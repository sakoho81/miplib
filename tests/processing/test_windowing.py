import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing.windowing import apply_hamming_window, apply_tukey_window


@pytest.mark.parametrize("shape", [(16,), (8, 8), (4, 4, 4)])
def test_hamming_preserves_shape(shape):
    data = np.random.rand(*shape)
    result = apply_hamming_window(data)
    assert result.shape == shape


@pytest.mark.parametrize("shape", [(16,), (8, 8), (4, 4, 4)])
def test_hamming_output_is_float64(shape):
    data = np.ones(shape, dtype=np.int32)
    result = apply_hamming_window(data)
    assert result.dtype == np.float64


@pytest.mark.parametrize("shape", [(16,), (8, 8), (4, 4, 4)])
def test_hamming_does_not_mutate_input(shape):
    data = np.ones(shape)
    original = data.copy()
    apply_hamming_window(data)
    npt.assert_array_equal(data, original)


def test_hamming_edge_attenuation_1d():
    data = np.ones(100)
    result = apply_hamming_window(data)
    mid = len(result) // 2
    assert result[0] < result[mid]
    assert result[-1] < result[mid]


def test_hamming_edge_attenuation_2d():
    data = np.ones((20, 20))
    result = apply_hamming_window(data)
    mid = result.shape[0] // 2
    assert result[0, 0] < result[mid, mid]


@pytest.mark.parametrize("shape", [(16,), (8, 8)])
def test_tukey_preserves_shape(shape):
    data = np.random.rand(*shape)
    result = apply_tukey_window(data)
    assert result.shape == shape


@pytest.mark.parametrize("shape", [(16,), (8, 8)])
def test_tukey_output_is_float64(shape):
    data = np.ones(shape, dtype=np.int32)
    result = apply_tukey_window(data)
    assert result.dtype == np.float64


@pytest.mark.parametrize("shape", [(16,), (8, 8)])
def test_tukey_does_not_mutate_input(shape):
    data = np.ones(shape)
    original = data.copy()
    apply_tukey_window(data)
    npt.assert_array_equal(data, original)


def test_tukey_alpha_param_affects_result():
    data = np.ones(100)
    r1 = apply_tukey_window(data, alpha=0.25)
    r2 = apply_tukey_window(data, alpha=0.75)
    assert not np.allclose(r1, r2)


def test_tukey_edge_attenuation_1d():
    data = np.ones(100)
    result = apply_tukey_window(data)
    mid = len(result) // 2
    assert result[0] < result[mid]
