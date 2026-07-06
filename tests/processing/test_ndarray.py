import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing.ndarray import (
    cast_to_dtype,
    center_of_mass,
    contract_to_shape,
    expand_to_shape,
    first_order_derivative_2d,
    float2dtype,
    get_rounded_kernel,
    mul_seq,
    normalize,
    nroot,
    rescale_to_min_max,
    reverse_array,
    safe_divide,
    start_to_offset_idx,
    start_to_stop_idx,
)


def test_nroot_array():
    arr = np.array([1.0, 8.0, 27.0])
    result = nroot(arr, 3)
    npt.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))


def test_nroot_single():
    assert nroot(8.0, 3) == pytest.approx(2.0)


def test_normalize_sums_to_one():
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    result = normalize(arr)
    assert result.sum() == pytest.approx(1.0)


def test_normalize_preserves_relative_scale():
    arr = np.array([1.0, 2.0])
    result = normalize(arr)
    assert result[1] / result[0] == pytest.approx(2.0)


def test_float2dtype_single():
    assert float2dtype("single") is np.float32


def test_float2dtype_none():
    assert float2dtype(None) is np.float32


def test_float2dtype_double():
    assert float2dtype("double") is np.float64


def test_float2dtype_invalid():
    with pytest.raises(NotImplementedError, match="'bad'"):
        float2dtype("bad")


def test_contract_to_shape_center_crop():
    data = np.ones((10, 10))
    result = contract_to_shape(data, (6, 6))
    assert result.shape == (6, 6)


def test_contract_to_shape_same_size():
    data = np.ones((5, 5))
    result = contract_to_shape(data, (5, 5))
    assert result.shape == (5, 5)


def test_contract_to_shape_invalid():
    data = np.ones((5, 5))
    with pytest.raises(ValueError, match="All target shape"):
        contract_to_shape(data, (10, 10))


def test_expand_to_shape_larger():
    data = np.ones((2, 3))
    result = expand_to_shape(data, (4, 5))
    assert result.shape == (4, 5)
    assert result[1, 2] == pytest.approx(1.0)


def test_expand_to_shape_same_size():
    data = np.ones((3, 3))
    result = expand_to_shape(data, (3, 3))
    npt.assert_array_equal(result, data)


def test_mul_seq_basic():
    assert mul_seq([1, 2, 3, 4]) == 24


def test_mul_seq_single():
    assert mul_seq([7]) == 7


def test_mul_seq_empty():
    assert mul_seq([]) == 1


def test_safe_divide_normal():
    num = np.array([4.0, 6.0])
    den = np.array([2.0, 3.0])
    result = safe_divide(num, den)
    npt.assert_array_almost_equal(result, np.array([2.0, 2.0]))


def test_safe_divide_by_zero():
    num = np.array([4.0, 0.0])
    den = np.array([0.0, 5.0])
    result = safe_divide(num, den)
    assert result[0] == 0.0
    assert result[1] == 0.0


def test_start_to_stop_idx():
    start = np.array([0, 1])
    stop = np.array([5, 6])
    slices = start_to_stop_idx(start, stop)
    assert slices == (slice(0, 5), slice(1, 6))


def test_start_to_offset_idx():
    start = np.array([0, 2])
    offset = np.array([3, 4])
    slices = start_to_offset_idx(start, offset)
    assert slices == (slice(0, 3), slice(2, 6))


def test_reverse_array_1d():
    arr = np.array([1, 2, 3])
    result = reverse_array(arr)
    npt.assert_array_equal(result, np.array([3, 2, 1]))


def test_reverse_array_2d():
    arr = np.array([[1, 2], [3, 4]])
    result = reverse_array(arr)
    assert result[0, 0] == 4
    assert result[1, 1] == 1


def test_reverse_array_3d():
    arr = np.arange(8).reshape(2, 2, 2)
    result = reverse_array(arr)
    assert result[0, 0, 0] == 7
    assert result[1, 1, 1] == 0


def test_reverse_array_does_not_mutate():
    arr = np.array([1, 2, 3])
    original = arr.copy()
    reverse_array(arr)
    npt.assert_array_equal(arr, original)


def test_first_order_derivative_2d():
    arr = np.ones((5, 5))
    result = first_order_derivative_2d(arr)
    assert result.shape == (5, 5)
    npt.assert_array_equal(result, np.zeros((5, 5)))


def test_get_rounded_kernel_shape():
    kernel = get_rounded_kernel(9)
    assert kernel.shape == (9, 9)


def test_get_rounded_kernel_center_is_one():
    kernel = get_rounded_kernel(31)
    assert kernel[15, 15] == 1


def test_get_rounded_kernel_corners_are_zero():
    kernel = get_rounded_kernel(31)
    assert kernel[0, 0] == 0


def test_center_of_mass():
    xx, yy = np.meshgrid(np.arange(5), np.arange(5))
    arr = np.zeros((5, 5))
    arr[2, 2] = 1.0
    cx, cy = center_of_mass(xx, yy, arr)
    assert cx == pytest.approx(2.0)
    assert cy == pytest.approx(2.0)


def test_rescale_to_min_max():
    data = np.array([0.0, 5.0, 10.0])
    result = rescale_to_min_max(data, 0, 100)
    assert result.max() == pytest.approx(100.0)


def test_cast_to_dtype_no_rescale():
    data = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    result = cast_to_dtype(data, np.float32, rescale=False)
    assert result.dtype == np.float32


def test_cast_to_dtype_same_dtype():
    data = np.array([1.0, 2.0], dtype=np.float64)
    result = cast_to_dtype(data, np.float64)
    npt.assert_array_equal(result, data)
