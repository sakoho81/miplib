import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing import ops_ext

# ---------------------------------------------------------------------------
# update_estimate_poisson
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_update_estimate_poisson_basic(dtype):
    estimate = np.array([1.0, 2.0, 3.0], dtype=dtype)
    update_values = np.array([0.5, 1.0, 2.0], dtype=dtype)

    ops_ext.update_estimate_poisson(estimate, update_values, 0.1)

    expected = np.array([0.5, 2.0, 6.0], dtype=dtype)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_poisson_complex64():
    estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    update_values = np.array([0.5 + 0.1j, 1.0 + 0.2j, 2.0 + 0.3j], dtype=np.complex64)

    ops_ext.update_estimate_poisson(estimate, update_values, 0.1)

    expected = np.array([0.5, 2.0, 6.0], dtype=np.float32)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_poisson_negative_clipping():
    estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    update_values = np.array([-0.5, 1.0, -2.0], dtype=np.float32)

    ops_ext.update_estimate_poisson(estimate, update_values, 0.1)

    expected = np.array([0.0, 2.0, 0.0], dtype=np.float32)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_poisson_multidimensional():
    estimate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    update_values = np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float32)

    ops_ext.update_estimate_poisson(estimate, update_values, 0.1)

    expected = np.array([[0.5, 3.0], [6.0, 3.2]], dtype=np.float32)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_poisson_convergence_param_invalid():
    estimate = np.array([1.0, 2.0], dtype=np.float32)
    update_values = np.array([0.5, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="convergence_param must be between 0 and 0.5"):
        ops_ext.update_estimate_poisson(estimate, update_values, -0.1)

    with pytest.raises(ValueError, match="convergence_param must be between 0 and 0.5"):
        ops_ext.update_estimate_poisson(estimate, update_values, 0.6)

    ops_ext.update_estimate_poisson(estimate.copy(), update_values, 0.0)
    ops_ext.update_estimate_poisson(estimate.copy(), update_values, 0.5)


def test_update_estimate_poisson_size_mismatch():
    estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    update_values = np.array([0.5, 1.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Arrays must have the same size"):
        ops_ext.update_estimate_poisson(estimate, update_values, 0.1)


def test_update_estimate_poisson_unsupported_dtype():
    estimate = np.array([1, 2, 3], dtype=np.int32)
    update_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(TypeError, match="Unsupported array types"):
        ops_ext.update_estimate_poisson(estimate, update_values, 0.1)


# ---------------------------------------------------------------------------
# update_estimate_gauss
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_update_estimate_gauss_basic(dtype):
    estimate = np.array([1.0, 2.0, 3.0], dtype=dtype)
    gradient = np.array([0.1, -0.2, 0.3], dtype=dtype)
    step_size = 0.5

    ops_ext.update_estimate_gauss(estimate, gradient, 0.1, step_size)

    expected = np.array([1.05, 1.9, 3.15], dtype=dtype)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_gauss_complex128():
    estimate = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gradient = np.array([0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j], dtype=np.complex128)
    step_size = 0.5

    ops_ext.update_estimate_gauss(estimate, gradient, 0.1, step_size)

    expected = np.array([1.05, 1.9, 3.15], dtype=np.float64)
    npt.assert_array_almost_equal(estimate, expected)


def test_update_estimate_gauss_convergence_param_invalid():
    estimate = np.array([1.0, 2.0], dtype=np.float32)
    gradient = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="convergence_param must be between 0 and 0.5"):
        ops_ext.update_estimate_gauss(estimate, gradient, -0.1, 1.0)

    with pytest.raises(ValueError, match="convergence_param must be between 0 and 0.5"):
        ops_ext.update_estimate_gauss(estimate, gradient, 0.6, 1.0)


def test_update_estimate_gauss_size_mismatch():
    estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient = np.array([0.1, 0.2], dtype=np.float32)

    with pytest.raises(ValueError, match="Arrays must have the same size"):
        ops_ext.update_estimate_gauss(estimate, gradient, 0.1, 1.0)


def test_update_estimate_gauss_unsupported_dtype():
    estimate = np.array([1, 2, 3], dtype=np.int32)
    gradient = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    with pytest.raises(TypeError, match="Unsupported array types"):
        ops_ext.update_estimate_gauss(estimate, gradient, 0.1, 1.0)


# ---------------------------------------------------------------------------
# inverse_division_inplace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype"),
    [(np.complex64, np.float32), (np.complex128, np.float64)],
)
def test_inverse_division_basic(complex_dtype, real_dtype):
    complex_array = np.array([1.0 + 2.0j, 3.0 + 4.0j, 2.0 + 0.0j], dtype=complex_dtype)
    real_array = np.array([2.0, 5.0, 4.0], dtype=real_dtype)

    ops_ext.inverse_division_inplace(complex_array, real_array)

    # 2/(1+2j) = 0.4 - 0.8j
    # 5/(3+4j) = 0.6 - 0.8j
    # 4/(2+0j) = 2.0 + 0.0j
    npt.assert_almost_equal(complex_array[0], 0.4 - 0.8j, decimal=6)
    npt.assert_almost_equal(complex_array[1], 0.6 - 0.8j, decimal=6)
    npt.assert_almost_equal(complex_array[2], 2.0 + 0.0j, decimal=6)


def test_inverse_division_zero_handling():
    complex_array = np.array([0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 3.0j], dtype=np.complex64)
    real_array = np.array([1.0, 0.0, 2.0], dtype=np.float32)

    ops_ext.inverse_division_inplace(complex_array, real_array)

    assert complex_array[0] == 0.0 + 0.0j
    assert complex_array[1] == 0.0 + 0.0j
    assert complex_array[2] != 0.0 + 0.0j


def test_inverse_division_size_mismatch():
    complex_array = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64)
    real_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Arrays must have the same size"):
        ops_ext.inverse_division_inplace(complex_array, real_array)


def test_inverse_division_dtype_mismatch():
    complex_array = np.array([1.0 + 1.0j], dtype=np.complex64)
    real_array = np.array([1.0], dtype=np.float64)

    with pytest.raises(TypeError, match="Unsupported array types"):
        ops_ext.inverse_division_inplace(complex_array, real_array)


# ---------------------------------------------------------------------------
# inverse_subtraction_inplace
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("complex_dtype", "real_dtype"),
    [(np.complex64, np.float32), (np.complex128, np.float64)],
)
def test_inverse_subtraction_basic(complex_dtype, real_dtype):
    complex_array = np.array([1.0 + 2.0j, 3.0 + 4.0j], dtype=complex_dtype)
    real_array = np.array([10.0, 20.0], dtype=real_dtype)
    multiplier = 2.0

    ops_ext.inverse_subtraction_inplace(complex_array, real_array, multiplier)

    expected_real = np.array([8.0, 14.0])
    expected_imag = np.array([2.0, 4.0])
    npt.assert_array_almost_equal(complex_array.real, expected_real)
    npt.assert_array_almost_equal(complex_array.imag, expected_imag)


def test_inverse_subtraction_size_mismatch():
    complex_array = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64)
    real_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Arrays must have the same size"):
        ops_ext.inverse_subtraction_inplace(complex_array, real_array, 1.0)


def test_inverse_subtraction_dtype_mismatch():
    complex_array = np.array([1.0 + 1.0j], dtype=np.complex128)
    real_array = np.array([1.0], dtype=np.float32)

    with pytest.raises(TypeError, match="Unsupported array types"):
        ops_ext.inverse_subtraction_inplace(complex_array, real_array, 1.0)


# ---------------------------------------------------------------------------
# kullback_leibler_divergence
# ---------------------------------------------------------------------------


def test_kullback_leibler_divergence_values():
    first = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    second = np.array([1.5, 2.5, 2.0], dtype=np.float64)

    result = ops_ext.kullback_leibler_divergence(first, second, 0.5)

    # Manually computed: avg of (second - first + first * ln(first / second))
    # (1.5 - 1 + 1*ln(1/1.5)) = 0.0945, (2.5 - 2 + 2*ln(0.8)) = 0.0537,
    # (2 - 3 + 3*ln(1.5)) = 0.2164, sum/3 = 0.12155
    npt.assert_almost_equal(result, 0.12155, decimal=4)


def test_kullback_leibler_divergence_identical():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = ops_ext.kullback_leibler_divergence(array, array, 0.5)
    npt.assert_almost_equal(result, 0.0, decimal=10)


def test_kullback_leibler_divergence_size_mismatch():
    first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    second = np.array([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError, match="Arrays must have the same size"):
        ops_ext.kullback_leibler_divergence(first, second, 0.5)


def test_kullback_leibler_divergence_dtype_mismatch():
    first = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    second = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with pytest.raises(TypeError, match="Array types must match"):
        ops_ext.kullback_leibler_divergence(first, second, 0.5)


# ---------------------------------------------------------------------------
# div_unit_grad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_div_unit_grad_constant(dtype):
    image = np.ones((3, 3, 3), dtype=dtype)
    voxel_spacing = (1.0, 1.0, 1.0)

    result = ops_ext.div_unit_grad(image, voxel_spacing)

    npt.assert_array_almost_equal(result, np.zeros_like(result), decimal=5)


def test_div_unit_grad_wrong_dimensions():
    image_2d = np.random.rand(5, 5).astype(np.float32)

    with pytest.raises(ValueError, match="Input must be 3D array"):
        ops_ext.div_unit_grad(image_2d, (1.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# div_unit_grad1
# ---------------------------------------------------------------------------


def test_div_unit_grad1_values():
    image = np.array([1.0, 2.0, 4.0, 3.0, 1.0], dtype=np.float64)

    result = ops_ext.div_unit_grad1(image, 1.0)

    expected = np.array([1.0, 0.0, -2.0, 0.0, 1.0], dtype=np.float64)
    npt.assert_array_almost_equal(result, expected)


def test_div_unit_grad1_linear():
    image = np.linspace(0, 10, 11, dtype=np.float64)
    result = ops_ext.div_unit_grad1(image, 1.0)
    assert np.all(np.isfinite(result))
    npt.assert_array_almost_equal(result[1:-1], np.zeros(9), decimal=10)


def test_div_unit_grad1_wrong_dtype():
    image = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    with pytest.raises(TypeError, match="Input must be float64"):
        ops_ext.div_unit_grad1(image, 1.0)
