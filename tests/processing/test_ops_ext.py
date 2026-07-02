"""
Comprehensive unit tests for miplib.processing.ops_ext Cython extension.
Tests all functions with various input types, edge cases, and error conditions.
"""

import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing import ops_ext


class TestUpdateEstimatePoisson:
    """Test update_estimate_poisson function."""

    def test_basic_float32(self):
        """Test basic functionality with float32 arrays."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        update_values = np.array([0.5, 1.0, 2.0], dtype=np.float32)
        convergence_param = 0.1

        original_estimate = estimate.copy()
        exact, stable, unstable, negative = ops_ext.update_estimate_poisson(
            estimate, update_values, convergence_param
        )

        # Check that estimate was modified in place
        expected = original_estimate * update_values
        npt.assert_array_almost_equal(estimate, expected)

        # Check return values are reasonable
        assert isinstance(exact, float)
        assert isinstance(stable, float)
        assert isinstance(unstable, float)
        assert isinstance(negative, float)
        assert negative == 0.0  # No negative values in this case

    def test_basic_float64(self):
        """Test basic functionality with float64 arrays."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        update_values = np.array([0.5, 1.0, 2.0], dtype=np.float64)
        convergence_param = 0.1

        original_estimate = estimate.copy()
        result = ops_ext.update_estimate_poisson(
            estimate, update_values, convergence_param
        )

        # Check that estimate was modified correctly
        expected = original_estimate * update_values
        npt.assert_array_almost_equal(estimate, expected)

        # Should return 4-tuple
        assert len(result) == 4

    def test_complex64_update_values(self):
        """Test with complex64 update values (should use real part only)."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        update_values = np.array(
            [0.5 + 0.1j, 1.0 + 0.2j, 2.0 + 0.3j], dtype=np.complex64
        )
        convergence_param = 0.1

        original_estimate = estimate.copy()
        ops_ext.update_estimate_poisson(estimate, update_values, convergence_param)

        # Should use only real parts
        expected = original_estimate * update_values.real
        npt.assert_array_almost_equal(estimate, expected)

    def test_negative_clipping(self):
        """Test that negative update values are clipped to zero."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        update_values = np.array([-0.5, 1.0, -2.0], dtype=np.float32)
        convergence_param = 0.1

        ops_ext.update_estimate_poisson(estimate, update_values, convergence_param)

        # Negative values should be clipped to zero
        expected = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        npt.assert_array_almost_equal(estimate, expected)

    def test_multidimensional_arrays(self):
        """Test with multidimensional arrays."""
        estimate = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        update_values = np.array([[0.5, 1.5], [2.0, 0.8]], dtype=np.float32)
        convergence_param = 0.1

        original_estimate = estimate.copy()
        ops_ext.update_estimate_poisson(estimate, update_values, convergence_param)

        expected = original_estimate * update_values
        npt.assert_array_almost_equal(estimate, expected)

    def test_convergence_parameter_validation(self):
        """Test validation of convergence parameter bounds."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        update_values = np.array([0.5, 1.0, 2.0], dtype=np.float32)

        # Test lower bound
        with pytest.raises(
            ValueError, match="convergence_param must be between 0 and 0.5"
        ):
            ops_ext.update_estimate_poisson(estimate, update_values, -0.1)

        # Test upper bound
        with pytest.raises(
            ValueError, match="convergence_param must be between 0 and 0.5"
        ):
            ops_ext.update_estimate_poisson(estimate, update_values, 0.6)

        # Test boundary values (should work)
        ops_ext.update_estimate_poisson(estimate.copy(), update_values, 0.0)
        ops_ext.update_estimate_poisson(estimate.copy(), update_values, 0.5)

    def test_size_mismatch_error(self):
        """Test error when array sizes don't match."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        update_values = np.array([0.5, 1.0], dtype=np.float32)  # Different size

        with pytest.raises(ValueError, match="Arrays must have the same size"):
            ops_ext.update_estimate_poisson(estimate, update_values, 0.1)

    def test_unsupported_dtype_error(self):
        """Test error with unsupported data types."""
        estimate = np.array([1, 2, 3], dtype=np.int32)  # Integer type
        update_values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(TypeError, match="Unsupported array types"):
            ops_ext.update_estimate_poisson(estimate, update_values, 0.1)


class TestUpdateEstimateGauss:
    """Test update_estimate_gauss function."""

    def test_basic_functionality(self):
        """Test basic Gaussian update functionality."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        gradient = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        convergence_param = 0.1
        step_size = 0.5

        original_estimate = estimate.copy()
        result = ops_ext.update_estimate_gauss(
            estimate, gradient, convergence_param, step_size
        )

        # Check update: estimate += step_size * gradient
        expected = original_estimate + step_size * gradient
        npt.assert_array_almost_equal(estimate, expected)

        # Should return 4-tuple
        assert len(result) == 4

    def test_complex_gradient(self):
        """Test with complex gradient values (should use real part only)."""
        estimate = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        gradient = np.array([0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j], dtype=np.complex128)
        convergence_param = 0.1
        step_size = 0.5

        original_estimate = estimate.copy()
        ops_ext.update_estimate_gauss(estimate, gradient, convergence_param, step_size)

        # Should use only real parts
        expected = original_estimate + step_size * gradient.real
        npt.assert_array_almost_equal(estimate, expected)

    def test_zero_denominator_handling(self):
        """Test handling when original estimate has zero values."""
        estimate = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        gradient = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        convergence_param = 0.1
        step_size = 0.5

        # Should not crash with zero denominators
        result = ops_ext.update_estimate_gauss(
            estimate, gradient, convergence_param, step_size
        )
        assert len(result) == 4


class TestInverseDivisionInplace:
    """Test inverse_division_inplace function."""

    def test_basic_complex64_float32(self):
        """Test basic complex division with complex64/float32."""
        complex_array = np.array(
            [1.0 + 2.0j, 3.0 + 4.0j, 2.0 + 0.0j], dtype=np.complex64
        )
        real_array = np.array([2.0, 5.0, 4.0], dtype=np.float32)

        ops_ext.inverse_division_inplace(complex_array, real_array)

        # Check first element: 2.0 / (1.0 + 2.0j) = 0.4 - 0.8j
        npt.assert_almost_equal(complex_array[0].real, 0.4, decimal=6)
        npt.assert_almost_equal(complex_array[0].imag, -0.8, decimal=6)

        # Check real-only case: 4.0 / (2.0 + 0.0j) = 2.0 + 0.0j
        npt.assert_almost_equal(complex_array[2].real, 2.0, decimal=6)
        npt.assert_almost_equal(complex_array[2].imag, 0.0, decimal=6)

    def test_basic_complex128_float64(self):
        """Test basic complex division with complex128/float64."""
        complex_array = np.array([1.0 + 1.0j, 0.0 + 2.0j], dtype=np.complex128)
        real_array = np.array([2.0, 4.0], dtype=np.float64)

        ops_ext.inverse_division_inplace(complex_array, real_array)

        # Check: 2.0 / (1.0 + 1.0j) = 2.0 * (1.0 - 1.0j) / 2.0 = 1.0 - 1.0j
        npt.assert_almost_equal(complex_array[0].real, 1.0, decimal=10)
        npt.assert_almost_equal(complex_array[0].imag, -1.0, decimal=10)

        # Check: 4.0 / (0.0 + 2.0j) = 4.0 * (0.0 - 2.0j) / 4.0 = 0.0 - 2.0j
        npt.assert_almost_equal(complex_array[1].real, 0.0, decimal=10)
        npt.assert_almost_equal(complex_array[1].imag, -2.0, decimal=10)

    def test_zero_handling(self):
        """Test handling of zero values."""
        complex_array = np.array(
            [0.0 + 0.0j, 1.0 + 1.0j, 2.0 + 3.0j], dtype=np.complex64
        )
        real_array = np.array([1.0, 0.0, 2.0], dtype=np.float32)

        ops_ext.inverse_division_inplace(complex_array, real_array)

        # Zero complex value should remain zero
        assert complex_array[0] == 0.0 + 0.0j

        # Zero real value should result in zero
        assert complex_array[1] == 0.0 + 0.0j

        # Non-zero case should work normally
        assert complex_array[2] != 0.0 + 0.0j

    def test_size_mismatch_error(self):
        """Test error when array sizes don't match."""
        complex_array = np.array([1.0 + 1.0j, 2.0 + 2.0j], dtype=np.complex64)
        real_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # Different size

        with pytest.raises(ValueError, match="Arrays must have the same size"):
            ops_ext.inverse_division_inplace(complex_array, real_array)


class TestInverseSubtractionInplace:
    """Test inverse_subtraction_inplace function."""

    def test_basic_functionality(self):
        """Test basic subtraction functionality."""
        complex_array = np.array(
            [1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], dtype=np.complex64
        )
        real_array = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        multiplier = 2.0

        original_imag = complex_array.imag.copy()
        ops_ext.inverse_subtraction_inplace(complex_array, real_array, multiplier)

        # Real part should be: real_array - multiplier * original_real
        expected_real = np.array([10.0 - 2.0 * 1.0, 20.0 - 2.0 * 3.0, 30.0 - 2.0 * 5.0])
        npt.assert_array_almost_equal(complex_array.real, expected_real)

        # Imaginary part should remain unchanged
        npt.assert_array_almost_equal(complex_array.imag, original_imag)


class TestKullbackLeiblerDivergence:
    """Test kullback_leibler_divergence function."""

    def test_basic_float32(self):
        """Test basic KL divergence computation with float32."""
        first_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        second_array = np.array([1.5, 2.5, 2.0], dtype=np.float32)
        threshold = 0.5

        result = ops_ext.kullback_leibler_divergence(
            first_array, second_array, threshold
        )

        # Should return a single float value
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert np.isfinite(result)

    def test_identical_arrays(self):
        """Test KL divergence with identical arrays (should be 0)."""
        array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        threshold = 0.5

        result = ops_ext.kullback_leibler_divergence(array, array, threshold)

        # KL divergence should be close to 0 for identical arrays
        npt.assert_almost_equal(result, 0.0, decimal=6)

    def test_dtype_mismatch_error(self):
        """Test error when array dtypes don't match."""
        first_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        second_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)  # Different dtype

        with pytest.raises(TypeError, match="Array types must match"):
            ops_ext.kullback_leibler_divergence(first_array, second_array, 0.5)


class TestDivUnitGrad:
    """Test div_unit_grad function."""

    def test_basic_3d_float32(self):
        """Test basic 3D unit gradient divergence with float32."""
        # Simple 3x3x3 array
        image_3d = np.random.rand(3, 3, 3).astype(np.float32)
        voxel_spacing = (1.0, 1.0, 1.0)

        result = ops_ext.div_unit_grad(image_3d, voxel_spacing)

        # Should return array of same shape
        assert result.shape == image_3d.shape
        assert result.dtype == image_3d.dtype
        assert np.all(np.isfinite(result))

    def test_constant_array(self):
        """Test with constant array (should give zero divergence)."""
        image_3d = np.ones((3, 3, 3), dtype=np.float32)
        voxel_spacing = (1.0, 1.0, 1.0)

        result = ops_ext.div_unit_grad(image_3d, voxel_spacing)

        # Constant array should have zero gradient divergence
        npt.assert_array_almost_equal(result, np.zeros_like(result), decimal=5)

    def test_wrong_dimensions_error(self):
        """Test error with non-3D input."""
        image_2d = np.random.rand(5, 5).astype(np.float32)  # 2D instead of 3D
        voxel_spacing = (1.0, 1.0, 1.0)

        with pytest.raises(ValueError, match="Input must be 3D array"):
            ops_ext.div_unit_grad(image_2d, voxel_spacing)


class TestDivUnitGrad1:
    """Test div_unit_grad1 function."""

    def test_basic_1d(self):
        """Test basic 1D unit gradient divergence."""
        image_1d = np.array([1.0, 2.0, 4.0, 3.0, 1.0], dtype=np.float64)
        pixel_spacing = 1.0

        result = ops_ext.div_unit_grad1(image_1d, pixel_spacing)

        # Should return array of same shape
        assert result.shape == image_1d.shape
        assert result.dtype == image_1d.dtype
        assert np.all(np.isfinite(result))

    def test_linear_function(self):
        """Test with linear function (should give zero divergence)."""
        image_1d = np.linspace(0, 10, 11, dtype=np.float64)  # Linear function
        pixel_spacing = 1.0

        result = ops_ext.div_unit_grad1(image_1d, pixel_spacing)

        # Linear function should have constant gradient, so zero divergence
        # (except at boundaries due to discretization)
        assert np.all(np.isfinite(result))
        assert np.abs(result[1:-1]).max() < 1e-10  # Middle points should be ~0

    def test_wrong_dtype_error(self):
        """Test error with non-float64 input."""
        image_1d = np.array(
            [1.0, 2.0, 3.0], dtype=np.float32
        )  # float32 instead of float64
        pixel_spacing = 1.0

        with pytest.raises(TypeError, match="Input must be float64"):
            ops_ext.div_unit_grad1(image_1d, pixel_spacing)


class TestModuleProperties:
    """Test module-level properties and metadata."""

    def test_version_exists(self):
        """Test that version string exists."""
        assert hasattr(ops_ext, "__version__")
        assert isinstance(ops_ext.__version__, str)
        assert len(ops_ext.__version__) > 0

    def test_all_functions_exist(self):
        """Test that all expected functions are available."""
        expected_functions = [
            "update_estimate_poisson",
            "update_estimate_gauss",
            "inverse_division_inplace",
            "inverse_subtraction_inplace",
            "kullback_leibler_divergence",
            "div_unit_grad",
            "div_unit_grad1",
        ]

        for func_name in expected_functions:
            assert hasattr(ops_ext, func_name), f"Missing function: {func_name}"
            assert callable(getattr(ops_ext, func_name)), f"Not callable: {func_name}"


@pytest.mark.slow
class TestPerformance:
    """Performance and stress tests."""

    def test_large_arrays(self):
        """Test with large arrays to check performance and memory usage."""
        size = 1000
        estimate = np.random.rand(size).astype(np.float32)
        update_values = np.random.rand(size).astype(np.float32)

        # Should not crash or take too long
        result = ops_ext.update_estimate_poisson(estimate, update_values, 0.1)
        assert len(result) == 4

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_3d_performance(self, dtype):
        """Test 3D gradient computation performance."""
        shape = (20, 20, 20)  # Moderate size for CI
        image_3d = np.random.rand(*shape).astype(dtype)
        voxel_spacing = (1.0, 1.0, 1.0)

        result = ops_ext.div_unit_grad(image_3d, voxel_spacing)
        assert result.shape == shape
        assert result.dtype == dtype
