"""
Comprehensive unit tests for miplib.processing.deconvolution.deconvolve module.
Tests the Richardson-Lucy deconvolution functionality.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.deconvolve import DeconvolutionRL


class MockOptions:
    """Mock options object with default deconvolution parameters."""

    def __init__(self):
        self.verbose = False
        self.save_intermediate_results = False
        self.memmap_estimates = False
        self.disable_tau1 = False  # Enable tau1 to create prev_estimate
        self.num_blocks = 1
        self.block_pad = 0
        self.max_nof_iterations = 10
        self.rl_frc_stop = 0
        self.rl_auto_background = False
        self.rl_background = 0
        self.convergence_epsilon = 0.001
        self.stop_tau = 1e-6
        self.first_estimate = "image"
        self.update_blind_psf = 0
        self.tv_lambda = 0  # Total variation regularization
        self.tv_iterations = 0


class MockWriter:
    """Mock writer for testing intermediate results."""

    def write(self, image):
        pass


class TestDeconvolutionRLInitialization:
    """Test DeconvolutionRL class initialization."""

    def test_init_basic(self):
        """Test basic initialization with minimal parameters."""
        # Create test data
        image_data = np.random.rand(10, 10).astype(np.float32)
        psf_data = np.random.rand(5, 5).astype(np.float32)
        psf_data /= psf_data.sum()  # Normalize PSF

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        writer = MockWriter()

        # Initialize deconvolution
        deconv = DeconvolutionRL(image, psf, writer, options)

        # Check basic attributes
        assert deconv.image is image
        assert deconv.psf.shape == psf.shape  # PSF gets processed internally
        assert deconv.options is options
        assert deconv.writer is writer
        assert deconv.iteration_count == 0
        assert deconv.imdims == 2

        # Check arrays are initialized
        assert deconv.estimate.shape == image.shape
        assert deconv.estimate_new.shape == image.shape

        # Clean up
        deconv.close()

    def test_init_with_different_dimensions(self):
        """Test initialization with 3D data."""
        image_data = np.random.rand(8, 8, 8).astype(np.float32)
        psf_data = np.random.rand(3, 3, 3).astype(np.float32)
        psf_data /= psf_data.sum()

        image = Image(image_data, spacing=(1.0, 1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0, 1.0))
        options = MockOptions()
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)

        assert deconv.imdims == 3
        assert deconv.estimate.shape == image.shape

        deconv.close()

    def test_init_with_memmap(self):
        """Test initialization with memory mapping enabled."""
        image_data = np.random.rand(5, 5).astype(np.float32)
        psf_data = np.ones((3, 3), dtype=np.float32) / 9

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        options.memmap_estimates = True
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)

        # Check that memmap directory was created
        assert os.path.isdir(deconv.memmap_directory)

        deconv.close()

    def test_init_invalid_inputs(self):
        """Test initialization with invalid inputs."""
        image_data = np.random.rand(5, 5).astype(np.float32)
        psf_data = np.ones((3, 3), dtype=np.float32) / 9

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        writer = MockWriter()

        # Test with non-Image objects
        with pytest.raises(AssertionError):
            DeconvolutionRL(image_data, psf, writer, options)

        with pytest.raises(AssertionError):
            DeconvolutionRL(image, psf_data, writer, options)


class TestDeconvolutionRLMethods:
    """Test individual methods of DeconvolutionRL class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.image_data = np.random.rand(6, 6).astype(np.float32) + 0.1
        self.psf_data = np.ones((3, 3), dtype=np.float32) / 9

        self.image = Image(self.image_data, spacing=(1.0, 1.0))
        self.psf = Image(self.psf_data, spacing=(1.0, 1.0))
        self.options = MockOptions()
        self.writer = MockWriter()

        self.deconv = DeconvolutionRL(self.image, self.psf, self.writer, self.options)

    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self, "deconv"):
            self.deconv.close()

    def test_progress_parameters_property(self):
        """Test progress_parameters property returns DataFrame."""
        progress_df = self.deconv.progress_parameters

        # Check it's a pandas DataFrame
        import pandas as pd

        assert isinstance(progress_df, pd.DataFrame)

        # Check columns
        expected_columns = ["t", "tau1", "leak", "e", "s", "u", "n", "uesu"]
        assert list(progress_df.columns) == expected_columns

    def test_get_result(self):
        """Test get_result method returns correct Image."""
        result = self.deconv.get_result()

        assert isinstance(result, Image)
        assert result.shape == self.image.shape
        assert result.spacing == self.image.spacing

    def test_get_8bit_result(self):
        """Test get_8bit_result method."""
        # Set some data in estimate
        self.deconv.estimate[:] = np.random.rand(*self.image.shape).astype(np.float32)

        result_8bit = self.deconv.get_8bit_result()

        assert isinstance(result_8bit, Image)
        assert result_8bit.dtype == np.uint8
        assert result_8bit.shape == self.image.shape

    def test_compute_estimate(self):
        """Test compute_estimate method."""
        # Set up estimate with some initial values (first estimate from image)
        self.deconv.estimate[:] = self.image[:].astype(np.float32)

        # Call compute_estimate - this should modify estimate_new
        result = self.deconv.compute_estimate()

        # Check that we get a result tuple
        assert isinstance(result, tuple)
        assert len(result) == 4  # e, s, u, n values

        # Check that estimate_new was calculated (Richardson-Lucy should produce non-zero values)
        # Since we start with the original image, RL should modify it
        assert not np.array_equal(self.deconv.estimate_new, self.deconv.estimate)

        # Check that iteration count is still managed externally (by execute method)
        assert self.deconv.iteration_count == 0


class TestDeconvolutionRLExecution:
    """Test execution of deconvolution algorithm."""

    def setup_method(self):
        """Set up test fixtures for execution tests."""
        # Create a simple test case: blurred impulse
        self.image_size = (8, 8)
        self.psf_size = (3, 3)

        # Create an impulse in the center
        impulse = np.zeros(self.image_size, dtype=np.float32)
        impulse[4, 4] = 1.0

        # Create a simple PSF (Gaussian-like)
        psf_data = np.array(
            [[0.1, 0.2, 0.1], [0.2, 0.4, 0.2], [0.1, 0.2, 0.1]], dtype=np.float32
        )

        # Simulate blurred image by convolving impulse with PSF
        from scipy.signal import convolve2d

        blurred = convolve2d(impulse, psf_data, mode="same", boundary="fill")

        self.image = Image(blurred + 0.01, spacing=(1.0, 1.0))  # Add small noise
        self.psf = Image(psf_data, spacing=(1.0, 1.0))
        self.options = MockOptions()
        self.options.max_nof_iterations = 5  # Short test
        self.options.verbose = False
        self.writer = MockWriter()

    def test_execute_basic(self):
        """Test basic execution of deconvolution."""
        deconv = DeconvolutionRL(self.image, self.psf, self.writer, self.options)

        # Mock the progress bar to avoid UI dependencies
        with patch(
            "miplib.processing.deconvolution.deconvolve.ops_output.ProgressBar"
        ) as mock_pb:
            mock_progress = Mock()
            mock_pb.return_value = mock_progress

            # Execute deconvolution
            deconv.execute()

        # Check that iterations ran
        assert deconv.iteration_count > 0
        assert deconv.iteration_count <= self.options.max_nof_iterations

        # Check that result has same shape as input
        result = deconv.get_result()
        assert result.shape == self.image.shape

        deconv.close()

    def test_execute_with_different_stopping_conditions(self):
        """Test execution with different stopping conditions."""
        self.options.max_nof_iterations = 2
        self.options.disable_tau1 = False  # Already False by default now
        self.options.stop_tau = 0.1  # High threshold for quick stopping

        deconv = DeconvolutionRL(self.image, self.psf, self.writer, self.options)

        with patch(
            "miplib.processing.deconvolution.deconvolve.ops_output.ProgressBar"
        ) as mock_pb:
            mock_progress = Mock()
            mock_pb.return_value = mock_progress

            deconv.execute()

        # Should have stopped early due to tau1 threshold
        assert deconv.iteration_count <= self.options.max_nof_iterations

        deconv.close()


class TestDeconvolutionRLEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_psf_handling(self):
        """Test handling of zero PSF (should not crash)."""
        image_data = np.ones((5, 5), dtype=np.float32)
        psf_data = np.zeros((3, 3), dtype=np.float32)

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        writer = MockWriter()

        # Should initialize without error
        deconv = DeconvolutionRL(image, psf, writer, options)
        assert deconv is not None

        deconv.close()

    def test_single_pixel_image(self):
        """Test with single pixel image."""
        image_data = np.array([[1.0]], dtype=np.float32)
        psf_data = np.array([[1.0]], dtype=np.float32)

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)
        result = deconv.get_result()

        assert result.shape == (1, 1)

        deconv.close()

    def test_cleanup_temp_directory(self):
        """Test that temporary directories are cleaned up properly."""
        image_data = np.ones((3, 3), dtype=np.float32)
        psf_data = np.ones((3, 3), dtype=np.float32) / 9

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))
        options = MockOptions()
        options.memmap_estimates = True
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)
        temp_dir = deconv.memmap_directory

        assert os.path.isdir(temp_dir)

        deconv.close()

        # Directory should be cleaned up after close
        assert not os.path.exists(temp_dir)


class TestDeconvolutionRLIntegration:
    """Integration tests with real-world-like scenarios."""

    def test_realistic_deconvolution_2d(self):
        """Test with a more realistic 2D scenario."""
        # Create a test image with multiple objects
        image_size = (20, 20)
        image_data = np.zeros(image_size, dtype=np.float32)

        # Add some "objects"
        image_data[5:8, 5:8] = 1.0
        image_data[12:15, 12:15] = 0.8
        image_data[8:11, 15:18] = 0.6

        # Create PSF (simple Gaussian-like)
        psf_data = np.array(
            [[0.05, 0.1, 0.05], [0.1, 0.6, 0.1], [0.05, 0.1, 0.05]], dtype=np.float32
        )

        # Blur the image
        from scipy.signal import convolve2d

        blurred = convolve2d(image_data, psf_data, mode="same", boundary="symm")

        # Add noise
        noise = np.random.poisson(blurred * 100) / 100.0
        noisy_blurred = noise.astype(np.float32) + 0.01

        image = Image(noisy_blurred, spacing=(0.1, 0.1))  # 0.1 μm pixels
        psf = Image(psf_data, spacing=(0.1, 0.1))

        options = MockOptions()
        options.max_nof_iterations = 10
        options.verbose = False
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)

        with patch("miplib.processing.deconvolution.deconvolve.ops_output.ProgressBar"):
            deconv.execute()

        result = deconv.get_result()

        # Basic checks
        assert result.shape == image.shape
        assert result.spacing == image.spacing
        assert np.all(result >= 0)  # Non-negative values
        assert np.max(result) > np.max(image)  # Should enhance contrast

        deconv.close()

    def test_with_progress_tracking(self):
        """Test that progress tracking works correctly."""
        image_data = np.random.rand(6, 6).astype(np.float32)
        psf_data = np.ones((3, 3), dtype=np.float32) / 9

        image = Image(image_data, spacing=(1.0, 1.0))
        psf = Image(psf_data, spacing=(1.0, 1.0))

        options = MockOptions()
        options.max_nof_iterations = 3
        options.disable_tau1 = False
        writer = MockWriter()

        deconv = DeconvolutionRL(image, psf, writer, options)

        with patch("miplib.processing.deconvolution.deconvolve.ops_output.ProgressBar"):
            deconv.execute()

        # Check progress tracking
        progress_df = deconv.progress_parameters
        assert len(progress_df) >= deconv.iteration_count

        # Check that some values were recorded
        recorded_iterations = progress_df.iloc[: deconv.iteration_count]
        assert not recorded_iterations.empty

        deconv.close()
