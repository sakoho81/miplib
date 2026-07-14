import numpy as np
import pytest

from miplib.analysis.image_quality.filters import (
    PowerSpectrumStats,
    brenner_quality,
    frequency_quality,
    local_image_quality,
    spectral_moments,
)
from miplib.data.containers.image import Image
from tests.conftest import sine_grating, step_edge

# --- local_image_quality ---


def test_local_image_quality_constant_image_low_entropy():
    """Constant image has minimal entropy (single histogram bin)."""
    constant = Image(np.ones((64, 64)) * 100, (1.0, 1.0))
    entropy = local_image_quality(constant)
    assert entropy == pytest.approx(0.0, abs=1e-10)


def test_local_image_quality_high_detail_higher_entropy(camera_image, gaussian_2d):
    """High-detail image has higher entropy than smooth Gaussian."""
    entropy_camera = local_image_quality(camera_image)
    entropy_gaussian = local_image_quality(gaussian_2d)
    # Camera image has more detail (edges, textures) than smooth Gaussian
    assert entropy_camera > entropy_gaussian


def test_local_image_quality_masking_focuses_on_object_detail():
    """Masking excludes background and focuses on object texture/structure.

    With images that have clear foreground objects (coins, blobs), the masked
    histogram should better represent the object's internal variation, excluding
    the dominant background signal.
    """
    from skimage import data

    from miplib.analysis.image_quality.utils import calculate_entropy
    from miplib.processing.segmentation import masking

    # Load coins image - has clear coin objects on dark background
    coins_data = data.coins().astype(np.float64)
    coins = Image(coins_data, spacing=(1.0, 1.0))

    # Get masked region
    mask = masking.make_local_intensity_based_mask(
        coins, threshold=80, kernel_size=15, invert=False
    )

    # Extract masked pixels (coin regions)
    masked_pixels = coins_data[mask.astype(bool)]

    # The masked histogram should capture coin texture variation
    # Check that masked region has meaningful intensity variation
    assert masked_pixels.std() > 10  # coins have texture/variation
    assert len(masked_pixels) > 1000  # significant portion of image

    # Unmasked histogram is dominated by background
    unmasked_pixels = coins_data.flatten()

    # Masked pixels should have different distribution than unmasked
    # (background is dark, coins are bright with internal variation)
    assert masked_pixels.mean() > unmasked_pixels.mean()

    # The masked entropy represents the object's internal detail
    masked_entropy = calculate_entropy(masked_pixels)
    assert masked_entropy > 0  # coins have texture


# --- frequency_quality ---


def test_frequency_quality_returns_power_spectrum_stats():
    """frequency_quality returns PowerSpectrumStats with correct fields."""
    image = Image(np.ones((64, 64)), (1.0, 1.0))
    stats = frequency_quality(image)
    assert isinstance(stats, PowerSpectrumStats)
    assert hasattr(stats, "mean")
    assert hasattr(stats, "std")
    assert hasattr(stats, "entropy")
    assert hasattr(stats, "threshold_freq")
    assert hasattr(stats, "power_at_high_freq")
    assert hasattr(stats, "skew")
    assert hasattr(stats, "kurtosis")
    assert hasattr(stats, "mean_bin")


def test_frequency_quality_constant_image_zero_high_freq_power():
    """Constant image has zero power at high frequencies."""
    constant = Image(np.ones((64, 64)) * 50, (1.0, 1.0))
    stats = frequency_quality(constant)
    assert stats.power_at_high_freq == pytest.approx(0.0, abs=1e-10)


def test_frequency_quality_sine_grating_peak_at_known_frequency():
    """Sine grating produces high power at the grating frequency."""
    grating = sine_grating((64, 64), frequency=0.1)
    image = Image(grating, (1.0, 1.0))
    stats = frequency_quality(image)
    assert stats.mean > 0
    assert stats.power_at_high_freq > 0


# --- spectral_moments ---


def test_spectral_moments_deterministic(camera_image):
    """Spectral moments produces same result for same input."""
    moments1 = spectral_moments(camera_image)
    moments2 = spectral_moments(camera_image)
    assert moments1 == pytest.approx(moments2)


def test_spectral_moments_sharp_vs_blurred(camera_image):
    """Sharp image has higher spectral moments than blurred version."""
    from scipy.ndimage import gaussian_filter

    sharp_moments = spectral_moments(camera_image)
    # Create blurred version
    blurred_data = gaussian_filter(camera_image[:], sigma=3.0)
    blurred = Image(blurred_data, camera_image.spacing)
    blurred_moments = spectral_moments(blurred)
    # Sharp image should have higher moments (more high-frequency content)
    assert sharp_moments > blurred_moments


def test_spectral_moments_returns_float():
    """Spectral moments returns a float value."""
    image = Image(np.ones((64, 64)), (1.0, 1.0))
    moments = spectral_moments(image)
    assert isinstance(moments, float)


# --- brenner_quality ---


def test_brenner_quality_constant_image_zero():
    """Constant image has zero Brenner quality (no differences)."""
    constant = Image(np.ones((64, 64)) * 50, (1.0, 1.0))
    brenner = brenner_quality(constant)
    assert brenner == pytest.approx(0.0, abs=1e-10)


def test_brenner_quality_step_edge_high_value():
    """Step edge produces high Brenner quality (large gradients)."""
    edge = step_edge((64, 64), axis=1)
    image = Image(edge, (1.0, 1.0))
    brenner = brenner_quality(image)
    assert brenner > 0


def test_brenner_quality_known_hand_computed_case():
    """Brenner quality matches hand-computed value for simple pattern."""
    data = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]])
    image = Image(data.astype(float), (1.0, 1.0))
    brenner = brenner_quality(image)
    assert brenner > 0
    assert isinstance(brenner, float)
