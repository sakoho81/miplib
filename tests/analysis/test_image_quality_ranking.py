import pytest

from miplib.analysis.image_quality.filters import PowerSpectrumStats
from miplib.analysis.image_quality.image_quality_ranking import (
    ImageQualityMetrics,
    evaluate_image_quality,
)
from miplib.data.containers.image import Image
from tests.conftest import gaussian_spot


def test_evaluate_image_quality_returns_correct_structure(camera_image):
    """evaluate_image_quality returns ImageQualityMetrics with correct fields."""
    metrics = evaluate_image_quality(camera_image)
    assert isinstance(metrics, ImageQualityMetrics)
    assert hasattr(metrics, "entropy")
    assert hasattr(metrics, "brenner")
    assert hasattr(metrics, "spectral_moments")
    assert hasattr(metrics, "power_stats")
    assert isinstance(metrics.power_stats, PowerSpectrumStats)


def test_evaluate_image_quality_deterministic(camera_image):
    """Same image produces same metrics."""
    metrics1 = evaluate_image_quality(camera_image)
    metrics2 = evaluate_image_quality(camera_image)
    assert metrics1.entropy == pytest.approx(metrics2.entropy)
    assert metrics1.brenner == pytest.approx(metrics2.brenner)
    assert metrics1.spectral_moments == pytest.approx(metrics2.spectral_moments)
    assert metrics1.power_stats.mean == pytest.approx(metrics2.power_stats.mean)


def test_evaluate_image_quality_known_values(camera_image):
    """Camera image produces expected metric ranges."""
    metrics = evaluate_image_quality(camera_image)
    # Camera image has significant detail, so entropy should be moderate to high
    assert 4.0 < metrics.entropy < 8.0
    # Brenner gradient should be positive (has edges)
    assert metrics.brenner > 1000
    # Spectral moments should be positive
    assert metrics.spectral_moments > 0
    # Power spectrum should have significant high-frequency content
    assert metrics.power_stats.mean > 0
    assert metrics.power_stats.power_at_high_freq > 0


def test_evaluate_image_quality_gaussian_spot():
    """Gaussian spot produces reasonable metrics."""
    g = gaussian_spot((64, 64), sigma=3.0)
    image = Image(g, (1.0, 1.0))
    metrics = evaluate_image_quality(image)
    assert metrics.entropy > 0
    assert metrics.brenner > 0
    assert metrics.power_stats.mean > 0
