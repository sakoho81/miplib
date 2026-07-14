import numpy as np
import pytest

from miplib.analysis.image_quality.filters import PowerSpectrumStats
from miplib.analysis.image_quality.image_quality_ranking import (
    ImageQualityMetrics,
    evaluate_image_quality,
)
from miplib.data.containers.image import Image
from tests.conftest import gaussian_spot


def test_evaluate_image_quality_returns_correct_structure():
    """evaluate_image_quality returns ImageQualityMetrics with correct fields."""
    rng = np.random.default_rng(42)
    image = Image(rng.random((64, 64)), (1.0, 1.0))
    metrics = evaluate_image_quality(image)
    assert isinstance(metrics, ImageQualityMetrics)
    assert hasattr(metrics, "entropy")
    assert hasattr(metrics, "brenner")
    assert hasattr(metrics, "spectral_moments")
    assert hasattr(metrics, "power_stats")
    assert isinstance(metrics.power_stats, PowerSpectrumStats)


def test_evaluate_image_quality_deterministic():
    """Same image produces same metrics."""
    rng = np.random.default_rng(42)
    image = Image(rng.random((64, 64)), (1.0, 1.0))
    metrics1 = evaluate_image_quality(image)
    metrics2 = evaluate_image_quality(image)
    assert metrics1.entropy == pytest.approx(metrics2.entropy)
    assert metrics1.brenner == pytest.approx(metrics2.brenner)
    assert metrics1.spectral_moments == pytest.approx(metrics2.spectral_moments)
    assert metrics1.power_stats.mean == pytest.approx(metrics2.power_stats.mean)


def test_evaluate_image_quality_all_metrics_finite():
    """All metrics are finite (not NaN or inf)."""
    rng = np.random.default_rng(42)
    image = Image(rng.random((64, 64)), (1.0, 1.0))
    metrics = evaluate_image_quality(image)
    assert np.isfinite(metrics.entropy)
    assert np.isfinite(metrics.brenner)
    assert np.isfinite(metrics.spectral_moments)
    assert np.isfinite(metrics.power_stats.mean)
    assert np.isfinite(metrics.power_stats.std)
    assert np.isfinite(metrics.power_stats.entropy)
    assert np.isfinite(metrics.power_stats.threshold_freq)
    assert np.isfinite(metrics.power_stats.power_at_high_freq)
    assert np.isfinite(metrics.power_stats.skew)
    assert np.isfinite(metrics.power_stats.kurtosis)
    assert np.isfinite(metrics.power_stats.mean_bin)


def test_evaluate_image_quality_gaussian_spot():
    """Gaussian spot produces reasonable metrics."""
    g = gaussian_spot((64, 64), sigma=3.0)
    image = Image(g, (1.0, 1.0))
    metrics = evaluate_image_quality(image)
    assert metrics.entropy > 0
    assert metrics.brenner > 0
    assert metrics.power_stats.mean > 0
