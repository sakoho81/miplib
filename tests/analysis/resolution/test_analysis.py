import numpy as np
import pytest

from miplib.analysis.resolution.analysis import (
    FitType,
    _first_guess,
)
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)

# ---------------------------------------------------------------------------
# _first_guess
# ---------------------------------------------------------------------------


def test_first_guess_normal_crossing():
    x = np.linspace(0, 1, 10)
    y = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    result = _first_guess(x, y, 0.35)
    assert result == pytest.approx(x[5])


def test_first_guess_never_crosses():
    x = np.linspace(0, 1, 10)
    y = np.ones(10) * 0.9
    result = _first_guess(x, y, 0.35)
    assert result is None


def test_first_guess_crosses_at_first_bin():
    x = np.linspace(0, 1, 10)
    y = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = _first_guess(x, y, 0.2)
    assert result == x[0]


@pytest.mark.parametrize(
    "threshold,expected_idx",
    [(0.3, 5), (0.5, 2), (0.7, 1)],
)
def test_first_guess_at_threshold(threshold, expected_idx):
    x = np.linspace(0, 1, 10)
    y = np.array([0.9, 0.8, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0])
    result = _first_guess(x, y, threshold)
    assert result == x[expected_idx]


# ---------------------------------------------------------------------------
# FourierCorrelationAnalysis — integration
# ---------------------------------------------------------------------------


def _make_analysis_options():
    from miplib.analysis.resolution.fourier_ring_correlation import FRCOptions

    return FRCOptions(frc_curve_fit_type=FitType.POLYNOMIAL)


def test_analysis_curve_never_crosses_threshold_does_not_crash():
    data = FourierCorrelationDataCollection()
    ds = FourierCorrelationData()
    n = 50
    ds.correlation["frequency"] = np.linspace(0, 1, n)
    ds.correlation["correlation"] = np.ones(n) * 0.9
    ds.correlation["points-x-bin"] = np.ones(n) * 100
    data[0] = ds

    from miplib.analysis.resolution.analysis import FourierCorrelationAnalysis

    analyzer = FourierCorrelationAnalysis(data, 0.1, _make_analysis_options())
    result = analyzer.execute()
    assert result is not None


def test_analysis_normal_curve_returns_resolution():
    data = FourierCorrelationDataCollection()
    ds = FourierCorrelationData()
    n = 100
    freq = np.linspace(0, 1, n)
    corr = 0.9 - 0.9 * freq + 0.1 * np.sin(freq * np.pi * 2) * 0.02
    ds.correlation["frequency"] = freq
    ds.correlation["correlation"] = corr
    ds.correlation["points-x-bin"] = np.ones(n) * 100
    data[0] = ds

    from miplib.analysis.resolution.analysis import FourierCorrelationAnalysis

    analyzer = FourierCorrelationAnalysis(data, 0.1, _make_analysis_options())
    result = analyzer.execute()
    resolution = result[0].resolution["resolution"]
    assert np.isfinite(resolution)
    assert resolution > 0


@pytest.mark.parametrize("n_angles", [1, 3])
def test_analysis_multiple_datasets(n_angles):
    data = FourierCorrelationDataCollection()
    for angle_idx in range(n_angles):
        ds = FourierCorrelationData()
        n = 50
        freq = np.linspace(0, 1, n)
        ds.correlation["frequency"] = freq
        ds.correlation["correlation"] = 0.9 - 0.9 * freq
        ds.correlation["points-x-bin"] = np.ones(n) * 100
        data[angle_idx * 45] = ds

    from miplib.analysis.resolution.analysis import FourierCorrelationAnalysis

    analyzer = FourierCorrelationAnalysis(data, 0.1, _make_analysis_options())
    result = analyzer.execute()
    assert len(result) == n_angles
    for _, ds in result:
        assert np.isfinite(ds.resolution["resolution"])
        assert ds.resolution["resolution"] > 0
