import numpy as np
import pytest

from miplib.analysis.resolution.analysis import _first_guess
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)


class TestFirstGuess:
    def test_normal_crossing(self):
        x = np.linspace(0, 1, 10)
        y = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        result = _first_guess(x, y, 0.35)
        assert result == pytest.approx(x[5])  # crosses between 5th and 6th

    def test_never_crosses(self):
        x = np.linspace(0, 1, 10)
        y = np.ones(10) * 0.9
        result = _first_guess(x, y, 0.35)
        assert result == x[-1]  # returns last frequency

    def test_crosses_at_first_bin(self):
        x = np.linspace(0, 1, 10)
        y = np.array([0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = _first_guess(x, y, 0.2)
        assert result == x[0]  # crosses at first bin, returns x[0]

    def test_exactly_at_threshold(self):
        x = np.linspace(0, 1, 10)
        y = np.array([0.9, 0.8, 0.7, 0.5, 0.5, 0.5, 0.3, 0.2, 0.1, 0.0])
        result = _first_guess(x, y, 0.5)
        assert result == x[2]  # x[3] == 0.5, returns x[2]


class TestFourierCorrelationAnalysis:
    """Integration tests for FourierCorrelationAnalysis."""

    def test_curve_never_crosses_threshold_does_not_crash(self):
        data = FourierCorrelationDataCollection()
        ds = FourierCorrelationData()
        n = 50
        ds.correlation["frequency"] = np.linspace(0, 1, n)
        ds.correlation["correlation"] = np.ones(n) * 0.9
        ds.correlation["points-x-bin"] = np.ones(n) * 100
        data[0] = ds

        from miplib.analysis.resolution.analysis import FourierCorrelationAnalysis

        args = type(
            "Args",
            (),
            {
                "resolution_threshold_criterion": "fixed",
                "resolution_threshold_value": 1.0 / 7,
                "resolution_snr_value": 0.25,
                "frc_curve_fit_degree": 8,
                "frc_curve_fit_type": "spline",
            },
        )()
        analyzer = FourierCorrelationAnalysis(data, 0.1, args)
        result = analyzer.execute()
        assert result is not None
        assert np.isfinite(result[0].resolution["resolution"])

    def test_normal_curve_returns_resolution(self):
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

        args = type(
            "Args",
            (),
            {
                "resolution_threshold_criterion": "fixed",
                "resolution_threshold_value": 1.0 / 7,
                "resolution_snr_value": 0.25,
                "frc_curve_fit_degree": 8,
                "frc_curve_fit_type": "polynomial",
            },
        )()
        analyzer = FourierCorrelationAnalysis(data, 0.1, args)
        result = analyzer.execute()
        resolution = result[0].resolution["resolution"]
        assert np.isfinite(resolution)
        assert resolution > 0

    def test_multiple_datasets(self):
        data = FourierCorrelationDataCollection()
        for angle in (0, 45, 90):
            ds = FourierCorrelationData()
            n = 50
            freq = np.linspace(0, 1, n)
            ds.correlation["frequency"] = freq
            ds.correlation["correlation"] = 0.9 - 0.9 * freq
            ds.correlation["points-x-bin"] = np.ones(n) * 100
            data[angle] = ds

        from miplib.analysis.resolution.analysis import FourierCorrelationAnalysis

        args = type(
            "Args",
            (),
            {
                "resolution_threshold_criterion": "fixed",
                "resolution_threshold_value": 1.0 / 7,
                "resolution_snr_value": 0.25,
                "frc_curve_fit_degree": 8,
                "frc_curve_fit_type": "polynomial",
            },
        )()
        analyzer = FourierCorrelationAnalysis(data, 0.1, args)
        result = analyzer.execute()
        assert len(result) == 3
        for _, ds in result:
            assert np.isfinite(ds.resolution["resolution"])
