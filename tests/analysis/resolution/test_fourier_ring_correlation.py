import argparse

import numpy as np
import pytest

from miplib.analysis.resolution.fourier_ring_correlation import (
    FRCOptions,
    accumulate_fourier_correlation,
    build_correlation_curve,
    create_fourier_iterator,
    namespace_to_frc_options,
)


class TestFRCOptions:
    def test_default_construction(self):
        opts = FRCOptions()
        assert opts.d_bin == 1.0
        assert opts.disable_hamming is False
        assert opts.frc_curve_fit_degree == 8
        assert opts.frc_curve_fit_type == "spline"
        assert opts.resolution_threshold_criterion == "fixed"
        assert opts.resolution_threshold_value == pytest.approx(1.0 / 7)
        assert opts.resolution_snr_value == 0.25

    def test_explicit_construction(self):
        opts = FRCOptions(d_bin=2.0, disable_hamming=True)
        assert opts.d_bin == 2.0
        assert opts.disable_hamming is True
        assert opts.frc_curve_fit_degree == 8  # default

    def test_partial_construction(self):
        opts = FRCOptions(frc_curve_fit_degree=12)
        assert opts.frc_curve_fit_degree == 12
        assert opts.d_bin == 1.0  # default

    def test_namespace_to_frc_options_full(self):
        ns = argparse.Namespace(
            d_bin=3.0,
            disable_hamming=True,
            frc_curve_fit_degree=12,
            frc_curve_fit_type="polynomial",
            resolution_threshold_criterion="half-bit",
            resolution_threshold_value=0.5,
            resolution_snr_value=0.5,
        )
        opts = namespace_to_frc_options(ns)
        assert opts.d_bin == 3.0
        assert opts.disable_hamming is True
        assert opts.frc_curve_fit_degree == 12
        assert opts.frc_curve_fit_type == "polynomial"
        assert opts.resolution_threshold_criterion == "half-bit"
        assert opts.resolution_threshold_value == 0.5
        assert opts.resolution_snr_value == 0.5

    def test_namespace_to_frc_options_partial(self):
        ns = argparse.Namespace(d_bin=5.0)
        opts = namespace_to_frc_options(ns)
        assert opts.d_bin == 5.0

    def test_namespace_to_frc_options_unknown_attrs_ignored(self):
        ns = argparse.Namespace(d_bin=2.0, unknown_attr=42)
        opts = namespace_to_frc_options(ns)
        assert opts.d_bin == 2.0

    def test_namespace_to_frc_options_empty_namespace(self):
        ns = argparse.Namespace()
        opts = namespace_to_frc_options(ns)
        assert opts.d_bin == 1.0  # all defaults


class TestCreateFourierIterator:
    def test_2d_shape_returns_ring_iterator(self):
        iterator = create_fourier_iterator((64, 64), d_bin=1.0)
        from miplib.data.iterators.fourier_ring_iterators import FourierRingIterator

        assert isinstance(iterator, FourierRingIterator)

    def test_3d_shape_raises(self):
        with pytest.raises(ValueError, match="Unsupported dimensionality"):
            create_fourier_iterator((32, 32, 32))

    def test_1d_shape_raises(self):
        with pytest.raises(ValueError, match="Unsupported dimensionality"):
            create_fourier_iterator((64,))


class TestAccumulateFourierCorrelation:
    def test_identical_ffts_give_c1_eq_c2_eq_c3(self):
        shape = (32, 32)
        iterator = create_fourier_iterator(shape, d_bin=2.0)
        fft = np.ones(shape, dtype=np.complex64)
        c1, c2, c3, n_points = accumulate_fourier_correlation(fft, fft, iterator)
        assert np.allclose(c1, c2)
        assert np.allclose(c1, c3)

    def test_orthogonal_ffts_give_zero_c1(self):
        shape = (32, 32)
        iterator = create_fourier_iterator(shape, d_bin=2.0)
        fft1 = np.ones(shape, dtype=np.complex64)
        fft2 = np.zeros(shape, dtype=np.complex64)
        fft2[0, 0] = 1.0  # one common point
        c1, c2, c3, n_points = accumulate_fourier_correlation(fft1, fft2, iterator)
        assert np.all(c1 >= 0.0)
        assert np.all(n_points > 0)

    def test_n_points_positive(self):
        shape = (16, 16)
        iterator = create_fourier_iterator(shape, d_bin=1.0)
        fft = np.ones(shape, dtype=np.complex64)
        _, _, _, n_points = accumulate_fourier_correlation(fft, fft, iterator)
        assert np.all(n_points > 0)

    def test_real_input_raises_no_error(self):
        shape = (16, 16)
        iterator = create_fourier_iterator(shape, d_bin=1.0)
        fft = np.random.randn(*shape).astype(np.complex64)
        c1, c2, c3, n_points = accumulate_fourier_correlation(fft, fft, iterator)
        assert len(c1) == len(c2) == len(c3) == len(n_points)


class TestBuildCorrelationCurve:
    def test_identical_inputs_give_correlation_one(self):
        radii = np.arange(10, dtype=np.float32)
        c1 = c2 = c3 = np.ones(10, dtype=np.float32)
        freq, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
        assert np.allclose(corr, 1.0)

    def test_zero_correlated_gives_zero(self):
        radii = np.arange(10, dtype=np.float32)
        c1 = np.zeros(10, dtype=np.float32)
        c2 = c3 = np.ones(10, dtype=np.float32)
        freq, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
        assert np.allclose(corr, 0.0)

    def test_correlation_bounded(self):
        radii = np.arange(10, dtype=np.float32)
        c1 = np.array(
            [0.0, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.0], dtype=np.float32
        )
        c2 = c3 = np.ones(10, dtype=np.float32)
        freq, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
        assert np.all(corr >= 0.0)
        assert np.all(corr <= 1.0)

    def test_frequency_range(self):
        radii = np.arange(10, dtype=np.float32)
        c1 = c2 = c3 = np.ones(10, dtype=np.float32)
        freq, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=10.0)
        assert np.allclose(freq, radii / 10.0)
        assert freq[0] == 0.0
        assert freq[-1] == 0.9

    def test_inf_handling(self):
        radii = np.arange(5, dtype=np.float32)
        c1 = np.ones(5, dtype=np.float32)
        c2 = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        c3 = np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        freq, corr = build_correlation_curve(c1, c2, c3, radii, nyquist=5.0)
        assert np.isfinite(corr).all()
        assert corr[0] == 0.0  # 0/0 → nan_to_num → 0
