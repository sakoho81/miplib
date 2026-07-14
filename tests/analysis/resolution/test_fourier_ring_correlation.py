import argparse

import pytest

from miplib.analysis.resolution.fourier_ring_correlation import (
    FRCOptions,
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
