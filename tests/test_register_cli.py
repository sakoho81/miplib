"""Tests for miplib/bin/register.py CLI helpers."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from miplib.bin.register import _build_options, _build_parser, _resolve_source
from miplib.data.adapters.registration_data import ArrayRegistrationDataSource
from miplib.processing.registration import RegistrationMethod
from miplib.processing.registration.options import Metric

# ---------------------------------------------------------------------------
# _build_parser
# ---------------------------------------------------------------------------


def test_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args(["a.tif", "b.tif"])
    assert args.images == ["a.tif", "b.tif"]
    assert args.method == RegistrationMethod.ITERATIVE_RIGID
    assert args.fixed_idx == 0
    assert args.learning_rate == 0.7
    assert args.max_iterations == 200
    assert args.metric == "correlation"
    assert args.translate_only is False
    assert args.overwrite is False
    assert args.verbose is False


def test_parser_method_choices():
    parser = _build_parser()
    args = parser.parse_args(["a.tif", "b.tif", "--method", "phase_correlation"])
    assert args.method == RegistrationMethod.PHASE_CORRELATION

    args = parser.parse_args(["a.tif", "b.tif", "--method", "affine"])
    assert args.method == RegistrationMethod.ITERATIVE_AFFINE


def test_parser_method_rejects_invalid():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["a.tif", "b.tif", "--method", "nonexistent"])


def test_parser_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "a.tif",
            "b.tif",
            "--translate-only",
            "--use-initializer",
            "--overwrite",
            "--verbose",
        ]
    )
    assert args.translate_only is True
    assert args.use_initializer is True
    assert args.overwrite is True
    assert args.verbose is True


def test_parser_output_options():
    parser = _build_parser()
    args = parser.parse_args(
        ["a.tif", "b.tif", "--output", "out1.tif", "out2.tif", "--output-dir", "/tmp"]
    )
    assert args.output == ["out1.tif", "out2.tif"]
    assert args.output_dir == "/tmp"


# ---------------------------------------------------------------------------
# _resolve_source
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_image_dir() -> str:
    """Create a temp directory with two MHA images (avoids bioformats/JVM)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = np.random.default_rng(42 + i).random((32, 32)).astype(np.float32)
            sitk.WriteImage(
                sitk.GetImageFromArray(img), str(Path(tmpdir) / f"img_{i:02d}.mha")
            )
        yield tmpdir


def test_resolve_source_file_list(tmp_image_dir: str):
    """Multiple positional args → file mode."""
    paths = sorted(str(p) for p in Path(tmp_image_dir).glob("*.mha"))[:2]
    args = argparse.Namespace(images=paths, fixed_idx=0)
    source = _resolve_source(args)
    assert isinstance(source, ArrayRegistrationDataSource)
    assert source.n_views == 2


def test_resolve_source_directory(tmp_image_dir: str):
    """Single directory arg → dir mode (glob .mha files)."""
    args = argparse.Namespace(images=[tmp_image_dir], fixed_idx=0)
    source = _resolve_source(args)
    assert isinstance(source, ArrayRegistrationDataSource)
    assert source.n_views == 3


def test_resolve_source_directory_no_images():
    """Empty directory → exits with error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        args = argparse.Namespace(images=[tmpdir], fixed_idx=0)
        with pytest.raises(SystemExit):
            _resolve_source(args)


# ---------------------------------------------------------------------------
# _build_options
# ---------------------------------------------------------------------------


def test_build_options_defaults():
    args = argparse.Namespace(
        method=RegistrationMethod.ITERATIVE_RIGID,
        learning_rate=0.7,
        min_step_length=0.001,
        max_iterations=200,
        relaxation_factor=0.7,
        metric=Metric.CORRELATION,
        sampling_percentage=1.0,
        mattes_histogram_bins=15,
        subpixel=100,
        translate_only=False,
        use_initializer=False,
    )
    opts = _build_options(args)
    assert opts.method == RegistrationMethod.ITERATIVE_RIGID
    assert opts.learning_rate == 0.7
    assert opts.max_iterations == 200
    assert opts.metric == Metric.CORRELATION


def test_build_options_overrides():
    args = argparse.Namespace(
        method=RegistrationMethod.ITERATIVE_AFFINE,
        learning_rate=0.5,
        min_step_length=0.0001,
        max_iterations=500,
        relaxation_factor=0.3,
        metric=Metric.MATTES,
        sampling_percentage=0.5,
        mattes_histogram_bins=30,
        subpixel=200,
        translate_only=True,
        use_initializer=True,
    )
    opts = _build_options(args)
    assert opts.method == RegistrationMethod.ITERATIVE_AFFINE
    assert opts.learning_rate == 0.5
    assert opts.max_iterations == 500
    assert opts.metric == Metric.MATTES
    assert opts.subpixel == 200
    assert opts.translate_only is True
    assert opts.use_initializer is True


def test_build_options_partial_overrides():
    args = argparse.Namespace(
        method=RegistrationMethod.ITERATIVE_RIGID,
        learning_rate=1.2,
        min_step_length=0.001,
        max_iterations=200,
        relaxation_factor=0.7,
        metric=Metric.CORRELATION,
        sampling_percentage=1.0,
        mattes_histogram_bins=15,
        subpixel=50,
        translate_only=True,
        use_initializer=False,
    )
    opts = _build_options(args)
    assert opts.learning_rate == 1.2  # overridden
    assert opts.subpixel == 50  # overridden
    assert opts.translate_only is True  # overridden
    assert opts.max_iterations == 200  # default preserved
    assert opts.metric == Metric.CORRELATION  # default preserved
