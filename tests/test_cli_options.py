from pathlib import Path

from miplib.ui.cli.miplib_entry_point_options import (
    get_power_options,
    get_quality_options,
)


def test_quality_options_defaults():
    """Quality CLI has correct defaults."""
    opts = get_quality_options(["/some/path"])
    assert opts.input == Path("/some/path")
    assert opts.output is None
    assert opts.rgb_channel == 1


def test_quality_options_with_output():
    """Quality CLI accepts output path."""
    opts = get_quality_options(["/some/path", "-o", "/output.csv"])
    assert opts.input == Path("/some/path")
    assert opts.output == Path("/output.csv")


def test_quality_options_rgb_channel():
    """Quality CLI accepts RGB channel selection."""
    opts = get_quality_options(["/some/path", "--rgb-channel", "0"])
    assert opts.rgb_channel == 0


def test_power_options_defaults():
    """Power CLI has correct defaults."""
    opts = get_power_options(["/some/dir"])
    assert opts.input == Path("/some/dir")
    assert opts.output is None
    assert opts.image_size == 512
    assert opts.rgb_channel == 1


def test_power_options_with_output():
    """Power CLI accepts output path."""
    opts = get_power_options(["/some/dir", "-o", "/output.csv"])
    assert opts.input == Path("/some/dir")
    assert opts.output == Path("/output.csv")


def test_power_options_image_size():
    """Power CLI accepts image size."""
    opts = get_power_options(["/some/dir", "--image-size", "256"])
    assert opts.image_size == 256
