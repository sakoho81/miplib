from math import pi

import pytest

from miplib.processing.converters import degrees_to_radians, radians_to_degrees


def test_degrees_to_radians_zero_shortcut():
    assert degrees_to_radians(0) == 0


def test_degrees_to_radians():
    assert degrees_to_radians(180) == pi
    assert degrees_to_radians(360) == 2 * pi
    assert degrees_to_radians(90) == pi / 2
    assert degrees_to_radians(270) == 3 * pi / 2


def test_radians_to_degrees_zero_shortcut():
    assert radians_to_degrees(0) == 0


def test_radians_to_degrees():
    assert radians_to_degrees(pi) == 180
    assert radians_to_degrees(2 * pi) == 360
    assert radians_to_degrees(pi / 2) == 90


def test_roundtrip():
    for deg in (30, 45, 60, 90, 180, 270, 360):
        assert radians_to_degrees(degrees_to_radians(deg)) == pytest.approx(deg)
