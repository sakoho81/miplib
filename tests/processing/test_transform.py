from math import pi

import numpy as np
import numpy.testing as npt
import pytest
import SimpleITK as sitk

from miplib.processing.transform import (
    make_translation_transforms_from_xy,
    rotate_xy_points_lists,
)

# -- make_translation_transforms_from_xy ---------------------------------------


def test_translation_transform_single_point():
    result = make_translation_transforms_from_xy([1.5], [-2.0])
    assert len(result) == 1
    assert isinstance(result[0], sitk.Transform)
    npt.assert_array_almost_equal(result[0].GetParameters(), (1.5, -2.0))


def test_translation_transform_multiple():
    xs = [0.0, 3.0, -1.5]
    ys = [0.0, -2.0, 4.0]
    result = make_translation_transforms_from_xy(xs, ys)
    assert len(result) == 3
    for tfm, (x, y) in zip(result, zip(xs, ys, strict=False)):
        npt.assert_array_almost_equal(tfm.GetParameters(), (x, y))


def test_translation_transform_empty():
    result = make_translation_transforms_from_xy([], [])
    assert result == []


def test_translation_transform_unequal_lengths():
    with pytest.raises(ValueError, match="equal length"):
        make_translation_transforms_from_xy([1, 2], [3])


# -- rotate_xy_points_lists ----------------------------------------------------


def test_rotate_zero_identity():
    xs, ys = [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]
    rx, ry = rotate_xy_points_lists(xs, ys, 0.0)
    npt.assert_array_almost_equal(rx, xs)
    npt.assert_array_almost_equal(ry, ys)


@pytest.mark.parametrize(
    "theta,x,y,expx,expy",
    [
        (pi / 2, 1.0, 0.0, 0.0, -1.0),
        (pi / 2, 0.0, 1.0, 1.0, 0.0),
        (pi, 1.0, 0.0, -1.0, 0.0),
        (pi, 0.0, 1.0, 0.0, -1.0),
    ],
)
def test_rotate_single_point_cardinal(theta, x, y, expx, expy):
    """Rotation by 90/180 degrees yields exact cardinal results."""
    rx, ry = rotate_xy_points_lists([x], [y], theta)
    npt.assert_almost_equal(rx[0], expx)
    npt.assert_almost_equal(ry[0], expy)


def test_rotate_pi_fourth():
    """Rotation by π/4 with easy coordinates: cos(π/4) = sin(π/4) = √2/2."""
    s = np.sqrt(2) / 2
    rx, ry = rotate_xy_points_lists([1.0], [0.0], pi / 4)
    npt.assert_almost_equal(rx[0], s)
    npt.assert_almost_equal(ry[0], -s)


def test_rotate_multiple_points():
    xs = [1.0, 0.0]
    ys = [0.0, 1.0]
    rx, ry = rotate_xy_points_lists(xs, ys, pi / 2)
    npt.assert_array_almost_equal(rx, [0.0, 1.0])
    npt.assert_array_almost_equal(ry, [-1.0, 0.0])


def test_rotate_empty():
    rx, ry = rotate_xy_points_lists([], [], pi / 4)
    assert rx == []
    assert ry == []


def test_rotate_roundtrip_2pi():
    """Full-circle rotation restores original coordinates."""
    xs = [1.5, -0.5, 3.0]
    ys = [2.0, 0.0, -1.0]
    rx, ry = rotate_xy_points_lists(xs, ys, 2 * pi)
    npt.assert_array_almost_equal(rx, xs)
    npt.assert_array_almost_equal(ry, ys)
