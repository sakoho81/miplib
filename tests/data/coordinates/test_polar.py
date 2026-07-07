import numpy as np
import pytest

from miplib.data.coordinates.polar import (
    PolarHighPassIndexer,
    PolarLowPassIndexer,
    SimplePolarIndexer,
    generate_polar_coordinate_grid,
)


def test_generate_polar_coordinate_even_shape():
    shape = (4, 4)
    spacing = (0.1, 0.2)
    axes = generate_polar_coordinate_grid(shape, spacing)
    assert len(axes) == 2
    assert axes[0].shape == (4,)
    assert axes[1].shape == (4,)
    # center element is zero for even shape with symmetric centering
    assert axes[0][2] == pytest.approx(0.0)


def test_generate_polar_coordinate_odd_shape():
    """The bug fix: odd shapes now center symmetrically around 0."""
    shape = (5, 5)
    spacing = (0.1, 0.1)
    axes = generate_polar_coordinate_grid(shape, spacing)
    # n=5: should be [-0.2, -0.1, 0.0, 0.1, 0.2] (symmetric)
    # not    [-0.3, -0.2, -0.1, 0.0, 0.1] (buggy asymmetric)
    assert axes[0][2] == pytest.approx(0.0)  # center element is 0.0


def test_generate_polar_coordinate_spacing_scales():
    shape = (4,)
    spacing = (2.0,)
    axes = generate_polar_coordinate_grid(shape, spacing)
    assert axes[0][0] == pytest.approx(-4.0)  # -2 * 2.0
    assert axes[0][-1] == pytest.approx(2.0)  # 1 * 2.0


def test_simple_polar_indexer_2d_shape():
    """meshgrid with default indexing='xy' swaps axis order."""
    idx = SimplePolarIndexer((8, 16))
    assert idx.r.shape == (16, 8)


def test_simple_polar_indexer_3d_shape():
    idx = SimplePolarIndexer((4, 8, 16))
    assert idx.r.shape == (8, 4, 16)


def test_simple_polar_indexer_center_is_zero():
    idx = SimplePolarIndexer((5, 5))
    assert idx.r[2, 2] == pytest.approx(0.0)


def test_simple_polar_indexer_corners_are_nonzero():
    idx = SimplePolarIndexer((5, 5))
    assert idx.r[0, 0] > 0.0


def test_simple_polar_indexer_rejects_1d_shape():
    with pytest.raises(ValueError, match="2D or 3D"):
        SimplePolarIndexer((8,))


def test_simple_polar_indexer_rejects_4d_shape():
    with pytest.raises(ValueError, match="2D or 3D"):
        SimplePolarIndexer((4, 4, 4, 4))


def test_low_pass_mask_selects_inner_region():
    idx = PolarLowPassIndexer((16, 16))
    mask = idx[2.0]
    assert mask[8, 8]  # center is within radius 2
    assert not mask[0, 0]  # corner is far outside


def test_high_pass_mask_selects_outer_region():
    idx = PolarHighPassIndexer((16, 16))
    mask = idx[4.0]
    assert mask[0, 0]  # corner is beyond radius 4
    assert not mask[8, 8]  # center is within radius 4


def test_low_pass_and_high_pass_are_complementary():
    shape = (8, 8)
    low_mask = PolarLowPassIndexer(shape)[3.0]
    high_mask = PolarHighPassIndexer(shape)[3.0]
    # at r=3 exactly, neither matches (low: r<3, high: r>3)
    # but the rest of the array should be complementary where r!=3
    on_ring = low_mask | high_mask
    radius = SimplePolarIndexer(shape).r
    assert np.all(on_ring | (radius == 3.0))
