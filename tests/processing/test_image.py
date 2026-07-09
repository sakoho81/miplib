"""Tests for miplib/processing/image.py.

Coverage targets:
  - Active functions used by the rest of the library:
      zoom_to_spacing, resize, zero_pad_to_shape, zero_pad_to_matching_shape,
      remove_zero_padding, checkerboard_split, reverse_checkerboard_split,
      zero_pad_to_cube, crop_to_largest_square
  - translate_image: FFT-based circular shift (verifies correct fftshift pairing)
"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from miplib.data.containers.image import Image
from miplib.processing.image import (
    checkerboard_split,
    crop_to_largest_square,
    remove_zero_padding,
    resize,
    reverse_checkerboard_split,
    translate_image,
    zero_pad_to_cube,
    zero_pad_to_matching_shape,
    zero_pad_to_shape,
    zoom_to_spacing,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _impulse(shape, spacing):
    """Single 1.0 at the centre of the array, zeros elsewhere."""
    arr = np.zeros(shape, dtype=np.float64)
    arr[tuple(s // 2 for s in shape)] = 1.0
    return Image(arr, spacing=spacing)


# ---------------------------------------------------------------------------
# zoom_to_spacing
# ---------------------------------------------------------------------------


def test_zoom_to_spacing_output_spacing():
    """Output spacing must exactly match the requested target spacing."""
    img = Image(np.ones((20, 20), dtype=np.float64), spacing=(0.2, 0.2))
    out = zoom_to_spacing(img, (0.1, 0.1))
    assert out.spacing == [0.1, 0.1]


def test_zoom_to_spacing_shape_scales_with_zoom():
    """Halving the spacing should double the number of pixels per axis."""
    img = Image(np.ones((20, 20), dtype=np.float64), spacing=(0.2, 0.2))
    out = zoom_to_spacing(img, (0.1, 0.1))
    assert out.shape == (40, 40)


def test_zoom_to_spacing_wrong_ndim():
    img = Image(np.ones((20, 20), dtype=np.float64), spacing=(0.2, 0.2))
    with pytest.raises(ValueError):
        zoom_to_spacing(img, (0.1,))


def test_zoom_to_spacing_type_error():
    with pytest.raises(TypeError):
        zoom_to_spacing(np.ones((10, 10)), (0.1, 0.1))


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------


def test_resize_output_shape():
    img = Image(np.ones((20, 20), dtype=np.float64), spacing=(1.0, 1.0))
    out = resize(img, (40, 10))
    assert out.shape == (40, 10)


def test_resize_spacing_adjusts_inversely_to_zoom():
    """If a dimension doubles in pixels, its spacing should halve."""
    img = Image(np.ones((20, 20), dtype=np.float64), spacing=(1.0, 1.0))
    out = resize(img, (40, 20))
    assert pytest.approx(out.spacing[0], rel=1e-6) == 0.5
    assert pytest.approx(out.spacing[1], rel=1e-6) == 1.0


def test_resize_type_error():
    with pytest.raises(TypeError):
        resize(np.ones((10, 10)), (20, 20))


# ---------------------------------------------------------------------------
# zero_pad_to_shape / remove_zero_padding (round-trip)
# ---------------------------------------------------------------------------


def test_zero_pad_round_trip_2d():
    """Padding then removing must recover the original data exactly."""
    data = np.arange(9, dtype=np.float64).reshape(3, 3)
    img = Image(data, spacing=(0.5, 0.5))
    padded = zero_pad_to_shape(img, (7, 7))
    assert padded.shape == (7, 7)
    recovered = remove_zero_padding(padded, (3, 3))
    assert_array_almost_equal(recovered, data)
    assert recovered.spacing == [0.5, 0.5]


def test_zero_pad_round_trip_3d():
    data = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    img = Image(data, spacing=(0.2, 0.1, 0.1))
    padded = zero_pad_to_shape(img, (9, 9, 9))
    assert padded.shape == (9, 9, 9)
    recovered = remove_zero_padding(padded, (3, 3, 3))
    assert_array_almost_equal(recovered, data)


def test_zero_pad_to_shape_preserves_spacing():
    img = Image(np.ones((4, 4), dtype=np.float64), spacing=(0.3, 0.3))
    out = zero_pad_to_shape(img, (8, 8))
    assert out.spacing == [0.3, 0.3]


def test_zero_pad_to_shape_noop_when_same():
    img = Image(np.ones((4, 4), dtype=np.float64), spacing=(1.0, 1.0))
    out = zero_pad_to_shape(img, (4, 4))
    assert out is img


def test_zero_pad_to_shape_type_error():
    with pytest.raises(TypeError):
        zero_pad_to_shape(np.ones((4, 4)), (8, 8))


def test_remove_zero_padding_type_error():
    with pytest.raises(TypeError):
        remove_zero_padding(np.ones((8, 8)), (4, 4))


# ---------------------------------------------------------------------------
# zero_pad_to_matching_shape
# ---------------------------------------------------------------------------


def test_zero_pad_to_matching_shape_equal_output_shape():
    """Both images padded to the larger shape; each retains its values."""
    img1 = Image(np.full((4, 6), 2.0), spacing=(1.0, 1.0))
    img2 = Image(np.full((6, 4), 7.0), spacing=(1.0, 1.0))
    out1, out2 = zero_pad_to_matching_shape(img1, img2)
    assert out1.shape == out2.shape == (6, 6)
    # Data is centred in the expanded array (asymmetric when difference is odd).
    # (4, 6) → (6, 6): 1 row of padding top, 1 row bottom.
    assert_array_almost_equal(out1[1:5, :], np.full((4, 6), 2.0))
    assert_array_almost_equal(out1[0, :], np.zeros(6))
    assert_array_almost_equal(out1[5, :], np.zeros(6))
    # (6, 4) → (6, 6): 1 col of padding left, 1 col right.
    assert_array_almost_equal(out2[:, 1:5], np.full((6, 4), 7.0))
    assert_array_almost_equal(out2[:, 0], np.zeros(6))
    assert_array_almost_equal(out2[:, 5], np.zeros(6))


def test_zero_pad_to_matching_shape_already_equal():
    img1 = Image(np.ones((5, 5), dtype=np.float64), spacing=(1.0, 1.0))
    img2 = Image(np.ones((5, 5), dtype=np.float64), spacing=(1.0, 1.0))
    out1, out2 = zero_pad_to_matching_shape(img1, img2)
    assert out1.shape == out2.shape == (5, 5)


def test_zero_pad_to_matching_shape_type_error():
    img = Image(np.ones((4, 4), dtype=np.float64), spacing=(1.0, 1.0))
    with pytest.raises(TypeError):
        zero_pad_to_matching_shape(np.ones((4, 4)), img)
    with pytest.raises(TypeError):
        zero_pad_to_matching_shape(img, np.ones((4, 4)))


# ---------------------------------------------------------------------------
# checkerboard_split
# ---------------------------------------------------------------------------


def test_checkerboard_split_2d_from_checkerboard():
    """Splitting a 0/1 checkerboard must produce constant-valued halves.

    The forward split selects pixel positions whose row+col sum is even
    (odd/odd and even/even pairs). On a checkerboard where (r+c) % 2 == 0
    marks 1 and the rest 0, both halves are all-1s.
    """
    n = 8
    y, x = np.mgrid[:n, :n]
    checkerboard = np.where((x + y) % 2 == 0, 1.0, 0.0)
    img = Image(checkerboard, spacing=(1.0, 1.0))
    h1, h2 = checkerboard_split(img)
    assert h1.shape == (4, 4)
    assert h2.shape == (4, 4)
    assert_array_almost_equal(h1, np.ones((4, 4)))
    assert_array_almost_equal(h2, np.ones((4, 4)))


def test_checkerboard_split_2d_disjoint_indices():
    """The two halves must sample different pixel positions (disjoint sets)."""
    n = 8
    data = np.arange(n * n, dtype=np.float64).reshape(n, n)
    img = Image(data, spacing=(1.0, 1.0))
    h1, h2 = checkerboard_split(img)
    assert len(np.intersect1d(h1.ravel(), h2.ravel())) == 0


def test_checkerboard_split_preserves_spacing():
    img = Image(np.ones((8, 8), dtype=np.float64), spacing=(0.5, 0.5))
    h1, h2 = checkerboard_split(img)
    assert h1.spacing == [0.5, 0.5]
    assert h2.spacing == [0.5, 0.5]


def test_checkerboard_split_type_error():
    with pytest.raises(TypeError):
        checkerboard_split(np.ones((8, 8)))


def test_reverse_checkerboard_split_2d_from_checkerboard():
    """Reverse split on a 0/1 checkerboard must produce constant-0 halves.

    The reverse split selects odd/even and even/odd pairs, whose row+col
    sums are odd — positions where the checkerboard value is 0.
    """
    n = 8
    y, x = np.mgrid[:n, :n]
    checkerboard = np.where((x + y) % 2 == 0, 1.0, 0.0)
    img = Image(checkerboard, spacing=(1.0, 1.0))
    h1, h2 = reverse_checkerboard_split(img)
    assert h1.shape == (4, 4)
    assert h2.shape == (4, 4)
    assert_array_almost_equal(h1, np.zeros((4, 4)))
    assert_array_almost_equal(h2, np.zeros((4, 4)))


def test_reverse_vs_forward_checkerboard_sample_different_pixels():
    """Reverse split must select different pixels than the forward split."""
    n = 8
    data = np.arange(n * n, dtype=np.float64).reshape(n, n)
    img = Image(data, spacing=(1.0, 1.0))
    fwd1, fwd2 = checkerboard_split(img)
    rev1, rev2 = reverse_checkerboard_split(img)
    # fwd1 samples odd-row/odd-col; rev1 samples odd-row/even-col — disjoint
    assert len(np.intersect1d(fwd1.ravel(), rev1.ravel())) == 0
    assert len(np.intersect1d(fwd2.ravel(), rev2.ravel())) == 0


def test_reverse_checkerboard_split_type_error():
    with pytest.raises(TypeError):
        reverse_checkerboard_split(np.ones((8, 8)))


# ---------------------------------------------------------------------------
# zero_pad_to_cube
# ---------------------------------------------------------------------------


def test_zero_pad_to_cube_produces_cubic_shape():
    img = Image(np.ones((4, 6, 8), dtype=np.float64), spacing=(1.0, 1.0, 1.0))
    out = zero_pad_to_cube(img)
    assert out.shape[0] == out.shape[1] == out.shape[2] == 8


def test_zero_pad_to_cube_noop_on_cube():
    img = Image(np.ones((5, 5, 5), dtype=np.float64), spacing=(1.0, 1.0, 1.0))
    out = zero_pad_to_cube(img)
    assert out is img


def test_zero_pad_to_cube_preserves_data():
    data = np.zeros((4, 4, 4), dtype=np.float64)
    data[2, 2, 2] = 1.0
    img = Image(data, spacing=(1.0, 1.0, 1.0))
    out = zero_pad_to_cube(img)
    # The original impulse must still be present somewhere in the cube
    assert out.max() == pytest.approx(1.0)


def test_zero_pad_to_cube_type_error():
    with pytest.raises(TypeError):
        zero_pad_to_cube(np.ones((4, 6)))


# ---------------------------------------------------------------------------
# crop_to_largest_square
# ---------------------------------------------------------------------------


def test_crop_to_largest_square_2d_pixel_dims():
    """Must crop centrally to the largest square, preserving pixel values."""
    data = np.arange(60, dtype=np.float64).reshape(10, 6)
    img = Image(data, spacing=(1.0, 1.0))
    out = crop_to_largest_square(img)
    assert out.shape[0] == out.shape[1] == 6
    # Central crop: removes 2 rows from each end of axis 0, none from axis 1
    assert_array_almost_equal(out, data[2:8, :])


def test_crop_to_largest_square_already_square():
    img = Image(np.ones((5, 5), dtype=np.float64), spacing=(1.0, 1.0))
    out = crop_to_largest_square(img)
    assert out.shape == (5, 5)


def test_crop_to_largest_square_type_error():
    with pytest.raises(TypeError):
        crop_to_largest_square(np.ones((10, 6)))


# ---------------------------------------------------------------------------
# translate_image
# ---------------------------------------------------------------------------


def test_translate_image_zero_shift_identity():
    """Zero shift must return an image numerically identical to the input."""
    data = np.random.default_rng(0).random((32, 32))
    img = Image(data, spacing=(1.0, 1.0))
    out = translate_image(img, (0.0, 0.0))
    assert_array_almost_equal(out, data, decimal=10)


def test_translate_image_integer_shift_moves_impulse():
    """An impulse shifted by (dy, dx) must have its peak at the new location."""
    shape = (32, 32)
    cy, cx = shape[0] // 2, shape[1] // 2
    img = _impulse(shape, spacing=(1.0, 1.0))

    dy, dx = 5, -3
    out = translate_image(img, (dy, dx))

    peak = np.unravel_index(np.argmax(out), out.shape)
    # Circular shift: peak should move to (cy+dy) % N
    assert peak[0] == (cy + dy) % shape[0]
    assert peak[1] == (cx + dx) % shape[1]


def test_translate_image_integer_shift_matches_roll():
    """An integer shift must match numpy.roll on non-trivial data."""
    n = 32
    data = np.arange(n * n, dtype=np.float64).reshape(n, n)
    img = Image(data, spacing=(1.0, 1.0))

    dy, dx = 3, -7
    out = translate_image(img, (dy, dx))
    expected = np.roll(np.roll(data, dy, axis=0), dx, axis=1)
    assert_array_almost_equal(out, expected, decimal=10)


def test_translate_image_roundtrip():
    """Shifting by Δ then by -Δ must recover the original image."""
    data = np.random.default_rng(7).random((32, 32))
    img = Image(data, spacing=(1.0, 1.0))
    shifted = translate_image(img, (4.0, -6.0))
    recovered = translate_image(shifted, (-4.0, 6.0))
    assert_array_almost_equal(recovered, data, decimal=8)


def test_translate_image_preserves_total_intensity():
    """A circular shift is energy-preserving; the sum must not change."""
    data = np.random.default_rng(3).random((32, 32))
    img = Image(data, spacing=(1.0, 1.0))
    out = translate_image(img, (3.0, 7.0))
    assert pytest.approx(out.sum(), rel=1e-9) == data.sum()


def test_translate_image_preserves_spacing():
    img = Image(np.ones((16, 16), dtype=np.float64), spacing=(0.25, 0.25))
    out = translate_image(img, (2.0, 2.0))
    assert out.spacing == [0.25, 0.25]


def test_translate_image_type_error():
    with pytest.raises(TypeError):
        translate_image(np.ones((16, 16)), (1.0, 1.0))


def test_translate_image_shift_ndim_mismatch():
    img = Image(np.ones((16, 16), dtype=np.float64), spacing=(1.0, 1.0))
    with pytest.raises(ValueError):
        translate_image(img, (1.0,))
