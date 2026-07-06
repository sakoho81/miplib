import numpy as np
import pytest

from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image


def _make_image(value=1.0):
    """Helper: create a small 5x5 Image with a constant value."""
    return Image(np.ones((5, 5), dtype=np.float64) * value, spacing=(0.1, 0.1))


def _make_populated():
    """Helper: create a 2x2 ArrayDetectorData fully populated with distinct images."""
    add = ArrayDetectorData(2, 2)
    for g in range(2):
        for d in range(2):
            add[g, d] = _make_image(g * 2 + d)
    return add


def test_construction_dimensions():
    add = ArrayDetectorData(3, 5)
    assert add.ndetectors == 3
    assert add.ngates == 5


def test_setitem_getitem_roundtrip():
    add = ArrayDetectorData(1, 1)
    img = _make_image()
    add[0, 0] = img
    result = add[0, 0]
    assert result.shape == (5, 5)
    assert result.spacing == [0.1, 0.1]
    np.testing.assert_array_equal(result, img)


def test_cells_are_independent_different_gates():
    """Verify the list-aliasing bug is fixed: setting [0,0] does not affect [1,0]."""
    add = ArrayDetectorData(2, 2)
    img_a = _make_image(1.0)
    img_b = _make_image(99.0)
    add[0, 0] = img_a
    add[1, 0] = img_b
    assert add[0, 0][0, 0] == 1.0
    assert add[1, 0][0, 0] == 99.0


def test_cells_are_independent_different_detectors():
    add = ArrayDetectorData(2, 2)
    img_a = _make_image(1.0)
    img_b = _make_image(99.0)
    add[0, 0] = img_a
    add[0, 1] = img_b
    assert add[0, 0][0, 0] == 1.0
    assert add[0, 1][0, 0] == 99.0


def test_setitem_rejects_non_tuple_key():
    add = _make_populated()
    with pytest.raises(TypeError, match="2-tuple"):
        add[0] = _make_image()


def test_setitem_rejects_wrong_length_key():
    add = _make_populated()
    with pytest.raises(TypeError, match="2-tuple"):
        add[0, 1, 2] = _make_image()


def test_setitem_rejects_non_image():
    add = _make_populated()
    with pytest.raises(TypeError, match="Image"):
        add[0, 0] = np.ones((5, 5))


def test_getitem_rejects_non_tuple_key():
    add = _make_populated()
    with pytest.raises(TypeError, match="2-tuple"):
        _ = add[0]


def test_getitem_rejects_wrong_length_key():
    add = _make_populated()
    with pytest.raises(TypeError, match="2-tuple"):
        _ = add[0, 1, 2]


def test_getitem_bounds_check_gate():
    add = _make_populated()
    with pytest.raises(IndexError, match="out of range"):
        _ = add[99, 0]


def test_getitem_bounds_check_detector():
    add = _make_populated()
    with pytest.raises(IndexError, match="out of range"):
        _ = add[0, 99]


def test_iteration_detectors_axis_yields_all():
    """With iteration_axis='detectors', inner loop is detector."""
    add = _make_populated()
    add.iteration_axis = "detectors"
    items = list(add)
    assert len(items) == 4


def test_iteration_gates_axis_yields_all():
    """With iteration_axis='gates', inner loop is gate. Verifies the bug
    where gates-first iteration never executed is now fixed."""
    add = _make_populated()
    add.iteration_axis = "gates"
    items = list(add)
    assert len(items) == 4


def test_iteration_exhaustion_resets():
    add = _make_populated()
    list(add)
    items = list(add)
    assert len(items) == 4


def test_iteration_axis_valid_value():
    add = _make_populated()
    add.iteration_axis = "detectors"
    add.iteration_axis = "gates"


def test_iteration_axis_invalid_value():
    add = _make_populated()
    with pytest.raises(ValueError, match="detectors or gates"):
        add.iteration_axis = "bad"


def test_get_photosensor_dimensions():
    add = _make_populated()
    result = add.get_photosensor(0)
    assert result.ndetectors == 2
    assert result.ngates == 1


def test_get_photosensor_data():
    add = _make_populated()
    result = add.get_photosensor(1)
    # gate=1, detector=0 has value 2.0; gate=1, detector=1 has value 3.0
    assert result[0, 0][0, 0] == pytest.approx(2.0)
    assert result[0, 1][0, 0] == pytest.approx(3.0)
