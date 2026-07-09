from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image_data import ImageData
from miplib.processing.fusion_utils import average_of_all, simple_fusion, sum_of_all


def _make_mock_data(arrays: list[np.ndarray], spacing=(1.0, 1.0)):
    data = MagicMock(spec=ImageData)
    data.get_number_of_images.return_value = len(arrays)
    data.get_image_size.return_value = arrays[0].shape
    data.get_voxel_size.return_value = list(spacing)
    data.get_image_data.side_effect = arrays
    return data


@pytest.fixture
def three_arrays():
    rng = np.random.default_rng(42)
    return [rng.normal(size=(16, 16)).astype(np.float64) for _ in range(3)]


def test_sum_of_all(three_arrays):
    data = _make_mock_data(three_arrays)
    result = sum_of_all(data)
    expected = three_arrays[0] + three_arrays[1] + three_arrays[2]
    npt.assert_allclose(result, expected, atol=1e-6)
    assert result.spacing == [1.0, 1.0]


def test_average_of_all(three_arrays):
    data = _make_mock_data(three_arrays)
    result = average_of_all(data)
    expected = (three_arrays[0] + three_arrays[1] + three_arrays[2]) / 3
    npt.assert_allclose(result, expected, atol=1e-6)


def test_simple_fusion(three_arrays):
    data = _make_mock_data(three_arrays)
    result = simple_fusion(data)
    assert result.shape == (16, 16)
    assert result.min() >= 0


def test_simple_fusion_identical():
    arr = np.ones((8, 8), dtype=np.float64)
    data = _make_mock_data([arr, arr, arr])
    result = simple_fusion(data)
    npt.assert_allclose(result, arr)


def test_sum_of_all_type_error():
    with pytest.raises(TypeError, match="Expected ImageData"):
        sum_of_all("not_image_data")
