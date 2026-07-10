from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.adapters.image_data import ArrayDataSource, ImageDataSource
from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType
from miplib.processing.deconvolution.blocks import BlockSpec

# ---------------------------------------------------------------------------
# ArrayDataSource
# ---------------------------------------------------------------------------


def _make_images(n_views, shape=(32, 32)):
    return [
        Image(
            (np.ones(shape, dtype=np.float64) * (i + 1)),
            spacing=(0.1, 0.1),
        )
        for i in range(n_views)
    ]


def test_array_source_single_view():
    images = _make_images(1)
    source = ArrayDataSource(images)
    assert source.n_views == 1
    assert source.shape == (32, 32)
    assert source.spacing == (0.1, 0.1)


def test_array_source_multi_view():
    images = _make_images(3)
    source = ArrayDataSource(images)
    assert source.n_views == 3
    # Each view returns its own image
    for v in range(3):
        full = source.get_full_image(v)
        expected_value = float(v + 1)
        assert full.shape == (32, 32)
        npt.assert_allclose(full, np.full((32, 32), expected_value))


def test_array_source_empty_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        ArrayDataSource([])


def test_array_source_get_block_interior():
    images = _make_images(1)
    source = ArrayDataSource(images)
    block = BlockSpec(start=np.array([10, 14]), size=np.array([12, 8]), pad=2)

    result = source.get_image_block(0, block)
    assert result.shape == (12, 8)

    p = block.pad
    inner = tuple(slice(p, p + s) for s in block.inner_size)
    npt.assert_array_equal(result[inner], images[0][block.inner_slice])


def test_array_source_get_block_at_boundary():
    images = _make_images(1)
    source = ArrayDataSource(images)
    block = BlockSpec(start=np.array([-4, 28]), size=np.array([10, 10]), pad=4)

    result = source.get_image_block(0, block)

    assert result.shape == (10, 10)
    npt.assert_array_equal(result[:4, :], 0)
    npt.assert_array_equal(result[:, 6:], 0)
    npt.assert_array_equal(result[4:10, :4], images[0][:6, 28:32])


def test_array_source_multi_view_distinct():
    images = [
        Image(np.full((32, 32), 1.0, dtype=np.float64), spacing=(0.1, 0.1)),
        Image(np.full((32, 32), 2.0, dtype=np.float64), spacing=(0.1, 0.1)),
        Image(np.full((32, 32), 3.0, dtype=np.float64), spacing=(0.1, 0.1)),
    ]
    source = ArrayDataSource(images)

    block = BlockSpec(start=np.array([8, 8]), size=np.array([16, 16]), pad=0)
    r0 = source.get_image_block(0, block)
    r1 = source.get_image_block(1, block)
    r2 = source.get_image_block(2, block)

    npt.assert_allclose(r0, 1.0)
    npt.assert_allclose(r1, 2.0)
    npt.assert_allclose(r2, 3.0)


def test_array_source_get_full_image():
    images = _make_images(2)
    source = ArrayDataSource(images)

    for v in range(2):
        full = source.get_full_image(v)
        npt.assert_array_equal(full, images[v])


# ---------------------------------------------------------------------------
# ImageDataSource -- mock delegation
# ---------------------------------------------------------------------------


def _make_mock_imagedata(n_views=3):
    data = MagicMock(spec=ImageData)
    data.get_image_size.return_value = (128, 128)
    data.get_voxel_size.return_value = (0.1, 0.1)
    data.get_registered_block.return_value = np.ones((16, 16), dtype=np.float64)
    return data


def test_image_source_properties():
    data = _make_mock_imagedata()
    source = ImageDataSource(data, [0, 1, 2], ImageType.REGISTERED, 0, 100)

    assert source.n_views == 3
    assert source.shape == (128, 128)
    assert source.spacing == (0.1, 0.1)


def test_image_source_empty_views_raises():
    data = _make_mock_imagedata()
    with pytest.raises(ValueError, match="must not be empty"):
        ImageDataSource(data, [])


def test_image_source_delegates_get_registered_block():
    data = _make_mock_imagedata(n_views=3)
    source = ImageDataSource(data, [0, 2, 4], ImageType.REGISTERED, 0, 50)

    block = BlockSpec(start=np.array([30, 50]), size=np.array([40, 32]), pad=4)
    source.get_image_block(1, block)

    assert data.get_registered_block.call_count == 1
    args, kwargs = data.get_registered_block.call_args
    expected_key = ImageKey(ImageType.REGISTERED, 2, 0, 50)
    assert args[0] == expected_key
    npt.assert_array_equal(args[1], block.inner_size)
    assert args[2] == block.pad
    npt.assert_array_equal(args[3], block.inner_start)


# ---------------------------------------------------------------------------
# ImageDataSource -- integration with real HDF5
# ---------------------------------------------------------------------------


@pytest.fixture
def hdf5_imagedata(tmp_path):
    path = str(tmp_path / "test.hdf5")
    data = ImageData(path)

    rng = np.random.default_rng(99)
    for i in range(3):
        arr = rng.normal(size=(64, 64)).astype(np.float32)
        data.add_registered_image(
            arr, scale=100, index=i, channel=0, angle=0.0, spacing=[0.1, 0.1]
        )

    yield data
    data.close()


def test_integration_roundtrip(hdf5_imagedata):
    source = ImageDataSource(hdf5_imagedata, [0])

    block = BlockSpec(start=np.array([0, 0]), size=np.array((64, 64)), pad=0)
    result = source.get_image_block(0, block)

    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    expected = hdf5_imagedata.get_image_data(key)
    npt.assert_array_equal(result, expected)


def test_integration_inner_block(hdf5_imagedata):
    source = ImageDataSource(hdf5_imagedata, [0])

    block = BlockSpec(start=np.array([8, 16]), size=np.array([32, 16]), pad=0)
    result = source.get_image_block(0, block)

    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    full = hdf5_imagedata.get_image_data(key)
    npt.assert_array_equal(result, full[8:40, 16:32])


def test_integration_boundary_block(hdf5_imagedata):
    source = ImageDataSource(hdf5_imagedata, [0])

    block = BlockSpec(start=np.array([56, -4]), size=np.array([16, 16]), pad=4)

    result = source.get_image_block(0, block)

    assert result.shape == (16, 16)
    npt.assert_array_equal(result[:, :4], 0)
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    full = hdf5_imagedata.get_image_data(key)
    npt.assert_array_equal(result[:8, 4:12], full[56:64, 0:8])
    npt.assert_array_equal(result[8:, :], 0)


def test_integration_multi_view(hdf5_imagedata):
    source = ImageDataSource(hdf5_imagedata, [0, 1, 2])

    block = BlockSpec(start=np.array((0, 0)), size=np.array((64, 64)), pad=0)
    r0 = source.get_image_block(0, block)
    r1 = source.get_image_block(1, block)

    assert not np.allclose(r0, r1)
