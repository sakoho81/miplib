import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType
from miplib.data.containers.image_data_store import HDF5ImageStore

# --- ImageType ----------------------------------------------------------------


def test_image_type_is_strenum():
    assert isinstance(ImageType.ORIGINAL, str)
    assert ImageType.ORIGINAL == "original"
    assert ImageType("registered") == ImageType.REGISTERED
    assert list(ImageType) == [
        ImageType.ORIGINAL,
        ImageType.REGISTERED,
        ImageType.FUSED,
        ImageType.PSF,
    ]


# --- ImageKey -----------------------------------------------------------------


def test_image_key_to_path_original():
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert key.to_path() == "original/0/channel_0_scale_100"


def test_image_key_to_path_fused():
    key = ImageKey(ImageType.FUSED, 0, 0, 100)
    assert key.to_path() == "fused/channel_0_scale_100"


def test_image_key_to_path_multichannel():
    key = ImageKey(ImageType.REGISTERED, 2, 1, 50)
    assert key.to_path() == "registered/2/channel_1_scale_50"


def test_image_key_frozen():
    with pytest.raises(Exception):
        key = ImageKey(ImageType.ORIGINAL, 0)
        key.index = 1  # type: ignore[misc]


def test_image_key_defaults():
    key = ImageKey(ImageType.PSF, index=0)
    assert key.channel == 0
    assert key.scale == 100


# --- HDF5ImageStore -----------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    path = str(tmp_path / "test.hdf5")
    s = HDF5ImageStore(path)
    yield s
    s.close()


def test_store_initial_state(store):
    assert store.series_count == 0
    assert store.channel_count == 1


def test_add_and_get_image_data(store):
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    store.add_image(key, data, angle=45.0, spacing=[0.1, 0.1, 0.1])

    assert store.check_if_exists(key)
    npt.assert_array_equal(store.get_image_data(key), data)


def test_add_image_stores_attributes(store):
    data = np.ones((4, 4), dtype=np.float64)
    key = ImageKey(ImageType.REGISTERED, 1, 0, 50)
    store.add_image(key, data, angle=90.0, spacing=[0.2, 0.2])

    attrs = store.get_image_attributes(key)
    assert attrs["angle"] == 90.0
    assert attrs["spacing"] == [0.2, 0.2]
    assert list(attrs["size"]) == [4, 4]


def test_add_psf_with_calculated_flag(store):
    data = np.ones((2, 2), dtype=np.float32)
    key = ImageKey(ImageType.PSF, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0], calculated=True)

    attrs = store.get_image_attributes(key)
    assert attrs["calculated"] is True


def test_add_image_increments_series_count(store):
    assert store.series_count == 0
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((2, 2)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    assert store.series_count == 1


def test_add_image_duplicate_index_no_increment(store):
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((2, 2)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 50),
        np.ones((1, 1)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    assert store.series_count == 1


def test_add_image_no_double_add(store):
    data = np.ones((2, 2))
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0])

    data2 = np.zeros((2, 2))
    store.add_image(key, data2, angle=1.0, spacing=[2.0, 2.0])

    npt.assert_array_equal(store.get_image_data(key), data)
    assert store.get_image_attributes(key)["angle"] == 0.0


def test_add_image_rejects_non_ndarray(store):
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        store.add_image(
            ImageKey(ImageType.ORIGINAL, 0, 0, 100),
            [1, 2, 3],  # type: ignore[arg-type]
            angle=0.0,
            spacing=[1.0],
        )


def test_add_fused_image(store):
    data = np.ones((5, 5), dtype=np.float32)
    store.add_fused_image(channel=0, scale=100, data=data, spacing=[0.5, 0.5])

    key = ImageKey(ImageType.FUSED, 0, 0, 100)
    assert store.check_if_exists(key)
    npt.assert_array_equal(store.get_image_data(key), data)


def test_add_fused_image_no_duplicate(store):
    data = np.ones((2, 2))
    store.add_fused_image(channel=0, scale=100, data=data, spacing=[1.0, 1.0])
    store.add_fused_image(
        channel=0, scale=100, data=np.zeros((2, 2)), spacing=[2.0, 2.0]
    )
    assert store.get_image_attributes(ImageKey(ImageType.FUSED, 0, 0, 100))[
        "spacing"
    ] == [1.0, 1.0]


def test_add_transform(store):
    data = np.ones((4, 4))
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0])
    store.add_transform(
        index=0,
        channel=0,
        scale=100,
        params=[1.0, 2.0, 3.0],
        fixed_params=[0.0],
        transform_type=10,
    )

    attrs = store.get_image_attributes(key)
    assert attrs["tfm_type"] == 10
    assert attrs["tfm_params"] == [1.0, 2.0, 3.0]
    assert attrs["tfm_fixed_params"] == [0.0]


def test_add_transform_missing_dataset_raises(store):
    with pytest.raises(ValueError, match="does not exist"):
        store.add_transform(
            index=0,
            channel=0,
            scale=100,
            params=[1.0],
            fixed_params=[0.0],
            transform_type=10,
        )


def test_get_number_of_images(store):
    assert store.get_number_of_images(ImageType.ORIGINAL) == 0

    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((2, 2)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 1, 0, 100),
        np.ones((2, 2)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )

    assert store.get_number_of_images(ImageType.ORIGINAL) == 2
    assert store.get_number_of_images(ImageType.REGISTERED) == 0


def test_get_scales_consistent(store):
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((3, 3)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 50),
        np.ones((1, 1)),
        angle=0.0,
        spacing=[2.0, 2.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 1, 0, 100),
        np.ones((3, 3)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 1, 0, 50),
        np.ones((1, 1)),
        angle=0.0,
        spacing=[2.0, 2.0],
    )

    scales = store.get_scales(ImageType.ORIGINAL)
    assert set(scales) == {50, 100}


def test_get_scales_inconsistent_raises(store):
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((3, 3)),
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    store.add_image(
        ImageKey(ImageType.ORIGINAL, 1, 0, 50),
        np.ones((1, 1)),
        angle=0.0,
        spacing=[2.0, 2.0],
    )

    with pytest.raises(ValueError, match="inconsisten"):
        store.get_scales(ImageType.ORIGINAL)


def test_check_if_exists(store):
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert not store.check_if_exists(key)

    store.add_image(key, np.ones((2, 2)), angle=0.0, spacing=[1.0, 1.0])
    assert store.check_if_exists(key)


def test_delete_dataset(store):
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    store.add_image(key, np.ones((2, 2)), angle=0.0, spacing=[1.0, 1.0])
    assert store.check_if_exists(key)

    store.delete_dataset(key)
    assert not store.check_if_exists(key)


def test_delete_nonexistent_is_silent(store):
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    store.delete_dataset(key)  # no error


def test_registered_block_full_read(store):
    data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0, 1.0])

    block = store.get_registered_block(
        key,
        block_size=np.array([4, 4, 4]),
        block_pad=0,
        block_start_index=np.array([0, 0, 0]),
    )
    npt.assert_array_equal(block, data)


def test_registered_block_with_padding(store):
    data = np.ones((4, 4), dtype=np.float32)
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0])

    block = store.get_registered_block(
        key,
        block_size=np.array([2, 2]),
        block_pad=1,
        block_start_index=np.array([0, 0]),
    )
    assert block.shape == (4, 4)


def test_registered_block_boundary_handling(store):
    data = np.ones((3, 3), dtype=np.float32)
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0])

    block = store.get_registered_block(
        key,
        block_size=np.array([2, 2]),
        block_pad=1,
        block_start_index=np.array([2, 2]),
    )
    assert block.shape == (4, 4)


def test_registered_block_non_registered_raises(store):
    data = np.ones((2, 2))
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    store.add_image(key, data, angle=0.0, spacing=[1.0, 1.0])

    with pytest.raises(ValueError, match="REGISTERED"):
        store.get_registered_block(
            key,
            block_size=np.array([2, 2]),
            block_pad=0,
            block_start_index=np.array([0, 0]),
        )


def test_context_manager(tmp_path):
    path = str(tmp_path / "ctx.hdf5")
    with HDF5ImageStore(path) as s:
        s.add_image(
            ImageKey(ImageType.ORIGINAL, 0, 0, 100),
            np.ones((2, 2)),
            angle=0.0,
            spacing=[1.0, 1.0],
        )
    # File closed; reopening should work
    s2 = HDF5ImageStore(path)
    assert s2.series_count == 1
    s2.close()


def test_reopen_preserves_data(tmp_path):
    path = str(tmp_path / "reopen.hdf5")
    s1 = HDF5ImageStore(path)
    s1.add_image(
        ImageKey(ImageType.ORIGINAL, 0, 0, 100),
        np.ones((3, 3)),
        angle=30.0,
        spacing=[0.5, 0.5],
    )
    s1.close()

    s2 = HDF5ImageStore(path)
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert s2.check_if_exists(key)
    assert s2.get_image_attributes(key)["angle"] == 30.0
    s2.close()


# --- ImageData facade ---------------------------------------------------------


@pytest.fixture
def imagedata(tmp_path):
    path = str(tmp_path / "imdata.hdf5")
    data = ImageData(path)
    yield data
    data.close()


def test_facade_add_original(imagedata):
    imagedata.add_original_image(
        np.ones((3, 3), dtype=np.float32),
        scale=100,
        index=0,
        channel=0,
        angle=30.0,
        spacing=[1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert imagedata.check_if_exists(ImageType.ORIGINAL, 0, 0, 100)
    m = imagedata.get_max(key)
    assert m == 1.0


def test_facade_add_original_anisotropic_resample(imagedata):
    data = np.ones((10, 16, 16), dtype=np.float32)
    imagedata.add_original_image(
        data,
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[2.0, 1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    img = imagedata.get_image(key)
    assert img.shape[0] == 20  # z-zoom: 2.0 / 1.0 = 2x


def test_facade_add_registered_overwrite(imagedata):
    key = ImageKey(ImageType.REGISTERED, 0, 0, 100)
    imagedata.add_registered_image(
        np.ones((3, 3)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    imagedata.add_registered_image(
        np.zeros((3, 3)),
        scale=100,
        index=0,
        channel=0,
        angle=90.0,
        spacing=[1.0, 1.0],
        overwrite=False,
    )
    assert imagedata.get_max(key) == 1.0

    imagedata.add_registered_image(
        np.zeros((3, 3)),
        scale=100,
        index=0,
        channel=0,
        angle=90.0,
        spacing=[1.0, 1.0],
        overwrite=True,
    )
    assert imagedata.get_max(key) == 0.0


def test_facade_get_voxel_size(imagedata):
    imagedata.add_original_image(
        np.ones((2, 3, 3)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[2.0, 1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    vs = imagedata.get_voxel_size(key)
    assert vs == [1.0, 1.0, 1.0]  # after isotropic resampling


def test_facade_get_rotation_angle(imagedata):
    imagedata.add_original_image(
        np.ones((3, 3)),
        scale=100,
        index=0,
        channel=0,
        angle=90.0,
        spacing=[1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    rads = imagedata.get_rotation_angle(key, radians=True)
    assert np.isclose(rads, np.pi / 2.0)
    degs = imagedata.get_rotation_angle(key, radians=False)
    assert degs == 90.0


def test_facade_get_image(imagedata):
    data = np.ones((4, 4), dtype=np.float32)
    imagedata.add_original_image(
        data,
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[0.5, 0.5],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    img = imagedata.get_image(key)
    assert isinstance(img, Image)
    assert img.spacing == [0.5, 0.5]


def test_facade_get_image_size(imagedata):
    imagedata.add_original_image(
        np.ones((4, 8, 16)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert imagedata.get_image_size(key) == (4, 8, 16)


def test_facade_get_dtype(imagedata):
    imagedata.add_original_image(
        np.ones((2, 2), dtype=np.int32),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    key = ImageKey(ImageType.ORIGINAL, 0, 0, 100)
    assert imagedata.get_dtype(key) == np.int32


def test_facade_get_number_of_images(imagedata):
    imagedata.add_original_image(
        np.ones((2, 2)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    imagedata.add_original_image(
        np.ones((2, 2)),
        scale=100,
        index=1,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    assert imagedata.get_number_of_images(ImageType.ORIGINAL) == 2


def test_facade_get_scales(imagedata):
    imagedata.add_original_image(
        np.ones((4, 4)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    imagedata.add_original_image(
        np.ones((2, 2)),
        scale=50,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[2.0, 2.0],
    )
    scales = imagedata.get_scales(ImageType.ORIGINAL)
    assert set(scales) == {50, 100}


def test_facade_add_fused(imagedata):
    data = np.ones((5, 5))
    imagedata.add_fused_image(data, channel=0, scale=100, spacing=[0.3, 0.3])
    key = ImageKey(ImageType.FUSED, 0, 0, 100)
    npt.assert_array_equal(imagedata.get_image_data(key), data)


def test_facade_series_and_channel_count(imagedata):
    assert imagedata.channel_count == 1
    assert imagedata.series_count == 0
    imagedata.add_original_image(
        np.ones((2, 2)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    assert imagedata.series_count == 1


def test_facade_add_original_rejects_non_ndarray(imagedata):
    with pytest.raises(TypeError, match="Expected numpy.ndarray"):
        imagedata.add_original_image(
            [1, 2, 3],  # type: ignore[arg-type]
            scale=100,
            index=0,
            channel=0,
            angle=0.0,
            spacing=[1.0],
        )


def test_facade_can_accept_string_image_type(imagedata):
    imagedata.add_original_image(
        np.ones((2, 2)),
        scale=100,
        index=0,
        channel=0,
        angle=0.0,
        spacing=[1.0, 1.0],
    )
    assert imagedata.get_number_of_images("original") == 1
    assert imagedata.get_scales("original") == [100]
