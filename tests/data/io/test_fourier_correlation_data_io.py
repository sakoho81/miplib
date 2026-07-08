import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)
from miplib.data.containers.image import Image
from miplib.data.io.fourier_correlation_data_reader import (
    FourierCorrelationDataReader,
)
from miplib.data.io.fourier_correlation_data_writer import (
    FourierCorrelationDataWriter,
)


@pytest.fixture
def sample_collection():
    """Build a minimal FourierCorrelationDataCollection for round-trip tests."""
    d1 = FourierCorrelationData()
    d1.resolution["threshold"] = np.array([0.1, 0.2, 0.3])
    d1.resolution["resolution"] = 123.4
    d1.resolution["resolution-point"] = (64, 64)
    d1.resolution["criterion"] = "one-bit"
    d1.resolution["resolution-threshold-coefficients"] = np.array([1.0, 2.0])
    d1.correlation["correlation"] = np.array([0.9, 0.8, 0.5])
    d1.correlation["frequency"] = np.array([0.0, 0.1, 0.2])
    d1.correlation["points-x-bin"] = np.array([100, 80, 50])
    d1.correlation["curve-fit"] = np.array([0.95, 0.82, 0.48])
    d1.correlation["curve-fit-coefficients"] = np.array([1.0, -2.0, 0.5])

    coll = FourierCorrelationDataCollection()
    coll[0] = d1
    return coll


def test_metadata_roundtrip(tmp_path):
    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_metadata({"author": "test", "version": 2})
    writer.close()

    reader = FourierCorrelationDataReader(str(tmp_path / "test.hdf5"))
    meta = reader.read_metadata()
    reader.close()
    assert meta["author"] == "test"
    assert meta["version"] == 2


def test_metadata_rejects_non_dict(tmp_path):
    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    with pytest.raises(TypeError, match="Expected dict"):
        writer.write_metadata("not a dict")  # type: ignore[arg-type]
    writer.close()


def test_images_roundtrip(tmp_path):
    img = Image(np.ones((8, 8), dtype=np.float32), spacing=(0.5, 0.5))

    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_images(img)
    writer.close()

    reader = FourierCorrelationDataReader(str(tmp_path / "test.hdf5"))
    images = reader.read_images()
    reader.close()
    assert len(images) == 1
    result = images[0]
    assert isinstance(result, Image)
    npt.assert_array_equal(result, img)
    assert result.spacing == [0.5, 0.5]


def test_images_spacing_preserves_floats(tmp_path):
    img = Image(np.ones((4, 4)), spacing=(0.123, 0.456))

    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_images(img)
    writer.close()

    reader = FourierCorrelationDataReader(str(tmp_path / "test.hdf5"))
    result = reader.read_images()[0]
    reader.close()
    npt.assert_array_almost_equal(result.spacing, [0.123, 0.456])


def test_images_single_index_read(tmp_path):
    img0 = Image(np.ones((4, 4)), spacing=(1.0, 1.0))
    img1 = Image(np.zeros((4, 4)), spacing=(1.0, 1.0))

    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_images((img0, img1))
    writer.close()

    reader = FourierCorrelationDataReader(str(tmp_path / "test.hdf5"))
    result = reader.read_images(index=1)
    reader.close()
    npt.assert_array_equal(result, img1)


def test_data_set_roundtrip(tmp_path, sample_collection):
    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_data_set(sample_collection)
    writer.close()

    reader = FourierCorrelationDataReader(str(tmp_path / "test.hdf5"))
    result = reader.read_data_set()
    reader.close()

    assert len(result) == 1
    original = sample_collection[0]
    recovered = result[0]

    npt.assert_array_equal(
        recovered.resolution["threshold"], original.resolution["threshold"]
    )
    assert recovered.resolution["resolution"] == original.resolution["resolution"]
    assert recovered.resolution["criterion"] == original.resolution["criterion"]
    npt.assert_array_equal(
        recovered.correlation["correlation"], original.correlation["correlation"]
    )
    npt.assert_array_equal(
        recovered.correlation["points-x-bin"], original.correlation["points-x-bin"]
    )


def test_data_set_duplicate_raises(tmp_path, sample_collection):
    writer = FourierCorrelationDataWriter(str(tmp_path), "test.hdf5")
    writer.write_data_set(sample_collection)
    with pytest.raises(ValueError, match="already exists"):
        writer.write_data_set(sample_collection)
    writer.close()


def test_reader_rejects_missing_file():
    with pytest.raises(ValueError, match="Not a valid filename"):
        FourierCorrelationDataReader("/nonexistent/path.hdf5")


def test_writer_rejects_bad_extension(tmp_path):
    with pytest.raises(ValueError, match=".hdf5"):
        FourierCorrelationDataWriter(str(tmp_path), "test.txt")
