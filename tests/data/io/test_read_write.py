import os

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.data.io import read, write


def _has_simpleitk():
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        return False
    return True


def _has_bioformats():
    try:
        import pims  # noqa: F401

        return pims.bioformats.available()
    except Exception:
        return False


# -- TIFF write + read round-trip ----------------------------------------------


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_tiff_roundtrip_2d(tmp_path):
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    img = Image(data, spacing=(0.1, 0.2))
    path = str(tmp_path / "test.tif")

    write.image(path, img)
    assert os.path.isfile(path)

    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_equal(arr, data)
    npt.assert_almost_equal(result.spacing[0], 0.1)
    npt.assert_almost_equal(result.spacing[1], 0.2)


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_tiff_roundtrip_3d(tmp_path):
    data = np.ones((4, 8, 8), dtype=np.float32)
    img = Image(data, spacing=(0.5, 0.1, 0.1))
    path = str(tmp_path / "test.tif")

    write.image(path, img)
    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_equal(arr, data)
    npt.assert_almost_equal(result.spacing[0], 0.5)
    npt.assert_almost_equal(result.spacing[1], 0.1)
    npt.assert_almost_equal(result.spacing[2], 0.1)


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_tiff_roundtrip_preserves_shape(tmp_path):
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    img = Image(data, spacing=(1.0, 1.0))
    path = str(tmp_path / "test.tif")

    write.image(path, img)
    result = read.get_image(path)
    arr = np.asarray(result)
    assert arr.shape == (10, 10)
    npt.assert_almost_equal(result.spacing[0], 1.0)
    npt.assert_almost_equal(result.spacing[1], 1.0)


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_tiff_roundtrip_with_non_unit_spacing(tmp_path):
    data = np.ones((6, 6), dtype=np.float32)
    img = Image(data, spacing=(3.0, 5.0))
    path = str(tmp_path / "test.tif")

    write.image(path, img)
    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_equal(arr, data)
    npt.assert_almost_equal(result.spacing[0], 3.0)
    npt.assert_almost_equal(result.spacing[1], 5.0)


# -- ITK write + read round-trip -----------------------------------------------


@pytest.mark.skipif(not _has_simpleitk(), reason="SimpleITK not available")
def test_itk_roundtrip_2d(tmp_path):
    data = np.eye(4, dtype=np.float32)
    img = Image(data, spacing=(0.3, 0.3))
    path = str(tmp_path / "test.mha")

    write.image(path, img)
    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_almost_equal(arr, data)
    npt.assert_almost_equal(result.spacing[0], 0.3)


@pytest.mark.skipif(not _has_simpleitk(), reason="SimpleITK not available")
def test_itk_roundtrip_3d(tmp_path):
    data = np.ones((3, 6, 6), dtype=np.float32)
    img = Image(data, spacing=(0.5, 0.2, 0.2))
    path = str(tmp_path / "test.mha")

    write.image(path, img)
    result = read.get_image(path)
    assert np.asarray(result).shape == (3, 6, 6)


# -- Edge cases ----------------------------------------------------------------


def test_write_rejects_non_image(tmp_path):
    with pytest.raises((AttributeError, TypeError)):
        write.image(str(tmp_path / "test.tif"), np.ones((4, 4)))  # type: ignore[arg-type]


def test_write_unknown_extension(tmp_path):
    img = Image(np.ones((4, 4)), spacing=(1.0, 1.0))
    with pytest.raises(ValueError, match="Unsupported file extension"):
        write.image(str(tmp_path / "test.png"), img)


def test_read_invalid_return_type():
    with pytest.raises(ValueError, match="return_type"):
        read.get_image("test.tif", return_type="invalid")


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_read_return_type_itk(tmp_path):
    import SimpleITK as sitk

    img = Image(np.ones((4, 4), dtype=np.float32), spacing=(0.5, 0.5))
    path = str(tmp_path / "test.tif")
    write.image(path, img)

    result = read.get_image(path, return_type="itk")
    assert isinstance(result, sitk.Image)


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_read_return_type_image(tmp_path):
    img = Image(np.ones((4, 4), dtype=np.float32), spacing=(0.5, 0.5))
    path = str(tmp_path / "test.tif")
    write.image(path, img)

    result = read.get_image(path, return_type="image")
    assert isinstance(result, Image)
    npt.assert_array_equal(np.asarray(result), np.ones((4, 4), dtype=np.float32))


# -- Bioformats ----------------------------------------------------------------


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_bioformats_reads_tiff_2d(tmp_path):
    data = np.arange(25, dtype=np.float32).reshape(5, 5)
    img = Image(data, spacing=(0.1, 0.2))
    path = str(tmp_path / "test.tif")
    write.image(path, img)

    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_equal(arr, data)


@pytest.mark.skipif(not _has_bioformats(), reason="Bioformats not available")
def test_bioformats_reads_tiff_3d(tmp_path):
    data = np.ones((3, 6, 6), dtype=np.float32)
    img = Image(data, spacing=(0.5, 0.1, 0.1))
    path = str(tmp_path / "test.tif")
    write.image(path, img)

    result = read.get_image(path)
    arr = np.asarray(result)
    npt.assert_array_equal(arr, data)
