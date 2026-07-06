import numpy as np
import numpy.testing as npt

from miplib.data.containers.image import Image


def test_construction_2d_shape_dtype(image_2d):
    assert image_2d.shape == (16, 16)
    assert image_2d.dtype == np.float64
    assert image_2d.ndim == 2


def test_construction_3d_shape(image_3d):
    assert image_3d.shape == (8, 16, 16)
    assert image_3d.ndim == 3


def test_image_is_ndarray_subclass(image_2d):
    assert isinstance(image_2d, np.ndarray)


def test_spacing_stored_as_list(image_2d):
    assert isinstance(image_2d.spacing, list)
    assert image_2d.spacing == [0.1, 0.1]


def test_spacing_values_3d(image_3d):
    assert image_3d.spacing == [0.2, 0.1, 0.1]


def test_spacing_preserved_on_slice(camera_image):
    s = camera_image[0:10, 10:20]
    assert s.spacing == camera_image.spacing


def test_spacing_preserved_on_ufunc(gaussian_2d):
    result = gaussian_2d + 5.0
    assert result.spacing == gaussian_2d.spacing


def test_spacing_preserved_on_astype(gaussian_2d):
    result = gaussian_2d.astype(np.float32)
    assert result.spacing == gaussian_2d.spacing


def test_spacing_preserved_after_abs(shepp_logan):
    result = np.abs(shepp_logan)
    assert result.spacing == shepp_logan.spacing


def test_filename_none_by_default(image_2d):
    assert image_2d.filename is None


def test_filename_preserved_through_slice():
    img = Image(np.ones((8, 8)), spacing=(1.0, 1.0), filename="test.tif")
    s = img[0:4, 0:4]
    assert s.filename == "test.tif"


def test_filename_none_on_constructed_without():
    img = Image(np.ones((8, 8)), spacing=(1.0, 1.0))
    assert img.filename is None


def test_camera_shape(camera_image):
    """Verify the skimage camera fixture is accessible through Image."""
    assert camera_image.shape == (512, 512)
    assert camera_image.spacing == [1.0, 1.0]


def test_shepp_logan_dtype(shepp_logan):
    """Shepp-Logan phantom is float64."""
    assert shepp_logan.dtype == np.float64


def test_blobs_3d_shape(blobs_3d):
    """3D binary blobs are 64x64x64."""
    assert blobs_3d.shape == (64, 64, 64)
    assert blobs_3d.ndim == 3


def test_gaussian_2d_values(gaussian_2d):
    """Gaussian spot: center > edges, peaks at 1.0."""
    assert gaussian_2d.shape == (32, 32)
    center = gaussian_2d[16, 16]
    corner = gaussian_2d[0, 0]
    assert center > corner
    npt.assert_approx_equal(center, 1.0)


def test_view_shares_memory(image_2d):
    """Image.view() shares memory with the original underlying array."""
    view = image_2d.view(np.ndarray)
    view[0, 0] = 99.0
    assert image_2d[0, 0] == 99.0
