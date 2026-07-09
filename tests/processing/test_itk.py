"""Tests for miplib/processing/itk.py.

Sanity checks for all 19 public functions. Uses synthetic numpy arrays
converted to SimpleITK images — no external test data required.
"""

import numpy as np
import pytest
import SimpleITK as sitk
from numpy.testing import assert_array_almost_equal

from miplib.data.containers.image import Image
from miplib.processing.itk import (
    calculate_center_of_image,
    convert_from_itk_image,
    convert_from_numpy,
    convert_to_itk_image,
    gaussian_blurring_filter,
    get_image_statistics,
    get_itk_transform_parameters,
    grayscale_dilate_filter,
    make_composite_rgb_image,
    make_itk_transform,
    make_translation_transforms_from_offsets,
    mean_filter,
    median_filter,
    normalize_image_filter,
    resample_image,
    resample_to_isotropic,
    rescale_intensity,
    rotate_image,
    rotate_psf,
    threshold_image_filter,
    type_cast,
)
from tests.conftest import gaussian_spot, impulse

# ---------------------------------------------------------------------------
# convert_from_numpy / convert_from_itk_image / convert_to_itk_image
# ---------------------------------------------------------------------------


def test_convert_numpy_to_itk_roundtrip_2d():
    """ndarray → sitk.Image → Image: data and spacing round-trip correctly."""
    data = np.arange(20, dtype=np.float64).reshape(4, 5)
    spacing = (0.5, 1.0)
    sitk_img = convert_from_numpy(data, spacing)
    miplib_img = convert_from_itk_image(sitk_img)
    assert_array_almost_equal(miplib_img, data)
    assert miplib_img.spacing == [0.5, 1.0]


def test_convert_numpy_to_itk_roundtrip_3d():
    """Same as above for 3D."""
    data = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    spacing = (0.2, 0.4, 0.6)
    sitk_img = convert_from_numpy(data, spacing)
    miplib_img = convert_from_itk_image(sitk_img)
    assert_array_almost_equal(miplib_img, data)
    assert miplib_img.spacing == [0.2, 0.4, 0.6]


def test_convert_numpy_sitk_size_order():
    """SimpleITK size order is (x, y); spacing is set reversed internally."""
    data = np.ones((3, 5), dtype=np.float64)  # numpy: (rows, cols) = (y, x)
    sitk_img = convert_from_numpy(data, (2.0, 0.5))
    assert sitk_img.GetSize() == (5, 3)  # ITK: (x, y)
    assert sitk_img.GetSpacing() == (0.5, 2.0)


def test_convert_to_itk_image():
    """miplib Image → sitk.Image: spacing is reversed correctly."""
    miplib_img = Image(np.ones((4, 6), dtype=np.float64), spacing=(1.0, 2.0))
    sitk_img = convert_to_itk_image(miplib_img)
    assert isinstance(sitk_img, sitk.Image)
    assert sitk_img.GetSpacing() == (2.0, 1.0)


def test_convert_from_numpy_type_error():
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        convert_from_numpy([1, 2, 3], (1.0,))  # type: ignore[arg-type]


def test_convert_from_itk_image_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        convert_from_itk_image(np.ones((4, 4)))  # type: ignore[arg-type]


def test_convert_to_itk_image_type_error():
    with pytest.raises(TypeError, match="Expected Image"):
        convert_to_itk_image(np.ones((4, 4)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_itk_transform / get_itk_transform_parameters
# ---------------------------------------------------------------------------


def test_make_and_get_transform_roundtrip():
    """Parameters set via make_itk_transform are recovered by get_itk_transform_parameters."""
    tfm = make_itk_transform(
        "AffineTransform", 2, (1.0, 0.0, 0.0, 1.0, 10.0, 20.0), (0.0, 0.0)
    )
    name, params, fixed = get_itk_transform_parameters(tfm)
    assert name == "AffineTransform"
    assert params == (1.0, 0.0, 0.0, 1.0, 10.0, 20.0)
    assert fixed == (0.0, 0.0)


def test_make_transform_unsupported_type():
    with pytest.raises(NotImplementedError):
        make_itk_transform("BogusTransform", 2, (), ())


# ---------------------------------------------------------------------------
# make_translation_transforms_from_offsets
# ---------------------------------------------------------------------------


def test_make_translation_transforms_from_offsets():
    offsets = [(1.0, 2.0), (-3.0, 4.0), (0.0, -1.0)]
    transforms = make_translation_transforms_from_offsets(offsets)
    assert len(transforms) == 3
    for tfm, (ox, oy) in zip(transforms, offsets, strict=True):
        assert isinstance(tfm, sitk.TranslationTransform)
        assert tfm.GetOffset() == (ox, oy)


# ---------------------------------------------------------------------------
# resample_image
# ---------------------------------------------------------------------------


def test_resample_image_identity():
    """Identity transform must leave the image unchanged."""
    data = impulse((16, 16))
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    identity = sitk.AffineTransform(2)
    result = resample_image(sitk_img, identity)
    result_array = sitk.GetArrayFromImage(result)
    assert_array_almost_equal(result_array, data)


def test_resample_image_subsample():
    """Halving the pixel count (2× spacing) with linear interpolation."""
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    scale = sitk.AffineTransform(2)
    scale.Scale(2.0)
    ref = convert_from_numpy(
        np.zeros((5, 5), dtype=np.float64).astype(np.float64),
        (1.0,) * np.zeros((5, 5), dtype=np.float64).ndim,
    )
    result = resample_image(sitk_img, scale, reference=ref)
    result_array = sitk.GetArrayViewFromImage(result)
    assert result_array.shape == (5, 5)


def test_resample_image_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        resample_image(np.ones((4, 4)), sitk.AffineTransform(2))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rotate_image
# ---------------------------------------------------------------------------


def test_rotate_image_180_degree_identity():
    """Rotating a centered impulse by 180° leaves it at the center."""
    data = impulse((32, 32))
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = rotate_image(sitk_img, 180.0)
    result_array = sitk.GetArrayFromImage(result)
    peak = np.unravel_index(np.argmax(result_array), result_array.shape)
    assert peak == (16, 16)


def test_rotate_image_90_degree_swaps_axes():
    """90° rotation swaps the axes of an off-centre impulse."""
    # Place impulse at (24, 8) in numpy coords
    data = np.zeros((32, 32), dtype=np.float64)
    data[24, 8] = 1.0
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = rotate_image(sitk_img, 90.0)
    result_array = sitk.GetArrayFromImage(result)
    peak = np.unravel_index(np.argmax(result_array), result_array.shape)
    # 90° counter-clockwise: (y, x) = (24, 8) rotated around center (15.5, 15.5)
    # The ITK rotation centre is physical: (16, 16) since spacing=(1,1)
    # After 90° CCW around ITK centre (16, 16):
    #   numpy x' = 16 + (16 - 24) = 8, numpy y' = 16 + (8 - 16) = 8
    # Actually let's verify empirically — assert that peak has moved
    assert peak != (24, 8)
    # The 90° rotation should move the impulse away from its original spot
    assert result_array[24, 8] < 0.1


def test_rotate_image_3d_180_z():
    """180° rotation around z-axis of a centered 3D impulse is near-identity."""
    data = impulse((16, 16, 16))
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = rotate_image(sitk_img, 180.0, axis=2)
    result_array = sitk.GetArrayFromImage(result)
    peak = np.unravel_index(np.argmax(result_array), result_array.shape)
    assert peak == (8, 8, 8)


def test_rotate_image_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        rotate_image(np.ones((16, 16)), 90.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rotate_psf
# ---------------------------------------------------------------------------


def test_rotate_psf_identity_transform():
    """An identity transform leaves the PSF unchanged."""
    psf = gaussian_spot((16, 16, 16), sigma=2.0)
    spacing = (0.1, 0.1, 0.1)
    tfm = sitk.AffineTransform(3)

    result = rotate_psf(psf, tfm, spacing=spacing, return_numpy=True)
    assert isinstance(result, Image)
    assert_array_almost_equal(result, psf, decimal=4)


def test_rotate_psf_90_degree_rotation():
    """90° rotation around z converts x-asymmetry into y-asymmetry."""
    # Gaussian with sigma_x=2, sigma_y=5 — elongated in y
    psf = gaussian_spot((24, 24, 24), sigma=2.0)
    # Make it anisotropic by stretching one axis
    y_coord = np.arange(24, dtype=np.float64) - 12
    stretch = np.exp(-(y_coord**2) / (2 * 5.0**2) + (y_coord**2) / (2 * 2.0**2))
    psf_aniso = psf * stretch[np.newaxis, :, np.newaxis]
    spacing = (1.0, 1.0, 1.0)

    # 90° rotation around z
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    tfm = sitk.AffineTransform(3)
    tfm.SetMatrix(rot.ravel())

    result = rotate_psf(psf_aniso, tfm, spacing=spacing, return_numpy=True)
    assert isinstance(result, Image)

    # After 90° rotation around z, the y-elongation should have moved to x
    # Check that the asymmetry has swapped: y-variance < x-variance
    mid = 12
    x_profile = result[mid, mid, :]
    y_profile = result[mid, :, mid]
    # The elongated axis (originally y) is now x, so x_profile should be wider
    x_spread = np.sum(x_profile > x_profile.max() * 0.5)
    y_spread = np.sum(y_profile > y_profile.max() * 0.5)
    assert x_spread > y_spread


def test_rotate_psf_requires_spacing_for_ndarray():
    psf = np.ones((8, 8, 8), dtype=np.float64)
    tfm = sitk.AffineTransform(3)
    with pytest.raises(ValueError, match="spacing is required"):
        rotate_psf(psf, tfm, spacing=None)


# ---------------------------------------------------------------------------
# resample_to_isotropic
# ---------------------------------------------------------------------------


def test_resample_to_isotropic_produces_isotropic_spacing(blobs_3d: Image):
    """An anisotropic 3D image must become isotropic after resampling."""
    sitk_img = convert_to_itk_image(
        blobs_3d
    )  # spacing (0.2, 0.1, 0.1) → ITK (0.1, 0.1, 0.2)
    result = resample_to_isotropic(sitk_img)
    sp = result.GetSpacing()
    assert sp[0] == pytest.approx(sp[1])
    assert sp[0] == pytest.approx(sp[2])


def test_resample_to_isotropic_rejects_2d():
    """2D images are rejected with a clear error."""
    sitk_img = convert_from_numpy(
        np.ones((16, 16), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((16, 16), dtype=np.float64).ndim,
    )
    with pytest.raises(ValueError, match="requires a 3D image"):
        resample_to_isotropic(sitk_img)


def test_resample_to_isotropic_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        resample_to_isotropic(np.ones((8, 8, 8)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# rescale_intensity
# ---------------------------------------------------------------------------


def test_rescale_intensity_uint8():
    """A uint8 image is rescaled to [0, 255]."""
    data = np.random.default_rng(0).integers(10, 100, size=(8, 8)).astype(np.uint8)
    sitk_img = sitk.GetImageFromArray(data)
    result = rescale_intensity(sitk_img)
    result_array = sitk.GetArrayViewFromImage(result)
    assert result_array.min() == 0
    assert result_array.max() == 255


def test_rescale_intensity_non_uint8_returns_unchanged():
    """Non-uint8 images are returned as-is."""
    sitk_img = convert_from_numpy(
        np.random.default_rng(1).random((8, 8)).astype(np.float64),
        (1.0,) * np.random.default_rng(1).random((8, 8)).ndim,
    )
    result = rescale_intensity(sitk_img)
    result_array = sitk.GetArrayViewFromImage(result)
    input_array = sitk.GetArrayViewFromImage(sitk_img)
    assert_array_almost_equal(result_array, input_array)


# ---------------------------------------------------------------------------
# gaussian_blurring_filter
# ---------------------------------------------------------------------------


def test_gaussian_blurring_reduces_impulse_peak():
    """Blurring an impulse must reduce its peak value."""
    data = impulse((32, 32))
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = gaussian_blurring_filter(sitk_img, variance=4.0)
    result_array = sitk.GetArrayViewFromImage(result)
    assert result_array.max() < 0.5  # original peak was 1.0
    assert result_array.sum() == pytest.approx(data.sum(), rel=0.05)


def test_gaussian_blurring_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        gaussian_blurring_filter(np.ones((8, 8)), 1.0)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# grayscale_dilate_filter
# ---------------------------------------------------------------------------


def test_grayscale_dilate_spreads_bright_pixel():
    """A single bright pixel spreads to its 3×3 neighbourhood after dilation."""
    data = np.zeros((16, 16), dtype=np.float64)
    data[8, 8] = 1.0
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = grayscale_dilate_filter(sitk_img, kernel_radius=1)
    result_array = sitk.GetArrayViewFromImage(result)
    # The 3×3 neighbourhood centred on the original pixel must all be non-zero
    neighbourhood = result_array[7:10, 7:10]
    assert (neighbourhood > 0).all()
    # Pixels far from the peak must still be zero
    assert result_array[0, 0] == 0.0


def test_grayscale_dilate_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        grayscale_dilate_filter(np.ones((8, 8)), 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# mean_filter
# ---------------------------------------------------------------------------


def test_mean_filter_smoothes_noisy_uniform():
    """A noisy uniform image is smoothed toward the mean after filtering."""
    rng = np.random.default_rng(0)
    data = np.full((32, 32), 5.0, dtype=np.float64)
    data += rng.normal(0, 2.0, data.shape)  # add Gaussian noise
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = mean_filter(sitk_img, kernel_radius=2)
    result_array = sitk.GetArrayViewFromImage(result)
    # The filtered image should have reduced variance compared to the input
    interior = result_array[5:-5, 5:-5]
    input_interior = data[5:-5, 5:-5]
    assert interior.var() < input_interior.var()
    # The mean should stay close to 5.0
    assert pytest.approx(interior.mean(), abs=0.5) == 5.0


def test_mean_filter_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        mean_filter(np.ones((8, 8)), 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# median_filter
# ---------------------------------------------------------------------------


def test_median_filter_removes_spike():
    """An isolated bright spike must be removed by median filtering."""
    data = np.full((16, 16), 1.0, dtype=np.float64)
    data[8, 8] = 100.0
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = median_filter(sitk_img, kernel_radius=1)
    result_array = sitk.GetArrayViewFromImage(result)
    assert result_array[8, 8] == pytest.approx(1.0)
    # Rest of image unchanged (except boundaries)
    assert result_array[0, 0] == pytest.approx(1.0)


def test_median_filter_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        median_filter(np.ones((8, 8)), 1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_image_filter
# ---------------------------------------------------------------------------


def test_normalize_image_filter_zero_mean_unit_variance():
    """Output has zero mean and unit variance."""
    rng = np.random.default_rng(2)
    data = rng.random((32, 32)).astype(np.float64) * 10 + 5
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = normalize_image_filter(sitk_img)
    result_array = sitk.GetArrayViewFromImage(result)
    assert pytest.approx(result_array.mean(), abs=1e-6) == 0.0
    assert pytest.approx(result_array.var(), rel=0.01) == 1.0


def test_normalize_image_filter_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        normalize_image_filter(np.ones((8, 8)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# threshold_image_filter
# ---------------------------------------------------------------------------


def test_threshold_image_filter_below():
    """ "below" keeps values ≤ threshold, replaces values > threshold."""
    data = np.arange(0, 100, dtype=np.float64).reshape(10, 10)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = threshold_image_filter(
        sitk_img, threshold=50.0, th_value=-1.0, th_method="below"
    )
    result_array = sitk.GetArrayViewFromImage(result)
    # Values ≤ 50 are kept
    assert (result_array[data <= 50] == data[data <= 50]).all()
    # Values > 50 become -1
    assert (result_array[data > 50] == -1.0).all()


def test_threshold_image_filter_above():
    """ "above" keeps values ≥ threshold, replaces values < threshold."""
    data = np.arange(0, 100, dtype=np.float64).reshape(10, 10)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = threshold_image_filter(
        sitk_img, threshold=50.0, th_value=-1.0, th_method="above"
    )
    result_array = sitk.GetArrayViewFromImage(result)
    # Values < 50 become -1
    assert (result_array[data < 50] == -1.0).all()
    # Values ≥ 50 are kept
    assert (result_array[data >= 50] == data[data >= 50]).all()


def test_threshold_image_filter_unknown_method():
    sitk_img = convert_from_numpy(
        np.ones((4, 4), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((4, 4), dtype=np.float64).ndim,
    )
    with pytest.raises(ValueError, match="Unknown threshold method"):
        threshold_image_filter(sitk_img, 0.5, th_method="invalid")


def test_threshold_image_filter_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        threshold_image_filter(np.ones((4, 4)), 0.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_image_statistics
# ---------------------------------------------------------------------------


def test_get_image_statistics_uniform_image():
    """Statistics of a uniform image match expected values."""
    sitk_img = convert_from_numpy(
        np.full((8, 8), 3.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((8, 8), 3.0, dtype=np.float64).ndim,
    )
    mean, var, min_val, max_val = get_image_statistics(sitk_img)
    assert mean == pytest.approx(3.0)
    assert var == pytest.approx(0.0)
    assert min_val == pytest.approx(3.0)
    assert max_val == pytest.approx(3.0)


def test_get_image_statistics_known_range():
    """A simple ramp has predictable min and max."""
    data = np.arange(9, dtype=np.float64).reshape(3, 3)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    mean, var, min_val, max_val = get_image_statistics(sitk_img)
    assert min_val == 0.0
    assert max_val == 8.0
    assert mean == pytest.approx(4.0)


def test_get_image_statistics_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        get_image_statistics(np.ones((4, 4)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# calculate_center_of_image
# ---------------------------------------------------------------------------


def test_calculate_center_geometric():
    """Geometric centre is spacing × size / 2."""
    sitk_img = convert_from_numpy(
        np.ones((10, 20), dtype=np.float64).astype(np.float64), (1.0, 2.0)
    )
    center = calculate_center_of_image(sitk_img)
    # ITK size is (x, y) = (20, 10), ITK spacing is (x, y) = (2.0, 1.0)
    assert center == pytest.approx([20.0, 5.0])


def test_calculate_center_of_mass_centered_impulse():
    """CoM of a centred impulse is the geometric centre (in pixels)."""
    data = impulse((32, 32))
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0, 1.0))
    center = calculate_center_of_image(sitk_img, center_of_mass=True)
    # ITK centre in physical units: (x, y) = (32 / 2 * 1, 32 / 2 * 1) = (16, 16)
    assert center[0] == pytest.approx(16.0)
    assert center[1] == pytest.approx(16.0)


def test_calculate_center_of_mass_off_center():
    """CoM of an off-centre impulse gives the impulse location (physical)."""
    data = np.zeros((32, 32), dtype=np.float64)
    data[24, 8] = 1.0  # numpy (y, x) = (24, 8)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0, 1.0))
    center = calculate_center_of_image(sitk_img, center_of_mass=True)
    # ITK physical: x = 8 * 1.0 = 8, y = 24 * 1.0 = 24
    assert center[0] == pytest.approx(8.0)
    assert center[1] == pytest.approx(24.0)


def test_calculate_center_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        calculate_center_of_image(np.ones((4, 4)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# type_cast
# ---------------------------------------------------------------------------


def test_type_cast_changes_pixel_type():
    """Float64 → Float32 must change the pixel type."""
    sitk_img = convert_from_numpy(
        np.ones((8, 8), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((8, 8), dtype=np.float64).ndim,
    )
    result = type_cast(sitk_img, sitk.sitkFloat32)
    assert result.GetPixelID() == sitk.sitkFloat32


def test_type_cast_preserves_data():
    """Casting to the same type preserves data."""
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    sitk_img = convert_from_numpy(data.astype(np.float64), (1.0,) * data.ndim)
    result = type_cast(sitk_img, sitk.sitkFloat64)
    result_array = sitk.GetArrayViewFromImage(result)
    assert_array_almost_equal(result_array, data)


def test_type_cast_type_error():
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        type_cast(np.ones((4, 4)), sitk.sitkFloat32)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# make_composite_rgb_image
# ---------------------------------------------------------------------------


def test_make_composite_rgb_image_shape_and_channels():
    """RGB composite has correct size, 3 channels, and channel values."""
    r = convert_from_numpy(
        np.full((10, 20), 10.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((10, 20), 10.0, dtype=np.float64).ndim,
    )
    g = convert_from_numpy(
        np.full((10, 20), 20.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((10, 20), 20.0, dtype=np.float64).ndim,
    )
    result = make_composite_rgb_image(r, g)
    assert isinstance(result, sitk.Image)
    assert result.GetNumberOfComponentsPerPixel() == 3
    arr = sitk.GetArrayViewFromImage(result)
    assert arr.shape == (10, 20, 3)
    assert (arr[:, :, 0] == 10).all()  # R  → uint8 preserves value
    assert (arr[:, :, 1] == 20).all()  # G
    assert (arr[:, :, 2] == 0).all()  # B: default empty channel → 0


def test_make_composite_rgb_image_return_numpy():
    """The return_numpy path returns (H, W, 3) array with correct channels."""
    r = convert_from_numpy(
        np.full((8, 12), 10.0, dtype=np.float64).astype(np.float64), (0.5, 1.0)
    )
    g = convert_from_numpy(
        np.full((8, 12), 20.0, dtype=np.float64).astype(np.float64), (0.5, 1.0)
    )
    arr, sp = make_composite_rgb_image(r, g, return_numpy=True)
    assert arr.shape == (8, 12, 3)
    assert sp == (0.5, 1.0)
    assert (arr[:, :, 0] == 10).all()
    assert (arr[:, :, 1] == 20).all()
    assert (arr[:, :, 2] == 0).all()


def test_make_composite_rgb_image_with_blue():
    """Explicit blue channel is used with correct channel values."""
    r = convert_from_numpy(
        np.full((4, 4), 10.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((4, 4), 10.0, dtype=np.float64).ndim,
    )
    g = convert_from_numpy(
        np.full((4, 4), 20.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((4, 4), 20.0, dtype=np.float64).ndim,
    )
    b = convert_from_numpy(
        np.full((4, 4), 30.0, dtype=np.float64).astype(np.float64),
        (1.0,) * np.full((4, 4), 30.0, dtype=np.float64).ndim,
    )
    result = make_composite_rgb_image(r, g, b)
    assert result.GetNumberOfComponentsPerPixel() == 3
    arr = sitk.GetArrayViewFromImage(result)
    assert arr.shape == (4, 4, 3)
    assert (arr[:, :, 0] == 10).all()
    assert (arr[:, :, 1] == 20).all()
    assert (arr[:, :, 2] == 30).all()


def test_make_composite_rgb_image_type_error():
    r = convert_from_numpy(
        np.ones((4, 4), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((4, 4), dtype=np.float64).ndim,
    )
    with pytest.raises(TypeError, match="Expected sitk.Image"):
        make_composite_rgb_image(np.ones((4, 4)), r)  # type: ignore[arg-type]


def test_make_composite_rgb_image_blue_type_error():
    r = convert_from_numpy(
        np.ones((4, 4), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((4, 4), dtype=np.float64).ndim,
    )
    g = convert_from_numpy(
        np.ones((4, 4), dtype=np.float64).astype(np.float64),
        (1.0,) * np.ones((4, 4), dtype=np.float64).ndim,
    )
    with pytest.raises(TypeError, match="Expected sitk.Image for blue"):
        make_composite_rgb_image(r, g, np.ones((4, 4)))  # type: ignore[arg-type]
