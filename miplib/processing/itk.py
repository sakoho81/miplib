"""
itkutils.py

Copyright (C) 2014 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains several utilities & filters for simplified
usage of ITK (www.itk.org) modules in Python. Most of the ITK classes
have been implemented in similar manner, so it should be rather
easy to include additional filters.

"""

import logging
from collections.abc import Sequence

import numpy as np
import scipy
import SimpleITK as sitk

from miplib.data.containers.image import Image
from miplib.processing import converters

logger = logging.getLogger(__name__)


def convert_from_itk_image(image: sitk.Image) -> Image:
    """Convert an ITK Image to a miplib Image with numpy-ordered spacing."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    array = sitk.GetArrayFromImage(image)
    # ITK stores dimensions in (x, y, z) order; numpy uses (z, y, x).
    # The array conversion re-orders dimensions, so spacing must be reversed.
    spacing = image.GetSpacing()[::-1]

    return Image(array, spacing)


def convert_to_itk_image(image: Image) -> sitk.Image:
    """Convert a miplib Image to an ITK Image."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    return convert_from_numpy(image, image.spacing)


def convert_from_numpy(array: np.ndarray, spacing: Sequence[float]) -> sitk.Image:
    """Convert a numpy array to a SimpleITK Image with given spacing."""
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(array).__name__}")

    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing[::-1])

    return image


def make_itk_transform(
    type: str,
    dims: int,
    parameters: Sequence[float],
    fixed_parameters: Sequence[float],
) -> sitk.Transform:
    """Construct an ITK spatial transform from known parameters.

    :param type:             ITK transform type, currently only ``"AffineTransform"``
    :param dims:             number of dimensions
    :param parameters:       transform parameters
    :param fixed_parameters: transform fixed parameters
    """
    if type == "AffineTransform":
        transform: sitk.Transform = sitk.AffineTransform(dims)
    else:
        raise NotImplementedError(f"Unsupported transform type {type!r}")

    transform.SetParameters(parameters)
    transform.SetFixedParameters(fixed_parameters)

    return transform


def get_itk_transform_parameters(
    transform: sitk.Transform,
) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    """Extract transform name, parameters, and fixed parameters."""
    tfm_type = transform.GetName()
    params = transform.GetParameters()
    fixed_params = transform.GetFixedParameters()

    return tfm_type, params, fixed_params


def resample_image(
    image: sitk.Image,
    transform: sitk.Transform,
    reference: sitk.Image | None = None,
    interpolation: str = "linear",
) -> sitk.Image:
    """Resample an image under a spatial transform.

    :param image:         input ITK image
    :param transform:     spatial transform to apply
    :param reference:     reference image defining output grid
                          (defaults to *image*)
    :param interpolation: ``"nearest"``, ``"linear"``, or ``"Bspline"``
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    if reference is None:
        reference = image

    if interpolation == "nearest":
        interpolator = sitk.sitkNearestNeighbor
    elif interpolation == "linear":
        interpolator = sitk.sitkLinear
    elif interpolation == "Bspline":
        interpolator = sitk.sitkBSpline
    else:
        raise ValueError(f"Unknown interpolation type {interpolation!r}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)

    resampler.SetInterpolator(interpolator)
    resampler.SetOutputPixelType(reference.GetPixelID())
    resampler.SetSize(reference.GetSize())
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def rotate_image(
    image: sitk.Image,
    angle: float,
    axis: int = 0,
    interpolation: str = "linear",
) -> sitk.Image:
    """Rotate an image around the selected axis.

    :param image:         a SimpleITK image
    :param angle:         rotation angle in degrees
    :param axis:          rotation axis (0-based)
    :param interpolation: ``"nearest"``, ``"linear"``, or ``"Bspline"``
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    radians = converters.degrees_to_radians(angle)

    if image.GetDimension() == 3:
        transform: sitk.Transform = sitk.Euler3DTransform()
        rotation = [0.0, 0.0, 0.0]
        rotation[axis] = radians
        transform.SetRotation(*rotation)  # type: ignore[attr-defined]  # type: ignore[attr-defined]
    elif image.GetDimension() == 2:
        transform = sitk.Euler2DTransform()
        transform.SetAngle(radians)
    else:
        raise ValueError(
            f"rotate_image supports 2D and 3D only, got {image.GetDimension()}D"
        )

    transform.SetCenter(calculate_center_of_image(image))  # type: ignore[attr-defined]

    return resample_image(image, transform, interpolation=interpolation)


def rotate_psf(
    psf: np.ndarray | sitk.Image,
    transform: sitk.Transform,
    spacing: Sequence[float] | None = None,
    return_numpy: bool = False,
) -> Image | sitk.Image:
    """Rotate a PSF by stripping the translation part of a transform.

    When a single PSF is used for multi-view fusion it must be rotated
    with the same rotation that was recovered during registration.

    :param psf:           a numpy array or SimpleITK image of the PSF
    :param transform:     the registration transform whose rotation is used
    :param spacing:       pixel spacing (required if *psf* is an ndarray)
    :param return_numpy:  if True, return a miplib Image instead of sitk.Image
    """
    if isinstance(psf, np.ndarray):
        if spacing is None:
            raise ValueError("spacing is required when psf is an ndarray")
        image = convert_from_numpy(psf, spacing)
    else:
        image = psf

    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    if isinstance(transform, sitk.AffineTransform):
        n_entries = len(transform.GetMatrix())
        ndim = int(np.sqrt(n_entries))
        array = np.array(transform.GetMatrix()).reshape(ndim, ndim)
        rotation = scipy.linalg.polar(array, "right")[0]
        matrix = tuple(rotation.ravel())
        transform.SetMatrix(matrix)
        transform.SetTranslation((0.0,) * ndim)
        center = calculate_center_of_image(image)
        transform.SetFixedParameters(center)
    else:
        # Zero out translation parameters (always the last ndim entries)
        params = list(transform.GetParameters())
        ndim = image.GetDimension()
        for i in range(len(params) - ndim, len(params)):
            params[i] = 0.0
        transform.SetParameters(tuple(params))
        center = calculate_center_of_image(image)
        # Euler transforms store an extra w-component (1.0) in fixed parameters
        if len(transform.GetFixedParameters()) == len(center) + 1:
            center.append(1.0)
        transform.SetFixedParameters(center)

    image = resample_image(image, transform)

    if return_numpy:
        return convert_from_itk_image(image)
    else:
        return image


def resample_to_isotropic(itk_image: sitk.Image) -> sitk.Image:
    """Resample a 3D confocal stack to isotropic pixel spacing.

    :param itk_image: a 3D ITK image with anisotropic Z spacing
    :return:          a resampled ITK image with isotropic spacing
    """
    if not isinstance(itk_image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(itk_image).__name__}")

    method = sitk.ResampleImageFilter()
    transform = sitk.Transform()
    transform.SetIdentity()

    method.SetInterpolator(sitk.sitkBSpline)
    method.SetDefaultPixelValue(0)

    spacing = list(itk_image.GetSpacing())

    if len(spacing) != 3:
        raise ValueError(
            f"resample_to_isotropic requires a 3D image, got {len(spacing)}D"
        )

    scaling = spacing[2] / spacing[0]
    spacing[:] = [spacing[0]] * 3

    method.SetOutputSpacing(spacing)
    method.SetOutputDirection(itk_image.GetDirection())
    method.SetOutputOrigin(itk_image.GetOrigin())

    size = list(itk_image.GetSize())
    size[2] = int(size[2] * scaling)
    method.SetSize(size)

    transform.SetIdentity()
    method.SetTransform(transform)

    return method.Execute(itk_image)


def rescale_intensity(image: sitk.Image) -> sitk.Image:
    """Scale intensities to the full range of the pixel type.

    Currently only supports 8-bit unsigned integer images.
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.RescaleIntensityImageFilter()
    image_type = image.GetPixelIDTypeAsString()
    if image_type == "8-bit unsigned integer":
        method.SetOutputMinimum(0)
        method.SetOutputMaximum(255)
    else:
        logger.warning(
            "The rescale intensity filter has not been implemented for %s",
            image_type,
        )
        return image

    return method.Execute(image)


def gaussian_blurring_filter(image: sitk.Image, variance: float) -> sitk.Image:
    """Apply a Gaussian blur with the given variance."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    filter = sitk.DiscreteGaussianImageFilter()
    filter.SetUseImageSpacing(False)
    filter.SetVariance(variance)

    return filter.Execute(image)


def grayscale_dilate_filter(image: sitk.Image, kernel_radius: int) -> sitk.Image:
    """Apply a grayscale dilation with a ball structuring element."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.GrayscaleDilateImageFilter()
    method.SetKernelRadius(kernel_radius)
    method.SetKernelType(sitk.sitkBall)

    return method.Execute(image)


def mean_filter(image: sitk.Image, kernel_radius: int) -> sitk.Image:
    """Apply a uniform mean (box) filter."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.MeanImageFilter()
    method.SetRadius(kernel_radius)

    return method.Execute(image)


def median_filter(image: sitk.Image, kernel_radius: int) -> sitk.Image:
    """Apply a median filter.

    :param image:         a SimpleITK image
    :param kernel_radius: median kernel radius
    :return:              filtered image
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.MedianImageFilter()
    kernel = [
        kernel_radius,
    ] * image.GetDimension()
    method.SetRadius(kernel)

    return method.Execute(image)


def normalize_image_filter(image: sitk.Image) -> sitk.Image:
    """Normalize pixel values to zero mean and unit variance."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.NormalizeImageFilter()
    return method.Execute(image)


def threshold_image_filter(
    image: sitk.Image,
    threshold: float,
    th_value: float = 0,
    th_method: str = "below",
) -> sitk.Image:
    """Threshold a grayscale image by setting values outside range to *th_value*.

    :param image:     a SimpleITK image
    :param threshold: threshold value
    :param th_value:  replacement value for pixels outside the threshold
    :param th_method: ``"below"`` — keep values ≤ threshold, replace others;
                      ``"above"`` — keep values ≥ threshold, replace others
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.ThresholdImageFilter()
    if th_method == "above":
        method.SetLower(threshold)
        method.SetUpper(float(np.finfo(np.float64).max))
    elif th_method == "below":
        method.SetUpper(threshold)
        method.SetLower(float(np.finfo(np.float64).min))
    else:
        raise ValueError(f"Unknown threshold method {th_method!r}")

    method.SetOutsideValue(th_value)

    return method.Execute(image)


def get_image_statistics(image: sitk.Image) -> tuple[float, float, float, float]:
    """Return (mean, variance, min, max) of an ITK image."""
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.StatisticsImageFilter()
    method.Execute(image)
    mean = method.GetMean()
    variance = method.GetVariance()
    max_val = method.GetMaximum()
    min_val = method.GetMinimum()

    return mean, variance, min_val, max_val


def type_cast(image: sitk.Image, output_type: int) -> sitk.Image:
    """Cast an ITK image to the given pixel type.

    :param image:       an ITK Image
    :param output_type: output pixel type (e.g. ``sitk.sitkFloat32``)
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    method = sitk.CastImageFilter()
    method.SetOutputPixelType(output_type)

    return method.Execute(image)


def calculate_center_of_image(
    image: sitk.Image, center_of_mass: bool = False
) -> list[float]:
    """Return the center of an image in physical coordinates.

    :param image:          a SimpleITK image
    :param center_of_mass: if True, use the intensity-weighted center;
                           otherwise use the geometric centre
    :return:               centre in physical units (ITK axis order)
    """
    if not isinstance(image, sitk.Image):
        raise TypeError(f"Expected sitk.Image, got {type(image).__name__}")

    imdims = image.GetSize()
    imspacing = image.GetSpacing()

    if center_of_mass:
        img = convert_from_itk_image(image)
        com = scipy.ndimage.center_of_mass(img)
        # com is in numpy (z, y, x) order; reverse to ITK (x, y, z) order
        # and convert from pixel to physical coordinates
        center = [c * s for c, s in zip(com[::-1], imspacing, strict=False)]
    else:
        center = [s * d / 2 for s, d in zip(imspacing, imdims, strict=False)]

    return center


def make_composite_rgb_image(
    red: sitk.Image,
    green: sitk.Image,
    blue: sitk.Image | None = None,
    return_numpy: bool = False,
) -> sitk.Image | tuple[np.ndarray, tuple[float, ...]]:
    """Combine two or three grayscale images into an RGB composite.

    :param red:          red channel image (sitk.Image)
    :param green:        green channel image (sitk.Image)
    :param blue:         blue channel image; if None an empty channel is used
    :param return_numpy: if True, return a (H, W, 3) numpy array and spacing
    :return:             an RGB composite image
    """
    if not isinstance(red, sitk.Image) or not isinstance(green, sitk.Image):
        raise TypeError(
            f"Expected sitk.Image for red and green, "
            f"got {type(red).__name__} and {type(green).__name__}"
        )

    red = sitk.Cast(red, sitk.sitkUInt8)
    green = sitk.Cast(green, sitk.sitkUInt8)
    if blue is not None:
        if not isinstance(blue, sitk.Image):
            raise TypeError(f"Expected sitk.Image for blue, got {type(blue).__name__}")
        blue = sitk.Cast(blue, sitk.sitkUInt8)
    else:
        blue = sitk.Image(red.GetSize(), sitk.sitkUInt8)
        blue.CopyInformation(red)

    if return_numpy:
        arrays = (
            sitk.GetArrayFromImage(red),
            sitk.GetArrayFromImage(green),
            sitk.GetArrayFromImage(blue),
        )
        spacing = red.GetSpacing()[::-1]
        return np.stack(arrays, axis=-1), spacing
    else:
        return sitk.Compose(red, green, blue)


def make_translation_transforms_from_offsets(
    offsets: Sequence[Sequence[float]],
) -> list[sitk.TranslationTransform]:
    """Create translation transforms from a list of offsets.

    :param offsets: each element is a per-axis offset in N dimensions
    :return:        list of ITK translation transforms
    """
    ndims = len(offsets[0])
    transforms = []

    for offset in offsets:
        tfm = sitk.TranslationTransform(ndims)
        tfm.SetParameters(offset)
        transforms.append(tfm)

    return transforms
