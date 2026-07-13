from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk

import miplib.processing.ism.helpers as ismutils
from miplib.data.adapters.registration_data import ArrayDetectorDataSource
from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image
from miplib.processing import itk
from miplib.processing import transform as tfm
from miplib.processing.registration import MultiViewRegistration, stack
from miplib.processing.registration.options import (
    RegistrationMethod,
    RegistrationOptions,
)


def find_image_shifts(
    data: ArrayDetectorData,
    method: RegistrationMethod = RegistrationMethod.ITERATIVE_RIGID,
    photosensor: int = 0,
    fixed_idx: int = 12,
    options: RegistrationOptions | None = None,
) -> tuple[np.ndarray, list[sitk.Transform]]:
    """Register all detector images to a reference detector.

    Args:
        data: ArrayDetectorData with all the individual images.
        method: Registration method preset.
        photosensor: Photosensor index (first gate).
        fixed_idx: Index of the reference detector. Defaults to 12 (IIT SPAD).
        options: RegistrationOptions for fine-grained control.

    Returns:
        (shifts, transforms): shifts in physical units (µm), and the
        corresponding ITK transforms.
    """
    if photosensor >= data.ngates:
        raise ValueError(
            f"photosensor={photosensor} out of range (ngates={data.ngates})"
        )

    source = ArrayDetectorDataSource(data, photosensor=photosensor)
    if options is None:
        options = RegistrationOptions(method=method, translate_only=True)
    mv = MultiViewRegistration(
        source, method=method, fixed_idx=fixed_idx, options=options
    )
    transforms = mv.execute()

    ndim = data[photosensor, 0].ndim
    shifts = np.zeros((data.ndetectors, ndim), dtype=np.float64)
    for idx, transform in enumerate(transforms):
        params = np.asarray(transform.GetParameters())
        shifts[idx] = params[::-1]

    return shifts, transforms


def find_static_image_shifts(
    pitch: float,
    wavelength: float,
    fov: float,
    na: float,
    alpha: float = 0.5,
    width: int = 5,
    rotation: float = 0.0,
) -> tuple[list[float], list[float], list[sitk.TranslationTransform]]:
    """Generate spatial transforms based on theoretical ISM parameters.

    Args:
        pitch: Detector pixel spacing.
        wavelength: Wavelength for Airy disk calculation.
        fov: SPAD field of view in Airy units.
        na: Objective numerical aperture.
        alpha: Reassignment factor in ]0, 1].
        width: Number of detectors along one dimension of the SPAD.
        rotation: Optional rotation angle in radians.

    Returns:
        (x_offsets, y_offsets, transforms): offsets in µm and ITK transforms.
    """
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in ]0, 1]")

    d_airy = 1.22 * wavelength / na
    d_detector_sp = fov * d_airy
    d_detector_ip = pitch * width

    magnification = d_detector_ip / d_detector_sp

    x, y = ismutils.calculate_theoretical_shifts_xy(pitch, magnification, alpha=alpha)
    if rotation != 0:
        x, y = tfm.rotate_xy_points_lists(y, x, rotation)

    transforms = tfm.make_translation_transforms_from_xy(y, x)
    return x, y, transforms


def shift_and_sum(
    data: ArrayDetectorData,
    transforms: list[sitk.Transform],
    photosensor: int = 0,
    detectors: Sequence[int] | None = None,
    supersampling: float = 1.0,
) -> Image:
    """Adaptive ISM pixel reassignment.

    Args:
        data: ArrayDetectorData with all the individual images.
        transforms: ITK transforms for each detector.
        photosensor: Photosensor index.
        detectors: Subset of detector indices to use. None means all.
        supersampling: Upsampling factor for output.

    Returns:
        Reconstructed Image.
    """
    if not isinstance(transforms, list) or len(transforms) != data.ndetectors:
        raise ValueError(
            f"Expected list of {data.ndetectors} transforms, "
            f"got {len(transforms) if isinstance(transforms, list) else type(transforms).__name__}"
        )

    if supersampling != 1.0:
        new_shape = [int(i * supersampling) for i in data[photosensor, 0].shape]
        new_spacing = tuple(i / supersampling for i in data[photosensor, 0].spacing)
        output = Image(np.zeros(new_shape, dtype=np.float64), new_spacing)
    else:
        output = Image(
            np.zeros(data[photosensor, 0].shape, dtype=np.float64),
            data[photosensor, 0].spacing,
        )

    if detectors is None:
        detectors = list(range(data.ndetectors))

    for i in detectors:
        resampled = itk.resample_image(
            itk.convert_to_itk_image(data[photosensor, i]),
            transforms[i],
            reference=itk.convert_to_itk_image(output),
        )
        output += itk.convert_from_itk_image(resampled)  # type: ignore[misc]

    return output


def shift(
    data: ArrayDetectorData, transforms: list[sitk.Transform]
) -> ArrayDetectorData:
    """Resample all images with the supplied transforms.

    Args:
        data: ArrayDetectorData with images.
        transforms: ITK transforms for each detector.

    Returns:
        New ArrayDetectorData with shifted images.
    """
    if not isinstance(transforms, list) or len(transforms) != data.ndetectors:
        raise ValueError(f"Expected list of {data.ndetectors} transforms")

    shifted = ArrayDetectorData(data.ndetectors, data.ngates)

    for gate in range(data.ngates):
        for idx in range(data.ndetectors):
            image = itk.resample_image(
                itk.convert_to_itk_image(data[gate, idx]), transforms[idx]
            )
            shifted[gate, idx] = itk.convert_from_itk_image(image)

    return shifted


def sum_images(
    data: ArrayDetectorData,
    photosensor: int = 0,
    detectors: Sequence[int] | None = None,
) -> Image:
    """Sum all (or a subset of) images in a detector array.

    Args:
        data: ArrayDetectorData with images.
        photosensor: Photosensor index.
        detectors: Subset of detector indices to sum. None means all.

    Returns:
        Summed Image.
    """
    if detectors is None:
        detectors = list(range(data.ndetectors))

    result = np.zeros(data[0, 0].shape, dtype=np.float64)

    for i in detectors:
        result += data[photosensor, i]

    return Image(result, data[0, 0].spacing)


def drift_correct_ism_stack(data: ArrayDetectorData) -> ArrayDetectorData:
    """Correct for xy-drift in 3D ISM datasets.

    Args:
        data: The data.

    Returns:
        Drift-corrected data.
    """
    sum_image = sum_images(data)
    shifts = stack.register_stack_slices(sum_image)

    result = ArrayDetectorData(data.ndetectors, data.ngates)

    for g_idx in range(data.ngates):
        for c_idx in range(data.ndetectors):
            result[g_idx, c_idx] = stack.shift_stack_slices(data[g_idx, c_idx], shifts)

    return result
