from math import floor

import numpy as np
import SimpleITK as sitk

from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image
from miplib.processing import itk
from miplib.processing.registration import registration
from miplib.processing.windowing import apply_hamming_window

import miplib.processing.ism.helpers as ismutils
from miplib.processing import transform as tfm


def find_image_shifts(data, options, photosensor=0, fixed_idx=12):
    """
    Register all images in an ISM ArrayDetectorData dataset. The central image (pixel 12)
    is used as reference and other images are aligned with it. This function used an
    iterative algorithm (ITK) that can be customized with various command line options.


    :param options: Various options that can be used on fine tune the image registration. Look into
    supertomo_options.py file
    :param data: ArrayDetectorData object with all the individual images
    :param photosensor: The photosensor number (from 0 upwards) that is to be processed
    :param fixed_idx: The index of the reference image. Defaults to 12 (IIT SPAD array)
    :return: a three element tuple: x offset, y offset, transforms. The x and y offsets are expressed
    in physical units (um). The transforms are sitk.TranslationTransform objects that can be used
    to resample the images into a common coordinate system.
    """
    assert isinstance(data, ArrayDetectorData)
    assert photosensor < data.ngates

    fixed_image = itk.convert_to_itk_image(data[photosensor, fixed_idx])
    x = []
    y = []
    transforms = []

    for idx in range(data.ndetectors):
        image = data[photosensor, idx]
        moving_image = itk.convert_to_itk_image(image)
        transform = registration.itk_registration_rigid_2d(fixed_image, moving_image, options)
        x_new, y_new = transform.GetParameters()
        x.append(x_new)
        y.append(y_new)
        transforms.append(transform)

    return x, y, transforms


def find_static_image_shifts(pitch, wavelength, fov, na, alpha=0.5, width=5, rotation=0):
    """
    Generate spatial transforms for ISM image reconstruction, based on theoretical values.
    :param pitch: the detector pixel spacing
    :param wavelength: the wavelength to be used in the calculations. Can be e.g. the average
    of the excitation and emission wavelengths
    :param fov: the size of the SPAD field of view in Airy units
    :param na: the objective numerical aperture
    :param alpha: the reassignment factor ]0, 1]
    :param width: the number of detectors along one dimension of the SPAD.
    :return: a list of ITK transforms that can be used to resample the images.
    """
    assert 0 < alpha <= 1

    d_airy = 1.22 * wavelength / na
    d_detector_sp = fov*d_airy
    d_detector_ip = pitch*width

    magnification = d_detector_ip/d_detector_sp

    x,y = ismutils.calculate_theoretical_shifts_xy(pitch, magnification, alpha=alpha)
    if rotation != 0:
        x,y = tfm.rotate_xy_points_lists(y, x, rotation)

    return x, y, tfm.make_translation_transforms_from_xy(y, x)


def find_image_shifts_frequency_domain(data, photosensor=0):
    """
    Register all image in an ISM ArrayDetectorDAta dataset, with a single step frequency domain
    phase correlation based method. This might be slightly faster than the iterative method
    above (depending on the sampling strategy in the latter mainly), but usually does not
    work quite as well.

    :param data: ArrayDetectorData object with all the individual images
    :param photosensor: The photosensor number (from 0 upwards) that is to be processed
    :return: a three element tuple: x offset, y offset, transforms. The x and y offsets are expressed
    in physical units (um). The transforms are sitk.TranslationTransform objects that can be used
    to resample the images into a common coordinate system.
    """
    assert isinstance(data, ArrayDetectorData)
    assert photosensor < data.ngates

    spacing = data[0,0].spacing
    fixed_image = Image(apply_hamming_window(data[photosensor, int(floor(data.ndetectors / 2))]), spacing)
    x = []
    y = []
    transforms = []

    for idx in range(data.ndetectors):
        moving_image = Image(apply_hamming_window(data[photosensor, idx]), spacing)
        y_new, x_new = registration.phase_correlation_registration(fixed_image, moving_image,
                                                                   verbose=True, resample=False)
        tfm = sitk.TranslationTransform(2)
        tfm.SetParameters((x_new, y_new))

        x.append(x_new)
        y.append(y_new)
        transforms.append(tfm)

    return x, y, transforms


def shift_and_sum(data, transforms, photosensor=0, detectors=None, supersampling=1.0):
    """
    Adaptive ISM pixel reassignment. Please use one of the functions above to figure out
    the shifts first, if you haven't already.

    :param supersampling: Insert a number != 1, if you want to rescale the result image to
    a different size. This might make sense, if you the original sampling has been sampled
    sparsely
    :param data: ArrayDetectorData object with all the individual images
    :param transforms: ITK spatial transformation that are to be used for the resampling
    :param photosensor: The photosensor index, if more than one
    :param detectors: a list of detectors to be included in the reconstruction. If None given (default),
    all the images will be used
    :return: reconstruction result Image
    """
    assert isinstance(data, ArrayDetectorData)
    assert isinstance(transforms, list) and len(transforms) == data.ndetectors

    if supersampling != 1.0:
        new_shape = list(int(i*supersampling) for i in data[photosensor, 0].shape)
        new_spacing = list(i/supersampling for i in data[photosensor, 0].spacing)
        output = Image(np.zeros(new_shape, dtype=np.float64), new_spacing)
    else:
        output = Image(np.zeros(data[photosensor, 0].shape, dtype=np.float64), data[photosensor, 0].spacing)

    if detectors is None:
        detectors = list(range(data.ndetectors))

    for i in detectors:
        image = itk.resample_image(
            itk.convert_to_itk_image(data[photosensor, i]),
            transforms[i],
            reference=itk.convert_to_itk_image(output))

        output += itk.convert_from_itk_image(image)

    return output


def shift(data, transforms):
    """
    Resamples all the images in an ArrayDetectorData structure with the supplied transforms,
    and saves the result in a new ArrayDetectorData structure

    :param data: ArrayDetectorData object with images
    :param transforms: A list of transforms (Simple ITK), one for each image
    :return: ArrayDetectorDAta object with shifted images
    """

    assert isinstance(data, ArrayDetectorData)
    assert isinstance(transforms, list) and len(transforms) == data.ndetectors

    shifted = ArrayDetectorData(data.ndetectors, data.ngates)

    for gate in range(data.ngates):
        for i in range(data.ndetectors):
            image = itk.resample_image(
                itk.convert_to_itk_image(data[gate, i]),
                transforms[i])

            shifted[gate, i] = itk.convert_from_itk_image(image)

    return shifted


def sum(data, photosensor=0, detectors=None):
    """
    Sums all the images in a ArrayDetectorData structure

    :param detectors: A subset of detectors to be summed. If left empty, all the images
    will be summed
    :param photosensor: The photosensor index.
    :param data: ArrayDetectorData object with images
    :return: result Image
    """
    assert isinstance(data, ArrayDetectorData)

    if detectors is None:
        detectors = list(range(data.ndetectors))

    result = np.zeros(data[0,0].shape, dtype=np.float64)

    for i in detectors:
        result += data[photosensor, i]

    return Image(result, data[0, 0].spacing)

