import numpy as np

from supertomo.data.containers.array_detector_data import ArrayDetectorData
from supertomo.data.containers.image import Image
from supertomo.processing.registration import registration
from supertomo.processing import itk
from math import floor


def find_image_shifts(data, options, gate=0):
    assert isinstance(data, ArrayDetectorData)
    assert gate < data.ngates

    fixed_image = itk.convert_to_itk_image(data[gate, int(floor(data.ndetectors/2))])
    x = []
    y = []
    transforms = []

    for image in data:
        moving_image = itk.convert_to_itk_image(image)
        transform = registration.itk_registration_rigid_2d(fixed_image, moving_image, options)
        x_new, y_new = transform.GetParameters()
        x.append(x_new)
        y.append(y_new)
        transforms.append(transform)

    return x, y, transforms


def shift_and_sum(data, transforms, gate=0):
    assert isinstance(data, ArrayDetectorData)
    assert isinstance(transforms, list) and len(transforms) == data.ndetectors

    output = Image(np.zeros(data[gate, 0].shape, dtype=np.float64), data[gate, 0].spacing)

    for i in range(data.ndetectors):
        image = itk.resample_image(
            itk.convert_to_itk_image(data[gate, i]),
            transforms[i])

        output += itk.convert_from_itk_image(image)

    return output

