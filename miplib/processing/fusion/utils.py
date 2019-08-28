import numpy as np

from miplib.data.containers.image_data import ImageData
from miplib.data.containers.image import Image


def sum_of_all(data_structure, channel=0, scale=100, image_type="original"):
    assert isinstance(data_structure, ImageData)

    n_views = data_structure.get_number_of_images(image_type)
    data_structure.set_active_image(0, channel, scale, image_type)
    result = np.zeros(data_structure.get_image_size(), dtype=np.float32)
    pixel_size = data_structure.get_voxel_size()

    for i in range(n_views):
        data_structure.set_active_image(i, channel, scale, image_type)
        result += data_structure[:]

    return Image(result, pixel_size)


def average_of_all(data_structure, channel=0, scale=100, image_type="original"):
    assert isinstance(data_structure, ImageData)
    n_views = data_structure.get_number_of_images(image_type)
    data_structure.set_active_image(0, channel, scale, image_type)
    pixel_size = data_structure.get_voxel_size()

    result = sum_of_all(data_structure, channel, scale, image_type)

    return Image(result/n_views, pixel_size)


def simple_fusion(data_structure, channel=0, scale=100):
    assert isinstance(data_structure, ImageData)
    image_type = "registered"

    n_views = data_structure.get_number_of_images(image_type)
    data_structure.set_active_image(0, channel, scale, image_type)
    pixel_size = data_structure.get_voxel_size()

    result = data_structure[:]

    for i in range(1, n_views):
        data_structure.set_active_image(i, channel, scale, image_type)
        result = (result - (result - data_structure[:]).clip(min=0)).clip(min=0).astype(np.float32)

    return Image(result, pixel_size)