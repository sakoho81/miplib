import numpy as np

from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType


def _to_imagetype(image_type: ImageType | str) -> ImageType:
    if isinstance(image_type, ImageType):
        return image_type
    return ImageType(image_type)


def sum_of_all(
    data_structure: ImageData,
    channel: int = 0,
    scale: int = 100,
    image_type: ImageType | str = ImageType.ORIGINAL,
) -> Image:
    """Sum all views of *image_type* into a single image."""
    if not isinstance(data_structure, ImageData):
        raise TypeError(f"Expected ImageData, got {type(data_structure).__name__}")

    img_type = _to_imagetype(image_type)
    n_views = data_structure.get_number_of_images(img_type)
    key0 = ImageKey(img_type, 0, channel, scale)
    result = np.zeros(data_structure.get_image_size(key0), dtype=np.float32)
    pixel_size = data_structure.get_voxel_size(key0)

    for i in range(n_views):
        key = ImageKey(img_type, i, channel, scale)
        result += data_structure.get_image_data(key)

    return Image(result, pixel_size)


def average_of_all(
    data_structure: ImageData,
    channel: int = 0,
    scale: int = 100,
    image_type: ImageType | str = ImageType.ORIGINAL,
) -> Image:
    """Average all views of *image_type* into a single image."""
    if not isinstance(data_structure, ImageData):
        raise TypeError(f"Expected ImageData, got {type(data_structure).__name__}")

    img_type = _to_imagetype(image_type)
    n_views = data_structure.get_number_of_images(img_type)
    key0 = ImageKey(img_type, 0, channel, scale)
    pixel_size = data_structure.get_voxel_size(key0)

    result = sum_of_all(data_structure, channel, scale, img_type)

    return Image(result / n_views, pixel_size)


def simple_fusion(
    data_structure: ImageData,
    channel: int = 0,
    scale: int = 100,
) -> Image:
    """Fuse registered views using a simple min-clip accumulation."""
    if not isinstance(data_structure, ImageData):
        raise TypeError(f"Expected ImageData, got {type(data_structure).__name__}")

    image_type = ImageType.REGISTERED
    n_views = data_structure.get_number_of_images(image_type)
    key0 = ImageKey(image_type, 0, channel, scale)
    pixel_size = data_structure.get_voxel_size(key0)

    result = data_structure.get_image_data(key0)

    for i in range(1, n_views):
        key = ImageKey(image_type, i, channel, scale)
        result = (
            (result - (result - data_structure.get_image_data(key)).clip(min=0))
            .clip(min=0)
            .astype(np.float32)
        )

    return Image(result, pixel_size)
