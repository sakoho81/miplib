import SimpleITK as sitk
import tifffile

import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image


def image(path, image):
    """Write an Image to disk, dispatching on file extension.

    :param path:    A full path to the image.
    :param image:   An Image (ndarray subclass with spacing).
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    if path.endswith((".tiff", ".tif")):
        __tiff(path, image, image.spacing)
    elif path.endswith((".mha", ".mhd")):
        __itk_image(path, image)
    else:
        raise ValueError(
            f"Unsupported file extension: {path}. Expected .tif, .tiff, .mha, or .mhd."
        )


def __itk_image(path, image):
    """Write an Image in ITK format (.mha/.mhd)."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    image = itkutils.convert_to_itk_image(image)
    sitk.WriteImage(image, path)


def __imagej_tiff(path, image, spacing):
    """
    Write a TIFF in ImageJ mode. May improve compatibility with
    older imageJ. I would recommend using the other one instead.
    :param path:    A full path to the image.
    :param image:   An image as :type image: numpy.ndarray.
    :param spacing: Pixel size ZXY, as a :type spacing: list.
    """
    tifffile.imsave(path, image, imagej=True, resolution=[1.0 / x for x in spacing])


def __tiff(path, image, spacing):
    """
    Write a TIFF. Will be automatically converted into BigTIFF, if the
    file is too big for regulare TIFF definition.

    :param path:    A full path to the image.
    :param image:   An image as Numpy.ndarray.
    :param spacing: Pixel size ZXY, as a list.
    """

    if image.ndim >= 3:
        image_description = f"images={image.shape[0]} slices={image.shape[0]} unit=micron spacing={spacing[0]}"
        tifffile.imsave(
            path,
            image,
            resolution=(1.0 / spacing[1], 1.0 / spacing[2]),
            metadata={"description": image_description},
        )
    else:
        tifffile.imsave(
            path, image, imagej=True, resolution=(1.0 / spacing[0], 1.0 / spacing[1])
        )
