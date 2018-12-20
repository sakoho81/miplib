import SimpleITK as sitk
import pims
import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image
from miplib.data.io import tiffile


def image(path, image):
    """
    A wrapper for the various image writing functions. The consumers
    should only call this function

    :param path:    A full path to the image.
    :param image:   An image as :type image: numpy.ndarray or sitk.image.

    :return:
    """

    assert isinstance(image, Image)

    if path.endswith(('.tiff', '.tif')):
        __tiff(path, image, image.spacing)
    else:
        __itk_image(path, image)


def __itk_image(path, image):
    """
    A writer for ITK supported image formats.

    :param path:    A full path to the image.
    :param image:   An image as :type image: numpy.ndarray.
    :param spacing: Pixel size ZXY, as a :type spacing: list.
    """
    assert isinstance(image, Image)

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
    tiffile.imsave(path,
                   image,
                   imagej=True,
                   resolution=list(1.0/x for x in spacing))


def __tiff(path, image, spacing):
    """
    Write a TIFF. Will be automatically converted into BigTIFF, if the
    file is too big for regulare TIFF definition.

    :param path:    A full path to the image.
    :param image:   An image as Numpy.ndarray.
    :param spacing: Pixel size ZXY, as a list.
    """

    if image.ndim == 3:
        image_description = "images={} slices={} unit=micron spacing={}".format(image.shape[0],
                                                                                image.shape[0],
                                                                                spacing[0])
        tiffile.imsave(path,
                       image,
                       resolution=(1.0/spacing[1], 1.0/spacing[2]),
                       metadata={'description': image_description})
    else:
        tiffile.imsave(path,
                       image,
                       imagej=True,
                       resolution=(1.0 / spacing[0], 1.0 / spacing[1]))

