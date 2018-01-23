import SimpleITK as sitk
from supertomo.data.io import tiffile
from supertomo.utils import itkutils


def image(path, image, spacing):
    """
    A wrapper for the various image writing functions. The consumers
    should only call this function

    :param path:    A full path to the image.
    :param image:   An image as :type image: numpy.ndarray or sitk.image.
    :param spacing: Pixel size ZXY, as a :type spacing: list.

    :return:
    """
    if path.endswith(('.mha', '.mhd')):
        __itk_image(path, image, spacing)
    elif path.endswith(('.tiff', '.tif')):
        __tiff(path, image, spacing)


def __itk_image(path, image, spacing):
    """
    A writer for ITK supported image formats.

    :param path:    A full path to the image.
    :param image:   An image as :type image: numpy.ndarray.
    :param spacing: Pixel size ZXY, as a :type spacing: list.
    """
    sitk.ImageFileWriter()
    if not isinstance(image, sitk.Image):
        image = itkutils.convert_from_numpy(image, spacing)
    image.SetSpacing(spacing)
    sitk.WriteImage(path, image)


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
    tiffile.imsave(path,
                   image,
                   resolution=(1.0/spacing[1], 1.0/spacing[2]),
                   metadata={'spacing': 1.0/spacing[0], 'unit': 'um'})