import os

import SimpleITK as sitk
import pims
import numpy as np

import miplib.processing.itk as itkutils
from . import tiffile
from miplib.data.containers.image import Image
from miplib.data.containers.array_detector_data import ArrayDetectorData

scale_c = 1.0e6


def get_image(filename, series=0, channel=0, return_type='image', bioformats=True):
    """
    A wrapper for the image read functions.
    Parameters
    :param series: The index of an image in a time series
    :param channel: The color channel in  a multi-channel image.
    :param bioformats: Toggle to disable bioformats reader.
    :param filename The full path to an image
    :param return_type Return the image as miplib Image. sitk.Image.
           the return type can be chosen with a string ('image, 'itk').

    """
    assert return_type in ('itk', 'image')

    if filename.endswith(".mha"):
        data = __itk_image(filename, return_type == 'itk')
    else:
        if bioformats:
            data = __bioformats(filename, series, channel, return_type == 'itk')
        else:
            data = __tiff(filename, return_type == 'itk')


    return data

def __itk_image(filename, return_itk=True):
    """
    A function for reading image file types typical to ITK (mha & mhd). This is mostly
    of a historical significance, because in the original miplib 1 such files were
    used, mostly for convenience.

    :param filename:     Path to an ITK image
    :param return_itk    Toggle whether to convert the ITK image into Numpy format
    :return:             Image data as a Numpy array, voxel spacing tuple
    """
    assert filename.endswith((".mha", ".mhd"))
    image = sitk.ReadImage(filename)
    if return_itk:
        return image
    else:
        return itkutils.convert_from_itk_image(image)


def __tiff(filename, memmap=False, return_itk=False):
    """
    ImageJ has a bit peculiar way of saving image metadata, especially the tags
    for voxel spacing, which is of main interest in miplib. This function reads
    a 3D TIFF into a Numpy array and also calculates the voxel spacing parameters
    from the TIFF tags. I will not guarantee that this will work with any other TIFF
    files.

    :param filename:    Path to a TIFF.
    :param memmap:      Enables Memory mapping in case the TIFF file is too large to
                        be read in memory completely.
    :param return_itk  Converts the Image data into a sitk.Image. This can be used
                        when single images are needed, instead of using the HDF5
                        structure adopted in miplib.
    :return:            Image data either as a Numpy array, voxel spacing tuple or a
                        sitk.Image
    """
    assert filename.endswith((".tif", ".tiff"))
    tags = {}
    # Read images and tags
    with tiffile.TiffFile(filename) as image:
        # Get images
        images = image.asarray(memmap=memmap)
        # Get tags
        page = image[0]
        for tag in list(page.tags.values()):
            tags[tag.name] = tag.value

    # Figure out z-spacing, which in ImageJ is hidden in the "image_description"
    # header (why, one might ask).
    image_descriptor = tags["image_description"].split("\n")
    z_spacing = None
    for line in image_descriptor:
        if "spacing" in line:
            z_spacing = float(line.split("=")[-1])
            break
    assert z_spacing is not None

    # Create a tuple for zxy-spacing. The order of the dimensions follows that of the
    # image data
    spacing = (z_spacing, scale_c/tags["x_resolution"][0], scale_c/tags[
        "y_resolution"][0])

    #print spacing
    if return_itk:
        return itkutils.convert_from_numpy(images, spacing)
    else:
        return images, spacing


def __itk_transform(path, return_itk=False):
    """
    Prior to starting to use the HDF5 format data storage images and spatial
    transforms were saved as separate image files on the hard drive. This
    function can be used to read a spatial transform saved from ITK. It is
    to transfer old files into the HDF5 format storage.

    Parameters
    ----------
    path        Path to the transform file (usually txt ended)

    Returns     Returns the transform type integer, parameters and fixed
                parameters.
    -------

    """

    if not os.path.isfile(path):
        raise ValueError("Not a valid path: %s" % path)

    transform = sitk.ReadTransform(path)

    if return_itk:
        return transform

    else:
        # #TODO: Check that this makes any sense. Also consult the ITK HDF implementation for ideas
        # with open(path, 'r') as f:
        #     for line in f:
        #         if line.startswith('Transform:'):
        #             type_string = line.split(': ')[1].split('_')[0]
        #             if "VersorRigid" in type_string:
        #                 transform_type = itk_transforms_c['sitkVersorRigid']
        #                 break
        #             else:
        #                 raise NotImplementedError("Unknown transform type: "
        #                                           "%s" % type_string)
        transform_type = transform.GetName()
        params = transform.GetParameters()
        fixed_params = transform.GetFixedParameters()
        return transform_type, params, fixed_params


def __bioformats(filename, series=0, channel=0, return_itk = False):
    """
    Read an image using the Bioformats importer. Good for most microscopy formats.

    :param filename:
    :param series:
    :param return_itk:
    :return:
    """
    assert pims.bioformats.available(), "Please install jpype in order to use " \
                                        "the bioformats reader."
    image = pims.bioformats.BioformatsReader(filename, series=series)

    # Get Pixel/Voxel size information
    if 'z' not in image.axes:
        spacing = (image.metadata.PixelsPhysicalSizeY(0),
                   image.metadata.PixelsPhysicalSizeX(0))
    else:
        spacing = (image.metadata.PixelsPhysicalSizeZ(0),
                   image.metadata.PixelsPhysicalSizeY(0),
                   image.metadata.PixelsPhysicalSizeX(0))

    # Get color channel
    if 'c' in image.sizes:
        image.iter_axes = 'c'
        assert len(image) > channel
        image = image[channel]
    else:
        image = image[0]

    if return_itk:
        return itkutils.convert_from_numpy(image, spacing)
    else:
        return Image(image, spacing)


