import os

import SimpleITK as sitk

from supertomo.definitions import *
import tiffile
from ..utils import itkutils

scale_c = 1.0e6


def get_imagej_tiff(filename, memmap=False, return_itk=False):
    """
    ImageJ has a bit peculiar way of saving image metadata, especially the tags
    for voxel spacing, which is of main interest in SuperTomo. This function reads
    a 3D TIFF into a Numpy array and also calculates the voxel spacing parameters
    from the TIFF tags. I will not guarantee that this will work with any other TIFF
    files.

    :param filename:    Path to a TIFF.
    :param memmap:      Enables Memory mapping in case the TIFF file is too large to
                        be read in memory completely.
    :param return_itk  Converts the Image data into a sitk.Image. This can be used
                        when single images are needed, instead of using the HDF5
                        structure adopted in SuperTomo2.
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
        for tag in page.tags.values():
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


def get_itk_image(filename, return_itk = True):
    """
    A function for reading image file types typical to ITK (mha & mhd). This is mostly
    of a historical significance, because in the original SuperTomo 1 such files were
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
        return itkutils.convert_to_numpy(image)


def get_image(filename, return_itk=False):
    """
    A wrapper for the functions above.
    Parameters
    ----------
    filename
    return_itk

    Returns
    -------

    """
    if filename.endswith((".tif", ".tiff")):
        return get_imagej_tiff(filename, return_itk)
    elif filename.endswith((".mha", ".mhd")):
        return get_itk_image(filename, return_itk)
    else:
        raise ValueError("No image reader specified for "
                         "%s" % filename)


def read_itk_transform(path, return_itk=False):
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


def write_imagej_tiff(path, image, spacing):
    tiffile.imsave(path,
                   image,
                   imagej=True,
                   resolution=list(1.0/x for x in spacing))