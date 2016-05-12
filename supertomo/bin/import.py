"""
A program to import data into our HDF5 archive format.
All the images should be contained within a single directory
that can be either provided as a command line parameter, or
alternatively the program will try to use the current working
directory as the source directory.

The files should be named according to the following pattern:

Original images:
===============================================================================

original_scale_<scale>_index_<index>_channel_<channel>_angle_<angle>.<suffix>

<scale> is the image size, as a percentage of the raw microscope image
<index> is the ordering index of the views. The regular STED image gets
        index 0, the first rotation index 1 etc.
<channel> the color channel index, should start from zero.
<angle> is the estimated rotation angle
<suffix> can be .tif/.tiff, .mhd/.mha


Registered images:
===============================================================================

registered_scale_<scale>_index_<index>_channel_<channel>_angle_<angle>.<suffix>

<scale> is the image size, as a percentage of the raw microscope image
<index> is the ordering index of the views. The regular STED image gets
        index 0, the first rotation index 1 etc.
<channel> the color channel index, should start from zero.
<angle> is the estimated rotation angle
<suffix> can be .tif/.tiff, .mhd/.mha

Transform files:
===============================================================================
transform_scale_<scale>_index_<index>_channel_<channel>_angle_<angle>.<suffix>

The <scale>, <index> and <angle> parameters correspond to the registered
image that the transform is coupled with.

PSF images:
===============================================================================

psf_scale_<scale>_index_<index>_channel_<channel>_angle_<angle>.<suffix>

<scale> is the image size, as a percentage of the raw microscope image
<index> is the ordering index of the views. The regular STED image gets
        index 0, the first rotation index 1 etc.
<channel> the color channel index, should start from zero.
<angle> is the estimated rotation angle
<suffix> can be .tif/.tiff, .mhd/.mha


"""
import sys
import os
from ..io import utils, image_data
from ..definitions import *


def main():
    # The user can give a directory from the command line or alternatively
    # the image are expected to reside in the current working directory.
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        assert os.path.isdir(directory)
    else:
        directory = os.getcwd()

    # Create a new HDF5 file. If a file exists, new data will be appended.
    file_name = raw_input("Give a name for the HDF5 file: ")
    file_name += ".hdf5"
    data_path = os.path.join(directory, file_name)
    data = image_data.ImageData(data_path)

    # Add image files that have been named according to the correct format
    for image_name in os.listdir(directory):
        full_path = os.path.join(directory, image_name)

        if full_path.endswith((".tiff", ".tif")):
            images, spacing = utils.get_imagej_tiff(full_path)
        elif full_path.endswith((".mhd", ".mha")):
            images, spacing = utils.get_itk_image(full_path)
        else:
            continue

        if not all(x in image_name for x in params_c) or not any(x in image_name for x in image_types_c):
            print "Unrecognized image name %s. Skipping it." % image_name
            continue

        image_type = image_name.split("_scale")[0]
        scale = image_name.split("scale_")[-1].split("_index")[0]
        index = image_name.split("index_")[-1].split("_channel")[0]
        channel = image_name.split("channel_")[-1].split("_angle")[0]
        angle = image_name.split("angle_")[-1].split(".")[0]

        assert all(x.isdigit() for x in (scale, index, channel, angle))
        # data, angle, spacing, index, scale, channel, chunk_size=None

        if image_type == "original":
            data.add_original_image(images, angle, spacing, index, scale, channel)
        elif image_type == "registered":
            data.add_registered_image(images, None, None, channel, index, scale, spacing)
        elif image_type == "psf":
            data.add_psf(images, angle, channel, index, scale, spacing)

    # Add transforms for registered images.
    for transform_name in os.listdir(directory):
        if not transform_name.endswith(".txt"):
            continue

        if not all(x in transform_name for x in params_c) or not "transform" in transform_name:
            print "Unrecognized transform name %s. Skipping it." % transform_name
            continue

        scale = image_name.split("scale_")[-1].split("_index")[0]
        index = image_name.split("index_")[-1].split("_channel")[0]
        channel = image_name.split("channel_")[-1].split("_angle")[0]

        full_path = os.path.join(directory, transform_name)
        transform_type, params, fixed_params = utils.read_itk_transform(full_path)
        data.add_transform(scale, index, channel, params, fixed_params, transform_type)

    data.close()


if __name__ == "__main__":
    main()