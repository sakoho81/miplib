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
import os
import sys

import numpy

from miplib.data.containers import image_data
from miplib.data.definitions import *
from miplib.data.io import read
from miplib.processing import itk as itkutils
from ..ui.cli import miplib_entry_point_options 


def main():
    options = miplib_entry_point_options.get_import_script_options(sys.argv[1:])
    directory = options.data_dir_path

    # Create a new HDF5 file. If a file exists, new data will be appended.
    file_name = input("Give a name for the HDF5 file: ")
    file_name += ".hdf5"
    data_path = os.path.join(directory, file_name)
    data = image_data.ImageData(data_path)

    # Add image files that have been named according to the correct format
    for image_name in os.listdir(directory):
        full_path = os.path.join(directory, image_name)

        if full_path.endswith((".tiff", ".tif", ".mhd", ".mha")):
            images = read.get_image(full_path)
            spacing = images.spacing
        else:
            continue

        if options.normalize_inputs:
            images = (images * (255.0/images.max())).astype(numpy.uint8)

        if not all(x in image_name for x in params_c) or not any(x in image_name for x in image_types_c):
            print("Unrecognized image name %s. Skipping it." % image_name)
            continue

        image_type = image_name.split("_scale")[0]
        scale = image_name.split("scale_")[-1].split("_index")[0]
        index = image_name.split("index_")[-1].split("_channel")[0]
        channel = image_name.split("channel_")[-1].split("_angle")[0]
        angle = image_name.split("angle_")[-1].split(".")[0]

        assert all(x.isdigit() for x in (scale, index, channel, angle))
        # data, angle, spacing, index, scale, channel, chunk_size=None

        if image_type == "original":
            data.add_original_image(images, scale, index, channel, angle, spacing)
        elif image_type == "registered":
            data.add_registered_image(images, scale, index, channel, angle, spacing)
        elif image_type == "psf":
            data.add_psf(images, scale, index, channel, angle, spacing)

    # Calculate resampled images
    if options.scales is not None:
        for scale in options.scales:
            print("Creating %s percent downsampled versions of the original images" % scale)
            data.create_rescaled_images("original", scale)

    # Add transforms for registered images.
    for transform_name in os.listdir(directory):
        if not transform_name.endswith(".txt"):
            continue

        if not all(x in transform_name for x in params_c) or not "transform" in transform_name:
            print("Unrecognized transform name %s. Skipping it." % transform_name)
            continue

        scale = transform_name.split("scale_")[-1].split("_index")[0]
        index = transform_name.split("index_")[-1].split("_channel")[0]
        channel = transform_name.split("channel_")[-1].split("_angle")[0]
        angle = transform_name.split("angle_")[-1].split(".")[0]

        full_path = os.path.join(directory, transform_name)

        # First calculate registered image if not in the data structure
        if not data.check_if_exists("registered", index, channel, scale):
            print("Resampling registered image for image nr. ", index)
            data.set_active_image(0, channel, scale, "original")
            reference = data.get_itk_image()
            data.set_active_image(index, channel, scale, "original")
            moving = data.get_itk_image()

            transform = read.__itk_transform(full_path, return_itk=True)

            registered = itkutils.resample_image(moving, transform, reference=reference)
            registered = itkutils.convert_from_itk_image(registered)
            spacing = registered.spacing

            data.add_registered_image(registered, scale, index, channel, angle, spacing)

        # The add it's transform
        transform_type, params, fixed_params = read.__itk_transform(full_path)
        data.add_transform(scale, index, channel, params, fixed_params, transform_type)

    # Calculate missing PSFs
    if options.calculate_psfs:
        data.calculate_missing_psfs()

    if options.copy_registration_result != -1:
        from_scale = options.copy_registration_result[0]
        to_scale = options.copy_registration_result[1]
        data.copy_registration_result(from_scale, to_scale)

    data.close()


if __name__ == "__main__":
    main()