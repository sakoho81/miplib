#!/usr/bin/env python

import datetime
import os
import sys

import SimpleITK as sitk

from miplib.data.io import read as ioutils
from miplib.ui import supertomo_options
from miplib.processing import itk as itkutils


def main():
    options = supertomo_options.get_transform_script_options(sys.argv[1:])
    fixed_image = None
    moving_image = None
    transform = None

    if options.hdf:
        raise NotImplementedError("Only single image files are supported "
                                  "currently")
    else:
        # CHECK FILES
        # Check that the fixed image exists
        fixed_image_path = os.path.join(
            options.working_directory,
            options.fixed_image
        )

        if not os.path.isfile(fixed_image_path):
            raise ValueError('No such file: %s' % options.fixed_image)

        # Check that the EM image exists
        moving_image_path = os.path.join(
            options.working_directory,
            options.moving_image)

        if not os.path.isfile(moving_image_path):
            raise ValueError('No such file: %s' % options.moving_image)

        transform_path = os.path.join(
            options.working_directory,
            options.transform)

        if not os.path.isfile(transform_path):
            raise ValueError('No such file: %s' % options.transform)

        # READ FILES
        fixed_image = ioutils.get_image(fixed_image_path, return_itk=True)
        moving_image = ioutils.get_image(moving_image_path, return_itk=True)
        transform = ioutils.__itk_transform(transform_path, return_itk=True)

        transformed_image = itkutils.resample_image(moving_image,
                                                    transform,
                                                    fixed_image)

        # OUTPUT
        ##########################################################################
        output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + '_supertomo_out'
        output_dir = os.path.join(options.working_directory, output_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Files are named according to current time (the date will be
        # in the folder name)
        date_now = datetime.datetime.now().strftime("%H-%M-%S")

        file_name = date_now + '_transformed.mha'

        rgb_image = itkutils.make_composite_rgb_image(fixed_image,
                                                               transformed_image)

        image_path = os.path.join(output_dir, file_name)
        sitk.WriteImage(rgb_image, image_path)


if __name__ == '__main__':
    main()