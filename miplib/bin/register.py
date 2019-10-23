#!/usr/bin/env python
# -*- python -*-

"""
register.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the miplib fusion calculation
"""

import os
import sys

import SimpleITK as sitk

from miplib.data.containers import image_data
from miplib.processing import itk as itkutils
from miplib.processing.registration import registration_mv
from miplib.ui import utils
from miplib.ui.cli import miplib_entry_point_options


def main():
    options = miplib_entry_point_options.get_register_script_options(sys.argv[1:])

    # Get Data
    filename = sys.argv[1]
    if not filename.endswith(".hdf5"):
        raise ValueError("Please specify a HDF5 data file")
    full_path = os.path.join(options.working_directory, filename)
    if not os.path.exists(full_path):
        raise ValueError("The specified file %s does not exist" % full_path)

    data = image_data.ImageData(full_path)

    # Check that requested image size exists. If not, create it.
    if options.scale not in data.get_scales("original"):
        print("Images at the defined scale do not exist in the data " \
              "structure. The original images will be now resampled. " \
              "This may take a long time depending on the image size " \
              "and the number of views.")
        data.create_rescaled_images("original", options.scale)

    data.set_active_image(0, options.channel, options.scale, "original")
    spacing = data.get_voxel_size()
    data.add_registered_image(data[:], options.scale, 0, options.channel,
                              0, spacing)

    if options.evaluate_results:
        fixed_image = data.get_itk_image()

    # Setup registration. View number 0 is always the reference for now.
    # The behavior can be easily changed if necessary.
    task = registration_mv.RotatedMultiViewRegistration(data, options)
    task.set_fixed_image(0)

    # Iterate over the rotated views
    for view in range(1, data.get_number_of_images("original")):
        task.set_moving_image(view)
        if data.check_if_exists("registered", view, options.channel,
                                options.scale):
            if utils.get_user_input("There is a saved registration result for "
                                    "the view %i. Do you want to skip it?" %
                                    view):
                continue

        task.execute()

        if options.evaluate_results:
            moving_image = task.get_resampled_result()
            sitk.Show(
                itkutils.make_composite_rgb_image(fixed_image, moving_image))
            if utils.get_user_input(
                    "Do you want to save the result (yes/no)? "):
                task.save_result()
                continue
            else:
                if utils.get_user_input(
                        "Skipping view %i. Do you want to continue "
                        "registration?" % view):
                    continue
                else:
                    print("Exiting registration without saving results")
                    break
        else:
            task.save_result()
            continue

    data.close()


if __name__ == "__main__":
    main()
