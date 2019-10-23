#!/usr/bin/env python

"""
fusion_main.py

Copyright (c) 2014 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.
"""

import datetime
import os
import sys

import SimpleITK as sitk

from miplib.processing.registration import registration
from miplib.ui.cli import miplib_entry_point_options
from miplib.processing import itk as itkutils


def main():
    options = miplib_entry_point_options.get_correlate_tem_script_options(sys.argv[1:])
    
    # SETUP
    ##########################################################################

    # Check that the STED image exists
    options.sted_image_path = os.path.join(options.working_directory,
                                         options.sted_image_path)
    if not os.path.isfile(options.sted_image_path):
        print('No such file: %s' % options.sted_image_path)
        sys.exit(1)

    # Check that the EM image exists
    options.em_image_path = os.path.join(options.working_directory,
                                           options.em_image_path)
    if not os.path.isfile(options.em_image_path):
        print('No such file: %s' % options.em_image_path)
        sys.exit(1)


        
    # Load input images
    sted_image = sitk.ReadImage(options.sted_image_path)
    em_image = sitk.ReadImage(options.em_image_path)


    # PRE-PROCESSING
    ##########################################################################
    # Save originals for possible later use
    sted_original = sted_image
    em_original = em_image

    if options.dilation_size != 0:
        print('Degrading input images with Dilation filter')
        sted_image = itkutils.grayscale_dilate_filter(
            sted_image,
            options.dilation_size
        )
        em_image = itkutils.grayscale_dilate_filter(
            em_image,
            options.dilation_size
        )

    if options.gaussian_variance != 0.0:
        print('Degrading the EM image with Gaussian blur filter')

        em_image = itkutils.gaussian_blurring_filter(
            em_image,
            options.gaussian_variance
        )
    if options.mean_kernel != 0:
        sted_image = itkutils.mean_filter(
            sted_image,
            options.mean_kernel
        )
        em_image = itkutils.mean_filter(
            em_image,
            options.mean_kernel
        )
    #todo: convert the pixel type into a PixelID enum
    if options.use_internal_type:

        sted_image = itkutils.type_cast(
            sted_image,
            options.image_type
        )
        em_image = itkutils.type_cast(
            em_image,
            options.image_type
         )
    #
    # if options.threshold > 0:
    #     sted_image = itkutils.threshold_image_filter(
    #         sted_image,
    #         options.threshold
    #     )
    #
    #     em_image = itkutils.threshold_image_filter(
    #         em_image,
    #         options.threshold
    #     )

    if options.normalize:
        print('Normalizing images')

        # Normalize
        sted_image = itkutils.normalize_image_filter(sted_image)
        em_image = itkutils.normalize_image_filter(em_image)

        if options.rescale_to_full_range:
            sted_image = itkutils.rescale_intensity(sted_image)
            em_image = itkutils.rescale_intensity(em_image)


    # REGISTRATION
    ##########################################################################

    if options.tfm_type == "rigid":
        final_transform = registration.itk_registration_rigid_2d(sted_image, em_image, options)
    elif options.tfm_type == "similarity":
        final_transform = registration.itk_registration_similarity_2d(sted_image, em_image, options)
    else:
        raise ValueError(options.tfm_type)
    em_image = itkutils.resample_image(
        em_original,
        final_transform,
        reference=sted_image
    )




    # OUTPUT
    ##########################################################################

    while True:
        keep = input("Do you want to keep the results (yes/no)? ")
        if keep in ('y', 'Y', 'yes', 'YES'):
            # Files are named according to current time (the date will be
            # in the folder name)

            # Output directory name will be automatically formatted according
            # to current date and time; e.g. 2014-02-18_supertomo_output
            output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + '_clem_output'
            output_dir = os.path.join(options.working_directory, output_dir)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            date_now = datetime.datetime.now().strftime("%H-%M-%S")
            file_name = date_now + \
                        '-clem_registration-' + \
                        options.registration_method + \
                        '.tiff'
            file_path = os.path.join(output_dir, file_name)
            tfm_name = date_now + '_transform' + '.txt'
            tfm_path = os.path.join(output_dir, tfm_name)
            sitk.WriteTransform(final_transform, tfm_path)

            rgb_image = itkutils.make_composite_rgb_image(sted_original, em_image)
            sitk.WriteImage(rgb_image, file_path)
            print("The image was saved to %s and the transform to %s in " \
                  "the output directory %s" % (file_name, tfm_name, output_dir))
            break
        elif keep in ('n', 'N', 'no', 'No'):
            print("Exiting without saving results.")
            break
        else:
            print("Unkown command. Please state yes or no")


if __name__ == "__main__":
    main()





