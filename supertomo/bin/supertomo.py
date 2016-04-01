#!/usr/bin/env python
# -*- python -*-

"""
fusion_main.py

Copyright (c) 2014 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the SuperTomo fusion calculation
"""
import os

### START UPDATE SYS.PATH ###
### END UPDATE SYS.PATH ###

import optparse
import sys
import datetime
import warnings

from iocbio.microscope import registration
from iocbio.io import image_stack, itkio
from iocbio.microscope import script_options as scriptopts
from iocbio.microscope import fusion, plots, itkutils


def check_necessary_inputs(options):

    if not (options.registration or options.fusion or options.transform):
        print "No operation selected. You must specify " \
              "--register and/or --fuse or --transform"
        return False

    if options.fixed_image_path is None:
        print "Fixed image not specified"
        return False

    if options.moving_image_path is None:
        print "Moving image not specified"
        return False

    if options.fusion and options.psf_path is None:
        print "PSF file not specified"
        return False

    return True

def check_path(path, prefix):
    if not os.path.isfile(path):
        path = os.path.join(prefix, path)
        if not os.path.isfile(path):
                print 'Not a valid file %s' % path
                return None
    return path

def main():
    """Main program """
    parser = optparse.OptionParser()
    scriptopts.set_fusion_options(parser)
    options, args = parser.parse_args()
    image_type = options.image_type


    # SETUP
    ##########################################################################

    # Check that all the necessary inputs are given
    if not check_necessary_inputs(options):
        sys.exit(1)

    # Check that fixed and moving images exist
    fixed_path = check_path(options.fixed_image_path, options.path_prefix)
    moving_path = check_path(options.moving_image_path, options.path_prefix)

    if (fixed_path is None) or (moving_path is None):
        sys.exit(1)

    # Load input images
    fixed_image = itkio.read_image(
        fixed_path,
        image_type
    )
    moving_image = itkio.read_image(
        moving_path,
        image_type
    )

    if options.fusion:
        psf_path = check_path(options.psf_path, options.path_prefix)
        if psf_path is None:
            sys.exit(1)
        psf = itkio.read_image(psf_path)

    # Output directory name will be automatically formatted according
    # to current date and time; e.g. 2014-02-18 iocbio_output
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d")+'_iocbio_output'
    output_dir = os.path.join(options.path_prefix, output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = None
     # PRE-PROCESSING
    ##########################################################################
    # Save originals for possible later use
    moving_original = moving_image
    fixed_original = fixed_image

    if options.dilation_size != 0:
        print 'Degrading input images with Dilation filter'
        fixed_image = itkutils.grayscale_dilate_filter(
            fixed_image,
            image_type,
            options.dilation_size
        )
        moving_image = itkutils.grayscale_dilate_filter(
            moving_image,
            image_type,
            options.dilation_size
        )

    if options.median_size != 0:
        print 'Smoothing input images with a Median filter'
        fixed_image = itkutils.median_filter(
            fixed_image,
            image_type,
            options.median_size
        )
        moving_image = itkutils.median_filter(
            moving_image,
            image_type,
            options.median_size
        )

    if options.gaussian_variance != 0.0:
        print 'Degrading input images with Gaussian blur filter'
        fixed_image = itkutils.gaussian_blurring_filter(
            fixed_image,
            image_type,
            options.gaussian_variance
        )
        moving_image = itkutils.gaussian_blurring_filter(
            moving_image,
            image_type,
            options.gaussian_variance
        )
    if options.mean_kernel != 0:
        print 'Smoothing input images with a uniform Mean filter'
        fixed_image = itkutils.mean_filter(
            fixed_image,
            image_type,
            options.mean_kernel
        )
        moving_image = itkutils.mean_filter(
            moving_image,
            image_type,
            options.mean_kernel
        )

    if options.use_internal_type or options.normalize:
         # Cast to internal type
        if '2' in options.image_type:
            image_type = options.internal_type+'2'
        else:
            image_type = options.internal_type+'3'

        fixed_image = itkutils.type_cast(
            fixed_image,
            options.image_type,
            image_type
        )
        moving_image = itkutils.type_cast(
            moving_image,
            options.image_type,
            image_type
        )


    if options.normalize:
        print 'Normalizing images'

        # Normalize
        fixed_image = itkutils.normalize_image_filter(
            fixed_image,
            image_type
        )
        moving_image = itkutils.normalize_image_filter(
            moving_image,
            image_type
        )


    if options.rescale_to_full_range:
        fixed_image = itkutils.rescale_intensity(
            fixed_image,
            image_type,
            image_type
        )
        moving_image = itkutils.rescale_intensity(
            moving_image,
            image_type,
            image_type
        )

    # REGISTRATION
    ##########################################################################
    if options.registration:

        final_transform = registration.itk_registration_rigid_3d(
            fixed_image,
            moving_image,
            options
        )

        if options.two_step_registration is True:
            options.registration_method = 'least_squares'
            final_transform = registration.itk_registration_rigid_3d(
                fixed_original,
                moving_original,
                options,
                initial_transform=final_transform
            )

        moving_image = itkutils.resample_image(
            moving_original,
            final_transform,
            image_type,
            reference=fixed_image
        )
        # Files are named according to current time (the date will be
        # in the folder name)

        date_now = datetime.datetime.now().strftime("%H-%M-%S")
        file_name = date_now + '_transform' + '.txt'
        file_name = os.path.join(output_dir, file_name)
        itkio.write_transform(file_name, final_transform)

        file_name = date_now + \
                    '_registration' + \
                    '_' + \
                    options.registration_method + \
                    '.mhd'
        file_name = os.path.join(output_dir, file_name)
        itkio.write_image(
            moving_image,
            file_name,
            image_type,
            output_type=options.image_type
        )

        if options.show_image:
            plots.plot_3d_volume(moving_image,
                                 image_type=image_type,
                                 plot_method=options.rendering_method)

    # TRANSFORM
    ##########################################################################
    if options.transform:

        print "Transforming image"

        transform_path = options.transform_path

        if not os.path.isfile(transform_path):
            transform_path = os.path.join(options.path_prefix, transform_path)
            if not os.path.isfile(transform_path):
                print 'Not a valid file %s' % options.transform_path
                sys.exit(1)

        transform = itkio.read_transform(transform_path)

        moving_image = itkutils.resample_image(
            moving_image,
            transform,
            image_type,
            fixed_image
        )

        file_name = datetime.datetime.now().strftime("%H-%M-%S") + \
                    '_transformed' + \
                    '.mhd'
        file_name = os.path.join(output_dir, file_name)

        itkio.write_image(
            moving_image,
            file_name,
            image_type,
            output_type=options.image_type
        )

    # FUSION
    ##########################################################################
    if options.fusion:
        if options.psf_type is 'single':
            if transform is None:
                transform_path = check_path(options.transform_path,
                                            options.path_prefix)
                if transform_path is None:
                    sys.exit(1)
                transform = itkio.read_transform(transform_path)

            psf2 = itkutils.rotate_psf(psf, transform, image_type,
                                       return_image_stack=True,
                                       convert_to_itk=False)
            psf_stacks = [
                image_stack.ImageStack.import_from_itk(psf, image_type),
                psf2
            ]
            del psf
            del psf2
        else:
            print "Only Single PSF option has been implemented for the moment"
            sys.exit(1)

        if options.subset_image != 1.0:
            fixed_image = itkutils.get_image_subset(
                fixed_image, image_type, options.subset_image)

            moving_image = itkutils.get_image_subset(
                moving_image, image_type, options.subset_image)

        image_stacks = [
            image_stack.ImageStack.import_from_itk(fixed_image, image_type),
            image_stack.ImageStack.import_from_itk(moving_image, image_type)
        ]

        del fixed_image
        del moving_image
        del fixed_original
        del moving_original

        print "Starting fusion"
        task = fusion.MultiViewFusionRL(psf_stacks, image_stacks, options)

        del psf_stacks
        del image_stacks

        result = task.deconvolve()

        if options.show_image:
            plots.plot_3d_volume(result, plot_method=options.rendering_method)

        file_name = datetime.datetime.now().strftime("%H-%M-%S") + \
                    '_fusion' + \
                    '.tif'
        file_name = os.path.join(output_dir, file_name)
        result.save(file_name)
        print 'Saved fusion result to %s' % file_name

if __name__ == "__main__":
    main()










