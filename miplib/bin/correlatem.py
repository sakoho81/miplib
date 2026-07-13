#!/usr/bin/env python

"""Correlative light-electron microscopy registration CLI."""

import datetime
import os
import sys

import SimpleITK as sitk

from miplib.processing import itk as itkutils
from miplib.processing.itk import convert_from_itk_image
from miplib.processing.registration import RegistrationMethod, register
from miplib.ui.cli import miplib_entry_point_options


def main():
    options = miplib_entry_point_options.get_correlate_tem_script_options(sys.argv[1:])

    options.sted_image_path = os.path.join(
        options.working_directory, options.sted_image_path
    )
    if not os.path.isfile(options.sted_image_path):
        sys.exit(f"No such file: {options.sted_image_path}")

    options.em_image_path = os.path.join(
        options.working_directory, options.em_image_path
    )
    if not os.path.isfile(options.em_image_path):
        sys.exit(f"No such file: {options.em_image_path}")

    sted_image = sitk.ReadImage(options.sted_image_path)
    em_image = sitk.ReadImage(options.em_image_path)

    sted_original = sted_image
    em_original = em_image

    if options.dilation_size != 0:
        print("Degrading input images with Dilation filter")
        sted_image = itkutils.grayscale_dilate_filter(sted_image, options.dilation_size)
        em_image = itkutils.grayscale_dilate_filter(em_image, options.dilation_size)

    if options.gaussian_variance != 0.0:
        print("Degrading the EM image with Gaussian blur filter")
        em_image = itkutils.gaussian_blurring_filter(
            em_image, options.gaussian_variance
        )

    if options.mean_kernel != 0:
        sted_image = itkutils.mean_filter(sted_image, options.mean_kernel)
        em_image = itkutils.mean_filter(em_image, options.mean_kernel)

    if options.use_internal_type:
        sted_image = itkutils.type_cast(sted_image, options.image_type)
        em_image = itkutils.type_cast(em_image, options.image_type)

    if options.normalize:
        print("Normalizing images")
        sted_image = itkutils.normalize_image_filter(sted_image)
        em_image = itkutils.normalize_image_filter(em_image)
        if options.rescale_to_full_range:
            sted_image = itkutils.rescale_intensity(sted_image)
            em_image = itkutils.rescale_intensity(em_image)

    fixed_img = convert_from_itk_image(sted_image)
    moving_img = convert_from_itk_image(em_image)

    method = RegistrationMethod.from_string(options.tfm_type)
    transform = register(fixed_img, moving_img, method)

    em_registered = itkutils.resample_image(
        em_original, transform, reference=sted_image
    )

    output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + "_clem_output"
    output_dir = os.path.join(options.working_directory, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date_now = datetime.datetime.now().strftime("%H-%M-%S")
    file_name = f"{date_now}-clem_registration-{options.registration_method}.tiff"
    file_path = os.path.join(output_dir, file_name)
    tfm_name = f"{date_now}_transform.txt"
    tfm_path = os.path.join(output_dir, tfm_name)
    sitk.WriteTransform(transform, tfm_path)

    rgb = itkutils.make_composite_rgb_image(sted_original, em_registered)
    sitk.WriteImage(rgb, file_path)
    print(f"Saved {file_name} and {tfm_name} to {output_dir}")


if __name__ == "__main__":
    main()
