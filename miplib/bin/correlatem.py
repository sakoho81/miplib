#!/usr/bin/env python

"""Correlative light-electron microscopy registration CLI."""

import datetime
import sys
from pathlib import Path

import SimpleITK as sitk

from miplib.processing import itk as itkutils
from miplib.processing.itk import convert_from_itk_image
from miplib.processing.registration import RegistrationMethod, register
from miplib.ui.cli import miplib_entry_point_options


def main():
    options = miplib_entry_point_options.get_correlate_tem_script_options(sys.argv[1:])
    wd = Path(options.working_directory)

    options.sted_image_path = str(wd / options.sted_image_path)
    if not Path(options.sted_image_path).is_file():
        sys.exit(f"No such file: {options.sted_image_path}")

    options.em_image_path = str(wd / options.em_image_path)
    if not Path(options.em_image_path).is_file():
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
    output_dir = wd / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    date_now = datetime.datetime.now().strftime("%H-%M-%S")
    file_name = f"{date_now}-clem_registration-{options.registration_method}.tiff"
    file_path = output_dir / file_name
    tfm_name = f"{date_now}_transform.txt"
    tfm_path = output_dir / tfm_name
    sitk.WriteTransform(transform, str(tfm_path))

    rgb = itkutils.make_composite_rgb_image(sted_original, em_registered)
    sitk.WriteImage(rgb, str(file_path))
    print(f"Saved {file_name} and {tfm_name} to {output_dir}")


if __name__ == "__main__":
    main()
