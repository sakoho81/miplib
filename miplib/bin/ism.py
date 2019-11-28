"""
A command line script for batch ISM image reconstruction.
You can use it by calling:

 § miplib.ism all <directory>

Where directory should contain the data files. Zeiss AiryScan data
or our Carma images are supported. The "all" parameter can take four
different values:

ism -- adative pixel reassignment only
wiener -- adaptive pixel reassignment and wiener filtering
rl -- adaptive pixel reassignment an RL deconvolution
all -- adaptive pixel reassignment and both wiener and RL

Please refer to miplib.ism --help for all the possible command line arguments.
In a typical case you should not need to touch them though.

"""

import os
import sys

import numpy as np

import miplib.processing.image as imops
import miplib.ui.cli.miplib_entry_point_options as options
import miplib.ui.plots.scatter as scatter
from miplib.data.containers.image import Image
from miplib.data.io import array_detector_data
from miplib.data.io.write import image as imwrite
from miplib.data.messages import image_writer_wrappers as imwrap
from miplib.processing.deconvolution import wiener, deconvolve
from miplib.processing.ism import reconstruction as ismrec
from miplib.processing.segmentation import masking
from miplib.psf import frc_psf


def main():
    args = options.get_ism_script_options(sys.argv[1:])

    # Only allow translation transform, as rotation is not needed here, 
    args.reg_translate_only = True

    # Get Carma mat files in the working direcory
    root_dir = args.directory
    files = tuple(f for f in os.listdir(root_dir) if f.endswith(".mat") or f.endswith(".czi"))

    # Generate ISM result for each file
    for filename in files:

        full_path = os.path.join(root_dir, filename)

        if full_path.endswith(".mat"):
            detector_type = "IIT_SPAD"
            filename_prefix = filename.split(".mat")[0]
            data = array_detector_data.read_carma_mat(full_path)
        elif full_path.endswith(".czi"):
            filename_prefix = filename.split(".czi")[0]
            data = array_detector_data.read_airyscan_data(full_path)
            detector_type = "AIRYSCAN"
        else:
            print("Unknown datatype {}".format(full_path.split(".")[-1]))
            exit(-1)

        print("Opened a {} file {} for ISM reconstruction".format(
            "Carma" if filename.endswith(".mat") else "AiryScan", filename))
        print("The file contains images from {} laser gates and {} detector "
              "channels.".format(data.ngates, data.ndetectors))
        print("The image size is {} and the pixel spacing {} um".format(
            data[0, 0].shape, data[0, 0].spacing))

        print("Calculating image shifts for adaptive reassignment.")
        # Run registration
        x, y, transforms = ismrec.find_image_shifts(
            data, args, fixed_idx=0 if detector_type == "AIRYSCAN" else 12)

        # Save shift scatter, if saving plots is enabled
        if args.save_plots:
            fig = scatter.xy_scatter_plot_with_labels(
                y, x, range(len(x)), size=args.plot_size)

            # from matplotlib import pyplot as plt
            #
            # plt.xlim([-.3, .3])
            # plt.ylim([-.3, .3])
            scatter_name = "{}_shifts_scatter.eps".format(filename_prefix)
            scatter_path = os.path.join(root_dir, scatter_name)
            fig.savefig(scatter_path, dpi=1200, bbox_inches='tight',
                        pad_inches=0, transparent=True)

        # Generate ISM and confocal results
        ism_result = ismrec.shift_and_sum(data, transforms)
        result_sum = ismrec.sum(data)

        ism_result_name = "{}_apr_ism.tif".format(filename_prefix)
        static_ism_result_name = "{}_static_ism.tif".format(filename_prefix)
        confocal_result_name = "{}_open.tif".format(filename_prefix)
        confocal_closed_result_name = "{}_closed.tif".format(filename_prefix)

        imwrite(os.path.join(root_dir, ism_result_name),
                imops.enhance_contrast(ism_result, percent_saturated=.3)
                if args.enhance_contrast_on_save
                else Image(ism_result.astype(np.uint16), ism_result.spacing))

        imwrite(os.path.join(root_dir, confocal_result_name),
                imops.enhance_contrast(result_sum, percent_saturated=.3)
                if args.enhance_contrast_on_save
                else Image(result_sum.astype(np.uint16), result_sum.spacing))

        imwrite(os.path.join(root_dir, confocal_closed_result_name),
                imops.enhance_contrast(data[0,12], percent_saturated=.3)
                if args.enhance_contrast_on_save
                else Image(data[0, 12].astype(np.uint16), result_sum.spacing))

        # Calculate static ISM if one so desires
        if any(args.ism_mode == mode for mode in ("static", "rl", "all")):
            print("Calculating static pixel reassignment.")
            static_ism_result = ismrec.shift_and_sum(
                data,
                ismrec.find_static_image_shifts(args.ism_spad_pitch,
                                                args.ism_wavelength,
                                                args.ism_spad_fov_au,
                                                args.ism_na,
                                                alpha=args.ism_alpha)
            )
            imwrite(os.path.join(root_dir, static_ism_result_name),
                    imops.enhance_contrast(static_ism_result, percent_saturated=.3)
                    if args.enhance_contrast_on_save
                    else Image(static_ism_result.astype(np.uint16), static_ism_result.spacing))

        # Run FRC based Wiener filter, if requested
        if any(args.ism_mode == mode for mode in ("wiener", "all")):
            print("Running the blind Wiener filter.")
            try:
                psf = frc_psf.generate_frc_based_psf(ism_result, args)
            except (IndexError, ValueError) as e:
                new_shape = tuple(int(.8 * dim) for dim in ism_result.shape)
                psf = frc_psf.generate_frc_based_psf(
                    imops.remove_zero_padding(ism_result, new_shape), args)

            wiener_result = wiener.wiener_deconvolution(
                ism_result, psf, snr=args.wiener_nsr
            )
            wiener_result_name = "{}_apr_ism_bplus_wiener.tif".format(filename_prefix)

            imwrite(os.path.join(root_dir, wiener_result_name),
                    imops.enhance_contrast(wiener_result, percent_saturated=.3)
                    if args.enhance_contrast_on_save
                    else imops.rescale_to_8_bit(wiener_result))

        # Run FRC based RL deconvolution, if requested
        if any(args.ism_mode == mode for mode in ("rl", "all")):
            print("Running the blind RL deconvolution.")

            try:
                psf = frc_psf.generate_frc_based_psf(ism_result, args)
            except (IndexError, ValueError) as e:
                new_shape = tuple(int(.8 * dim) for dim in ism_result.shape)
                psf = frc_psf.generate_frc_based_psf(
                    imops.remove_zero_padding(ism_result, new_shape), args)

            # Enable automatic background correction with --rl-auto-background
            if args.rl_auto_background:
                background_mask = masking.make_local_intensity_based_mask(
                    ism_result, threshold=30, kernel_size=60, invert=True)
                masked_image = Image(ism_result * background_mask, ism_result.spacing)
                args.rl_background = np.mean(masked_image[masked_image > 0])

            # If saving intermediate results, a temp directory is created here
            if args.save_intermediate_results:
                temp_dir = os.path.join(root_dir, "{}_temp".format(filename_prefix))
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)
                writer = imwrap.TiffImageWriter(temp_dir)
            else:
                writer = None

            task = deconvolve.DeconvolutionRL(ism_result, psf, writer, args)
            task.execute()
            rl_result = task.get_8bit_result()

            rl_results_name = "{}_apr_ism_bplus_rl_{}.tif".format(
                filename_prefix, task.iteration_count)

            imwrite(os.path.join(root_dir, rl_results_name),
                    imops.enhance_contrast(rl_result, percent_saturated=.3)
                    if args.enhance_contrast_on_save
                    else imops.rescale_to_8_bit(rl_result))

        print("Done.")
        print()


if __name__ == "__main__":
    main()
