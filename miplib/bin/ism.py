import sys
import os
import numpy as np

from matplotlib import pyplot as plt
import miplib.ui.plots.scatter as scatter
import miplib.ui.plots.frc as frcplots

from miplib.data.io import carma
from miplib.data.containers.image import Image
import miplib.processing.image as imops
from miplib.processing.ism import reconstruction as ismrec
from miplib.processing.deconvolution import deconvolve, wiener
from miplib.psf import frc_psf
from miplib.data.messages import image_writer_wrappers as imwrap
from miplib.processing.segmentation import masking
import miplib.ui.cli.miplib_entry_point_options as options

from miplib.data.io.write import image as imwrite


def main():
    args = options.get_ism_script_options(sys.argv[1:])
    
    # Only allow translation transform, as rotation is not needed here, 
    args.reg_translate_only = True

    # Get Carma mat files in the working direcory
    root_dir = args.directory
    files = tuple(f for f in os.listdir(root_dir) if f.endswith(".mat"))

    # Generate ISM result for each file
    for filename in files:
        filename_prefix = filename.split(".mat")[0]

        full_path = os.path.join(root_dir, filename)

        data = carma.read_carma_mat(full_path)
        
        print ("Opened a Carma file {} for ISM reconstruction".format(filename))
        print ("The file contains images from {} laser gates and {} detector " 
               "channels.".format(data.ngates, data.ndetectors))
        print ("The image size is {} and the pixel spacing {} um".format(data[0,0].shape, data[0,0].spacing))

        print ("Calculating image shifts for adaptive reassignment.")
        # Run registration
        x,y,transforms = ismrec.find_image_shifts(data, args)

        # Save shift scatter, if saving plots is enabled
        if args.save_plots:
            fig = scatter.xy_scatter_plot_with_labels(
                y,x, range(len(x)), size=args.plot_size)
            scatter_name = "{}_shifts_scatter.eps".format(filename_prefix)
            scatter_path = os.path.join(root_dir, scatter_name)
            fig.savefig(scatter_path, dpi=1200, bbox_inches='tight', 
                pad_inches=0, transparent=True)

        # Generate ISM and confocal results
        ism_result = ismrec.shift_and_sum(data, transforms)
        result_sum = ismrec.sum(data)

        ism_result_name = "{}_apr_ism.tif".format(filename_prefix)
        confocal_result_name = "{}_open.tif".format(filename_prefix)

        imwrite(os.path.join(root_dir, ism_result_name), 
            imops.rescale_to_8_bit(ism_result))
        imwrite(os.path.join(root_dir, confocal_result_name), 
            imops.rescale_to_8_bit(result_sum))

        # Run FRC based Wiener filter, if requested
        if any(args.ism_mode == mode for mode in ("wiener", "all")):
            print ("Running the blind Wiener filter.") 
            psf = frc_psf.generate_frc_based_psf(ism_result, args)
            wiener_result = wiener.wiener_deconvolution(
                ism_result, psf, snr=args.wiener_nsr
                )
            wiener_result_name = "{}_apr_ism_bplus_wiener.tif".format(filename_prefix)
            imwrite(os.path.join(root_dir, wiener_result_name), 
            imops.rescale_to_8_bit(wiener_result))

        # Run FRC based RL deconvolution, if requested
        if any(args.ism_mode == mode for mode in ("rl", "all")): 
            print("Running the blind RL deconvolution.")
            psf = frc_psf.generate_frc_based_psf(ism_result, args)

            # Enable automatic background correction with --rl-auto-background
            if args.rl_auto_background:
                background_mask= masking.make_local_intensity_based_mask(
                    ism_result, 40, invert=True)
                masked_image = Image(ism_result*background_mask, ism_result.spacing)
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
            imops.rescale_to_8_bit(rl_result))
        
        print ("Done.")
        print()



if __name__ == "__main__":
    main()
