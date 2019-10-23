
"""
fuse.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the miplib fusion calculation
"""

import os
import sys
import time

import miplib.data.containers.image_data as image_data
import miplib.processing.fusion.fusion as fusion
import miplib.processing.fusion.fusion_cuda as gpufusion
import miplib.processing.to_string as genutils
import miplib.ui.cli.miplib_entry_point_options as arguments
import miplib.ui.utils as uiutils


def main():

    options = arguments.get_fusion_script_options(sys.argv[1:])
    full_path = os.path.join(options.working_directory,
                             options.data_file)

    if not os.path.isfile(full_path):
        raise AttributeError("No such file: %s" % full_path)
    elif not full_path.endswith(".hdf5"):
        raise AttributeError("Not a HDF5 file")

    data = image_data.ImageData(full_path)

    if options.scale not in data.get_scales("registered"):
        print("Images at the defined scale do not exist in the data structure." \
              "The original images will be now resampled. This may take a long" \
              "time depending on the image size and the number of views.")
        data.create_rescaled_images("registered", options.scale)

    if data.get_number_of_images("psf") != data.get_number_of_images("registered"):
        print("Some PSFs are missing. They are going to be calculated from the " \
              "original STED PSF (that is assumed to be at index 0).")
        data.calculate_missing_psfs()

    if not options.disable_cuda:
        print("Trying to run the image fusion with GPU acceleration.")
        task = gpufusion.MultiViewFusionRLCuda(data, options)
    else:
        task = fusion.MultiViewFusionRL(data, options)

    # task = fusion.MultiViewFusionRL(data, options)
    begin = time.time()
    task.execute()
    end = time.time()

    if options.evaluate_results:
        task.show_result()

    print("Fusion complete.")
    print("The fusion process with %i iterations " \
          "took %s (H:M:S) to complete." % (options.max_nof_iterations,
                                            genutils.format_time_string(
                                                end-begin)))

    if uiutils.get_user_input("Do you want to save the result to TIFF? "):
        file_path = os.path.join(options.working_directory,
                                 "fusion_result.tif")
        task.save_to_tiff(file_path)

    if uiutils.get_user_input("Do you want to save the result to the HDF data "
                              "structure? "):
        task.save_to_hdf()

    task.close()
    data.close()


if __name__ == "__main__":
    main()
