"""
fuse.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the miplib fusion calculation.
"""

import os
import sys
import time

import miplib.data.containers.image_data as image_data
import miplib.processing.to_string as genutils
import miplib.ui.cli.miplib_entry_point_options as arguments
import miplib.ui.utils as uiutils
from miplib.data.containers.image_data import ImageKey, ImageType
from miplib.processing.deconvolution.deconvolver import RLDeconvolver


def main():
    options = arguments.get_fusion_script_options(sys.argv[1:])
    full_path = os.path.join(options.working_directory, options.data_file)

    if not os.path.isfile(full_path):
        raise AttributeError(f"No such file: {full_path}")
    elif not full_path.endswith(".hdf5"):
        raise AttributeError("Not a HDF5 file")

    data = image_data.ImageData(full_path)

    if options.scale not in data.get_scales(ImageType.REGISTERED):
        print(
            "Images at the defined scale do not exist in the data structure."
            "The original images will be now resampled. This may take a long"
            "time depending on the image size and the number of views."
        )
        data.create_rescaled_images(ImageType.REGISTERED, options.scale)

    n_psfs = data.get_number_of_images(ImageType.PSF)
    n_registered = data.get_number_of_images(ImageType.REGISTERED)
    if n_psfs != n_registered:
        print(
            "Some PSFs are missing. They are going to be calculated from the "
            "original STED PSF (that is assumed to be at index 0)."
        )
        data.calculate_missing_psfs()

    views = getattr(options, "fuse_views", -1)
    if views == -1:
        views = list(range(n_registered))
    elif hasattr(views, "__iter__"):
        pass
    else:
        views = [views]

    channel = getattr(options, "channel", 0)
    scale = getattr(options, "scale", 100)

    images = [
        data.get_image(ImageKey(ImageType.REGISTERED, v, channel, scale)) for v in views
    ]
    psfs = [data.get_image(ImageKey(ImageType.PSF, v, channel, scale)) for v in views]

    backend = "cuda" if not getattr(options, "disable_cuda", False) else "cpu"
    if backend == "cuda":
        print("Trying to run the image fusion with GPU acceleration.")

    task = RLDeconvolver(
        images,
        psfs,
        fusion_mode=getattr(options, "fusion_method", "summative"),
        backend=backend,
        n_blocks=getattr(options, "blocks", 1),
        block_pad=getattr(options, "pad", 0),
        max_iterations=getattr(options, "max_nof_iterations", 100),
        stop_tau=getattr(options, "rltv_stop_tau", 0.002),
        tv_lambda=getattr(options, "tv_lambda", 0.0),
        epsilon=getattr(options, "convergence_epsilon", 0.05),
        first_estimate=getattr(options, "first_estimate", "image_mean"),
        estimate_constant=getattr(options, "estimate_constant", 1.0),
        virtual_psf="opt" in getattr(options, "fusion_method", "summative"),
        memmap_estimates=getattr(options, "memmap_estimates", False),
        verbose=True,
    )

    begin = time.time()
    task.run()
    end = time.time()

    if getattr(options, "evaluate_results", False):
        task.show_result()

    print("Fusion complete.")
    print(
        "The fusion process with %i iterations "
        "took %s (H:M:S) to complete."
        % (options.max_nof_iterations, genutils.format_time_string(end - begin))
    )

    if uiutils.get_user_input("Do you want to save the result to TIFF? "):
        file_path = os.path.join(options.working_directory, "fusion_result.tif")
        task.save_to_tiff(file_path)

    if uiutils.get_user_input(
        "Do you want to save the result to the HDF data structure? "
    ):
        task.save_to_hdf()

    task.close()
    data.close()


if __name__ == "__main__":
    main()
