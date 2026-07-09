"""
fuse.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the miplib fusion calculation.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import miplib.data.containers.image_data as image_data
import miplib.processing.to_string as genutils
import miplib.ui.cli.miplib_entry_point_options as arguments
import miplib.ui.utils as uiutils
from miplib.data.adapters.image_data import ImageDataSource
from miplib.data.containers.image_data import ImageKey, ImageType
from miplib.processing.deconvolution.backends import _CUDA_AVAILABLE
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions


def _resolve_views(views, n_registered):
    if views == -1:
        return list(range(n_registered))
    if hasattr(views, "__iter__"):
        return list(views)
    return [views]


def _resolve_backend(options):
    """Determine backend from CLI options, falling back to CPU if CUDA unavailable."""
    if getattr(options, "enable_cuda", False):
        if not _CUDA_AVAILABLE:
            print("CUDA not available, falling back to CPU.")
            return "cpu"
        print("Running image fusion with GPU acceleration.")
        return "cuda"
    return "cpu"


def _create_estimate(source, options):
    """Create an estimate array, optionally memory-mapped to disk.

    Returns ``(estimate, tmpdir)`` where *tmpdir* is ``None`` for
    in-memory arrays.  The caller owns *tmpdir* lifecycle.
    """
    if not getattr(options, "memmap_estimates", False):
        return None, None
    tmpdir = tempfile.TemporaryDirectory()
    estimate = np.memmap(
        Path(tmpdir.name) / "estimate.dat",
        dtype=np.float32,
        mode="w+",
        shape=source.shape,
    )
    return estimate, tmpdir


def _create_source_and_psfs(data, options):
    """Build an ImageDataSource and load PSFs from CLI options."""
    views = _resolve_views(
        getattr(options, "fuse_views", -1),
        data.get_number_of_images(ImageType.REGISTERED),
    )
    channel = getattr(options, "channel", 0)
    scale = getattr(options, "scale", 100)

    source = ImageDataSource(data, views, ImageType.REGISTERED, channel, scale)
    psfs = [data.get_image(ImageKey(ImageType.PSF, v, channel, scale)) for v in views]
    return source, psfs


def _progress_print(task, t0):
    df = task.tracker.to_dataframe()
    if len(df) == 0:
        return
    elapsed = time.time() - t0
    row = df.iloc[-1]
    print(
        f"\riter {len(df):3d}  "
        f"tau1={row['tau1']:.4f}  "
        f"leak={row['leak']:.3e}  "
        f"elapsed={genutils.format_time_string(elapsed)}",
        end="",
        flush=True,
    )


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

    source, psfs = _create_source_and_psfs(data, options)
    estimate, tmpdir = _create_estimate(source, options)
    backend = _resolve_backend(options)

    algo_options = RLOptions(
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
    )

    begin = time.time()
    task = RLDeconvolver(
        source,
        psfs,
        estimate=estimate,
        options=algo_options,
        progress_callback=lambda t: _progress_print(t, begin),
    )
    for _ in task:
        pass
    end = time.time()
    print()  # newline after progress line

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

    if tmpdir is not None:
        tmpdir.cleanup()
    data.close()


if __name__ == "__main__":
    main()
