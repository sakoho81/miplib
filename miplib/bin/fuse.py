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
from contextlib import ExitStack
from pathlib import Path

import numpy as np

import miplib.data.containers.image_data as image_data
import miplib.data.io.write as imwrite
import miplib.processing.to_string as genutils
import miplib.ui.cli.miplib_entry_point_options as arguments
from miplib.data.adapters.image_data import ImageDataSource
from miplib.data.containers.image_data import ImageKey, ImageType
from miplib.processing.deconvolution.backends import (
    ViewData,
    resolve_backend,
)
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
from miplib.processing.deconvolution.psf_utils import compute_virtual_psfs, prepare_psfs


def _resolve_views(views, n_registered):
    if views == -1:
        return list(range(n_registered))
    if hasattr(views, "__iter__"):
        return list(views)
    return [views]


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


def _save_results(data, result, options):
    save_tiff = getattr(options, "save_tiff", None)
    if save_tiff:
        imwrite.image(save_tiff, result)
    if getattr(options, "save_hdf", False):
        channel = getattr(options, "channel", 0)
        scale = getattr(options, "scale", 100)
        data.add_fused_image(
            channel, scale, result.view(np.ndarray), list(result.spacing)
        )


def _create_view_data(data, views, options):
    """Build an ImageDataSource, load and prepare PSFs, return ViewData."""
    channel = getattr(options, "channel", 0)
    scale = getattr(options, "scale", 100)

    source = ImageDataSource(data, views, ImageType.REGISTERED, channel, scale)
    psf_images = [
        data.get_image(ImageKey(ImageType.PSF, v, channel, scale)) for v in views
    ]
    norms, adjs = prepare_psfs(psf_images, source.spacing)

    virtual_psf = "opt" in getattr(options, "fusion_method", "summative")
    if virtual_psf and len(norms) >= 2:
        adjs = compute_virtual_psfs(norms, adjs)

    return ViewData(
        source=source,
        psfs=norms,
        adj_psfs=adjs,
        weights=[1.0] * source.n_views,
        backgrounds=[0.0] * source.n_views,
    )


def _allocate_estimate(source, options, tmpdir: Path | str | None = None):
    """Allocate and initialise the estimate array."""
    if tmpdir is not None:
        estimate = np.memmap(
            Path(tmpdir) / "estimate.dat",
            dtype=np.float32,
            mode="w+",
            shape=source.shape,
        )
    else:
        estimate = np.zeros(source.shape, dtype=np.float32)
    create_estimate(
        source,
        getattr(options, "first_estimate", FirstEstimate.IMAGE_MEAN),
        constant=getattr(options, "estimate_constant", 1.0),
        out=estimate,
    )
    return estimate


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

    views = _resolve_views(getattr(options, "fuse_views", -1), n_registered)
    view_data = _create_view_data(data, views, options)

    with ExitStack() as stack:
        tmpdir: Path | None = None
        if getattr(options, "memmap_estimates", False):
            tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        estimate = _allocate_estimate(view_data.source, options, tmpdir)

        backend_name = "cuda" if getattr(options, "enable_cuda", False) else "cpu"
        backend = resolve_backend(backend_name, view_data, view_data.source.shape)

        algo_options = RLOptions(
            fusion_mode=getattr(options, "fusion_method", "summative"),
            n_blocks=getattr(options, "blocks", 1),
            block_pad=getattr(options, "pad", 0),
            max_iterations=getattr(options, "max_nof_iterations", 100),
            stop_tau=getattr(options, "rltv_stop_tau", 0.002),
            tv_lambda=getattr(options, "tv_lambda", 0.0),
            epsilon=getattr(options, "convergence_epsilon", 0.05),
        )

        begin = time.time()
        task = RLDeconvolver(backend, estimate=estimate, options=algo_options)
        for _ in task:
            _progress_print(task, begin)
        end = time.time()
        print()

        print("Fusion complete.")
        print(
            "The fusion process with %i iterations "
            "took %s (H:M:S) to complete."
            % (options.max_nof_iterations, genutils.format_time_string(end - begin))
        )

        _save_results(data, task.result(), options)

    data.close()


if __name__ == "__main__":
    main()
