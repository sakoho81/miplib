"""Deconvolve a single image with a given or generated PSF."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
from miplib.data.io.read import get_image
from miplib.data.io.write import image as write_image
from miplib.processing.deconvolution.backends import ViewData, resolve_backend
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psf
from miplib.psf.psfgen import PsfFromFwhm

logger = logging.getLogger(__name__)


def _load_image(path: str) -> Image:
    import SimpleITK as sitk

    result = get_image(path)
    if isinstance(result, Image):
        return result
    if isinstance(result, sitk.Image):
        data = sitk.GetArrayFromImage(result)
        return Image(data, spacing=[1.0] * result.GetDimension())
    raise TypeError(f"Unexpected return type from get_image: {type(result).__name__}")


def _get_psf(
    image: Image,
    psf_path: str | None,
    fwhm: list[float],
    fov: list[float] | None,
) -> Image:
    """Load a PSF from file or generate a Gaussian PSF from FWHM."""
    if psf_path:
        return _load_image(psf_path)

    ndim = image.ndim
    if len(fwhm) == 1 and ndim == 2:
        fwhm = fwhm * 2

    if fov is not None:
        fov_um = list(fov)
    else:
        fov_um = list(map(float, image.shape[:2]))

    if ndim == 2:
        return PsfFromFwhm(
            fwhm=fwhm, shape=image.shape, dims=(fov_um[0], fov_um[1])
        ).xy()
    else:
        return PsfFromFwhm(
            fwhm=fwhm[:2], shape=image.shape[1:], dims=(fov_um[0], fov_um[1])
        ).volume()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deconvolve a single image using Richardson-Lucy iteration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Deconvolve with auto-generated Gaussian PSF\n"
            "  miplib-deconvolve image.tif --fwhm 2.0 --iterations 50\n\n"
            "  # Deconvolve with a PSF loaded from file\n"
            "  miplib-deconvolve image.tif --psf psf.tif --iterations 100\n\n"
            "  # Deconvolve with TV regularization and CUDA\n"
            "  miplib-deconvolve image.tif --fwhm 1.5 --tv-lambda 0.01 --enable-cuda\n"
        ),
    )
    parser.add_argument("image", help="Path to the input image (TIFF).")
    parser.add_argument(
        "--psf", help="Path to a PSF image. If omitted, a Gaussian PSF is generated."
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        nargs="+",
        default=[2.0],
        help="PSF FWHM in µm (one value for isotropic, two for XY).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        nargs="+",
        default=None,
        help="Field-of-view in µm. Default: image shape is used as FOV (1 px/µm).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum RL iterations (default: 100).",
    )
    parser.add_argument(
        "--stop-tau",
        type=float,
        default=1e-4,
        help="Stop when tau1 drops below this threshold.",
    )
    parser.add_argument(
        "--output", "-o", default="deconv_result.tif", help="Output file path."
    )
    parser.add_argument("--enable-cuda", action="store_true", help="Use CUDA backend.")
    parser.add_argument(
        "--first-estimate",
        default="image_mean",
        choices=[e.value for e in FirstEstimate],
    )
    parser.add_argument(
        "--tv-lambda",
        type=float,
        default=0.0,
        help="Total variation regularization strength.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-7,
        help="RL convergence epsilon (clipping floor).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main():
    args = _build_parser().parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    image = _load_image(args.image)
    psf = _get_psf(
        image, args.psf, list(args.fwhm), list(args.fov) if args.fov else None
    )

    psf_arr, adj_psf_arr = prepare_psf(psf, image.spacing)

    source = ArrayDataSource([image])
    vd = ViewData(
        source=source,
        psfs=[psf_arr],
        adj_psfs=[adj_psf_arr],
        weights=[1.0],
        backgrounds=[0.0],
    )

    backend = resolve_backend(
        "cuda" if args.enable_cuda else "cpu", vd, vd.source.shape
    )
    estimate = create_estimate(source, FirstEstimate(args.first_estimate))

    options = RLOptions(
        max_iterations=args.iterations,
        stop_tau=args.stop_tau,
        tv_lambda=args.tv_lambda,
        epsilon=args.epsilon,
    )

    print(
        f"Deconvolving {os.path.basename(args.image)} "
        f"({image.shape}) with {args.iterations} iterations "
        f"({'CUDA' if args.enable_cuda else 'CPU'})"
    )
    begin = time.time()

    deconv = RLDeconvolver(backend, estimate=estimate, options=options)
    for _ in deconv:
        iteration = deconv.tracker.to_dataframe().shape[0]
        if iteration % 10 == 0:
            print(
                f"  Iteration {iteration:4d}  tau1 = {deconv.tracker.current_tau1:.6f}"
            )

    elapsed = time.time() - begin
    print(f"Finished in {elapsed:.1f}s")

    write_image(args.output, deconv.result())
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
