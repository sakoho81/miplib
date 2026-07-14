#!/usr/bin/env python
"""ISM reconstruction CLI.

Iterates over .mat or .czi files in a directory, runs adaptive pixel
reassignment with optional Wiener or Richardson-Lucy deconvolution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from miplib.data.containers.image import Image
from miplib.data.io import array_detector_data as detio
from miplib.data.io.write import image as write_image
from miplib.processing.deconvolution.backends import ViewData, resolve_backend
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psf
from miplib.processing.deconvolution.wiener import wiener_deconvolution
from miplib.processing.image import rescale_to_8_bit
from miplib.processing.ism.reconstruction import find_image_shifts, shift_and_sum
from miplib.processing.ism.reconstruction import sum_images as sum_detector_images
from miplib.processing.registration.options import RegistrationMethod
from miplib.psf.psfgen import PsfFromFwhm, generate_frc_based_psf

logger = logging.getLogger(__name__)


def _get_data(path: str) -> tuple[str, object]:
    """Read a .mat or .czi file and return (prefix, ArrayDetectorData)."""
    if path.endswith(".mat"):
        return path[:-4], detio.read_carma_mat(path)
    if path.endswith(".czi"):
        return path[:-4], detio.read_airyscan_data(path)
    raise ValueError(f"Unsupported format: {path}")


def _get_psf(image: Image, args: argparse.Namespace) -> Image:
    """Load PSF from file, generate from FWHM, or estimate via FRC."""
    if args.psf:
        import SimpleITK as sitk

        sitk_psf = sitk.ReadImage(str(args.psf))
        data = sitk.GetArrayFromImage(sitk_psf)
        return Image(data, spacing=image.spacing)

    if args.frc_psf:
        return generate_frc_based_psf(image)

    fwhm = args.fwhm
    if image.ndim == 2:
        return PsfFromFwhm(fwhm=fwhm, shape=image.shape, dims=tuple(image.shape)).xy()

    return PsfFromFwhm(
        fwhm=fwhm[:2], shape=image.shape[1:], dims=tuple(image.shape[1:])
    ).volume()


def _run_rl_deconv(image: Image, psf: Image, args: argparse.Namespace) -> Image:
    """Run RL deconvolution on a single image."""
    from miplib.data.adapters.image_data import ArrayDataSource

    psf_arr, adj_psf_arr = prepare_psf(psf, image.spacing)
    source = ArrayDataSource([image])
    vd = ViewData(
        source=source,
        psfs=[psf_arr],
        adj_psfs=[adj_psf_arr],
        weights=[1.0],
        backgrounds=[0.0],
    )
    backend = resolve_backend("cpu", vd, vd.source.shape)
    estimate = create_estimate(source, strategy=FirstEstimate.IMAGE_MEAN)

    options = RLOptions(
        max_iterations=getattr(args, "max_iterations", 20),
        stop_tau=getattr(args, "stop_tau", 1e-4),
    )
    deconv = RLDeconvolver(backend, estimate=estimate, options=options)
    return deconv.run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ISM image scanning microscopy reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Basic pixel reassignment\n"
            "  miplib-ism ./data/\n\n"
            "  # With Wiener deconvolution, FRC-based PSF\n"
            "  miplib-ism ./data/ --ism-mode wiener --frc-psf\n\n"
            "  # With RL deconvolution, FWHM-based PSF\n"
            "  miplib-ism ./data/ --ism-mode rl --fwhm 2.0 --max-iterations 50\n"
        ),
    )
    parser.add_argument(
        "directory", type=Path, help="Directory containing .mat or .czi files"
    )
    parser.add_argument(
        "--ism-mode",
        choices=["reassign", "wiener", "rl", "all"],
        default="reassign",
        help="Reconstruction mode (default: reassign)",
    )
    parser.add_argument(
        "--method",
        choices=["rigid", "phase_correlation"],
        default="rigid",
        help="Registration method (default: rigid)",
    )
    parser.add_argument(
        "--fixed-idx",
        type=int,
        default=None,
        help="Reference detector index (default: 12 for IIT SPAD, 0 for AiryScan)",
    )

    # PSF options
    psf_group = parser.add_argument_group("PSF generation")
    psf_group.add_argument("--psf", type=Path, help="Path to a PSF image file")
    psf_group.add_argument(
        "--fwhm",
        type=float,
        nargs="+",
        default=[2.0],
        help="PSF FWHM in µm (default: 2.0)",
    )
    psf_group.add_argument(
        "--frc-psf",
        action="store_true",
        help="Estimate PSF via FRC-based blind deconvolution",
    )

    # Deconvolution options
    deconv_group = parser.add_argument_group("Deconvolution")
    deconv_group.add_argument(
        "--max-iterations", type=int, default=20, help="RL max iterations"
    )
    deconv_group.add_argument(
        "--wiener-nsr",
        type=float,
        default=100.0,
        help="Noise-to-signal ratio for Wiener filter",
    )

    parser.add_argument("--verbose", action="store_true")
    return parser


def main():
    args = _build_parser().parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    root = args.directory
    if not root.is_dir():
        sys.exit(f"Not a directory: {args.directory}")

    files = sorted(f for f in root.iterdir() if f.suffix in (".mat", ".czi"))
    if not files:
        sys.exit(f"No .mat or .czi files found in {root}")

    method_map = {
        "rigid": RegistrationMethod.ITERATIVE_RIGID,
        "phase_correlation": RegistrationMethod.PHASE_CORRELATION,
    }
    method = method_map[args.method]

    for full_path in files:
        prefix, data = _get_data(str(full_path))
        prefix = root / Path(prefix).name

        # Default fixed_idx based on detector type
        fixed_idx = args.fixed_idx
        if fixed_idx is None:
            fixed_idx = 0 if str(full_path).endswith(".czi") else 12

        print(
            f"\n{full_path.name}: {data.ndetectors} detectors, "
            f"{data.ngates} gates, shape={data[0, 0].shape}, "
            f"spacing={data[0, 0].spacing}"
        )

        print("  Registering detector images...")
        t0 = time.time()
        shifts, transforms = find_image_shifts(data, method=method, fixed_idx=fixed_idx)
        print(f"  Registration done in {time.time() - t0:.1f}s")

        print("  Pixel reassignment...")
        t0 = time.time()
        ism_result = shift_and_sum(data, transforms)
        print(f"  Reassignment done in {time.time() - t0:.1f}s")

        confocal = sum_detector_images(data)
        write_image(f"{prefix}_confocal.tif", rescale_to_8_bit(confocal))
        write_image(f"{prefix}_ism.tif", rescale_to_8_bit(ism_result))

        if args.ism_mode in ("wiener", "all"):
            print("  Wiener deconvolution...")
            psf = _get_psf(ism_result, args)
            t0 = time.time()
            wiener = wiener_deconvolution(ism_result, psf, nsr=args.wiener_nsr)
            print(f"  Wiener done in {time.time() - t0:.1f}s")
            write_image(f"{prefix}_ism_wiener.tif", rescale_to_8_bit(wiener))

        if args.ism_mode in ("rl", "all"):
            print("  RL deconvolution...")
            psf = _get_psf(ism_result, args)
            t0 = time.time()
            rl_result = _run_rl_deconv(ism_result, psf, args)
            print(f"  RL done in {time.time() - t0:.1f}s")
            write_image(f"{prefix}_ism_rl.tif", rescale_to_8_bit(rl_result))

        print(f"  Saved results with prefix: {prefix}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
