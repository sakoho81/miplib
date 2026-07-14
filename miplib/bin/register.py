"""Register multiple images to a reference image."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from miplib.data.adapters.registration_data import (
    ArrayRegistrationDataSource,
    HDF5RegistrationDataSource,
    RegistrationDataSource,
)
from miplib.data.containers.image import Image
from miplib.data.io.read import get_image
from miplib.processing.itk import (
    convert_from_itk_image,
    convert_to_itk_image,
    resample_image,
)
from miplib.processing.registration import (
    MultiViewRegistration,
    RegistrationMethod,
    RegistrationOptions,
)
from miplib.processing.registration.options import Metric

_IMAGE_EXTS = {".tif", ".tiff", ".mha", ".mhd", ".png", ".jpg", ".jpeg", ".bmp"}


def _load_image(path: str) -> Image:
    import SimpleITK as sitk

    result = get_image(path)
    if isinstance(result, Image):
        return result
    if isinstance(result, sitk.Image):
        return convert_from_itk_image(result)
    raise TypeError(f"Unexpected return type from get_image: {type(result).__name__}")


def _resolve_source(args: argparse.Namespace) -> RegistrationDataSource:
    output_paths: list[str] | None = getattr(args, "output", None)
    output_dir: str = getattr(args, "output_dir", ".")

    if len(args.images) == 1:
        p = args.images[0]

        if p.suffix == ".hdf5":
            from miplib.data.containers.image_data import ImageData, ImageType

            data = ImageData(str(p))
            channel: int = getattr(args, "channel", 0)
            scale_val: int = getattr(args, "scale", 100)
            if scale_val not in data.get_scales(ImageType.ORIGINAL):
                data.create_rescaled_images(ImageType.ORIGINAL, scale_val)

            n_views = data.get_number_of_images(ImageType.ORIGINAL)
            overwrite: bool = getattr(args, "overwrite", False)
            views = [
                v
                for v in range(n_views)
                if overwrite
                or v == args.fixed_idx
                or not data.check_if_exists(ImageType.REGISTERED, v, channel, scale_val)
            ]
            if not views:
                sys.exit(
                    "All views already registered. Use --overwrite to re-register."
                )

            return HDF5RegistrationDataSource(data, views, channel, scale_val)

        if p.is_dir():
            paths: list[str] = []
            for ext in _IMAGE_EXTS:
                paths.extend(str(f) for f in sorted(p.glob(f"*{ext}")))
            if not paths:
                sys.exit(f"No images found in: {p}")
            return ArrayRegistrationDataSource(
                [_load_image(p_) for p_ in paths],
                output_dir=output_dir,
                output_paths=output_paths,
            )

    return ArrayRegistrationDataSource(
        [_load_image(p) for p in args.images],
        output_dir=output_dir,
        output_paths=output_paths,
    )


def _build_options(args: argparse.Namespace) -> RegistrationOptions | None:
    opts = RegistrationOptions(method=args.method)
    for field in (
        "learning_rate",
        "min_step_length",
        "max_iterations",
        "relaxation_factor",
        "sampling_percentage",
        "mattes_histogram_bins",
        "subpixel",
        "translate_only",
        "use_initializer",
    ):
        val = getattr(args, field, None)
        if val is not None:
            setattr(opts, field, val)
    if getattr(args, "metric", None):
        metric = args.metric
        if isinstance(metric, str):
            metric = Metric.from_string(metric)
        opts.metric = metric
    return opts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register multiple images to a reference image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Rigid registration with default settings\n"
            "  miplib-register image1.tif image2.tif image3.tif --fixed-idx 0\n\n"
            "  # Phase correlation on a whole directory\n"
            "  miplib-register ./frames/ --method phase_correlation\n\n"
            "  # Rigid registration from an HDF5 file\n"
            "  miplib-register data.hdf5 --channel 0 --scale 100\n"
        ),
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Input images, a directory, or an HDF5 file",
    )
    parser.add_argument(
        "--method",
        type=RegistrationMethod.from_string,
        default=RegistrationMethod.ITERATIVE_RIGID,
        metavar="METHOD",
        help="Registration method: rigid, similarity, affine, phase_correlation (default: rigid)",
    )
    parser.add_argument(
        "--fixed-idx", type=int, default=0, help="Reference image index (default: 0)"
    )
    parser.add_argument(
        "--output",
        "-o",
        nargs="*",
        type=Path,
        help="Output paths for registered images",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("."),
        type=Path,
        help="Output directory (default: current directory)",
    )

    # HDF5 options
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--scale", type=int, default=100)

    # Optimizer
    parser.add_argument("--learning-rate", type=float, default=0.7)
    parser.add_argument("--min-step-length", type=float, default=0.001)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--relaxation-factor", type=float, default=0.7)

    # Metric
    parser.add_argument(
        "--metric",
        choices=["correlation", "mattes", "mean-squared-difference"],
        default="correlation",
    )
    parser.add_argument("--sampling-percentage", type=float, default=1.0)
    parser.add_argument("--mattes-histogram-bins", type=int, default=15)

    # Phase correlation
    parser.add_argument("--subpixel", type=int, default=100)

    # Flags
    parser.add_argument("--translate-only", action="store_true")
    parser.add_argument("--use-initializer", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-register views that already have REGISTERED images",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser


def main():
    args = _build_parser().parse_args(sys.argv[1:])

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    source = _resolve_source(args)

    if source.n_views < 2:
        sys.exit("Need at least 2 images to register")

    print(
        f"Registering {source.n_views} views to index {args.fixed_idx} "
        f"using {args.method.value}"
    )
    begin = time.time()

    mv = MultiViewRegistration(
        source,
        method=args.method,
        fixed_idx=args.fixed_idx,
        options=_build_options(args),
    )
    transforms = mv.execute()

    fixed_image = source.get_image(args.fixed_idx)
    for idx, transform in enumerate(transforms):
        params = tuple(transform.GetParameters())
        print(f"  View {idx}: {params}")

        if idx == args.fixed_idx:
            continue

        moving = source.get_image(idx)
        itk_moving = convert_to_itk_image(moving)
        resampled = resample_image(
            itk_moving, transform, reference=convert_to_itk_image(fixed_image)
        )
        result_img = convert_from_itk_image(resampled)
        source.save_result(idx, result_img, transform)

    elapsed = time.time() - begin
    print(f"Registration complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
