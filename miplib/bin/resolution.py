import datetime
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import pandas

import miplib.analysis.resolution.fourier_ring_correlation as frc
import miplib.utils.string as strutils
from miplib.analysis.resolution import common as frc_common
from miplib.data.io import read as imread
from miplib.utils.dataclasses import options_from_dict


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Fourier ring correlation analysis",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--frc-mode", choices=["two-image", "one-image"], default="one-image"
    )
    parser.add_argument(
        "--frc-options",
        type=str,
        default=None,
        help='FRC options as JSON dict, e.g. \'{"d_bin": 2, "resolution_threshold_criterion": "half-bit"}\'',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    path = args.directory

    frc_options = options_from_dict(
        frc_common.FRCOptions,
        json.loads(args.frc_options) if args.frc_options else None,
    )

    output_dir = args.directory
    date_now = datetime.datetime.now().strftime("%H-%M-%S")

    filename = f"{date_now}_miplib_{args.frc_mode}_frc_results.csv"
    filename = output_dir / filename

    files_list = sorted(
        str(p) for p in path.iterdir() if p.suffix in (".jpg", ".tif", ".tiff", ".png")
    )
    print(f"Number of images to analyze: {len(files_list)}")

    df_main = pandas.DataFrame(
        0, index=list(range(len(files_list))), columns=["Image", "Resolution"]
    )

    if args.frc_mode == "two-image":

        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a, strict=False)

        for idx, (im1, im2) in enumerate(pairwise(files_list)):
            image1 = imread.get_image(im1)
            image2 = imread.get_image(im2)

            result = frc.calculate_two_image_frc(image1, image2, frc_options)
            title = strutils.common_start(im1, im2)

            resolution = result.resolution["resolution"]
            df_main.iloc[idx] = title, resolution

    elif args.frc_mode == "one-image":
        for idx, im in enumerate(files_list):
            image = imread.get_image(im)
            print(f"Analyzing image {im}")

            result = frc.calculate_single_image_frc(image, frc_options)

            title = im.split(".")[0]
            resolution = result.resolution["resolution"]
            df_main.iloc[idx] = title, resolution

    else:
        raise NotImplementedError()

    df_main.index = list(range(len(df_main)))
    df_main.to_csv(filename)


if __name__ == "__main__":
    main()
