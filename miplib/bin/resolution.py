"""
A convenience script to calculate FRC for images in a directory.
I have a nicer version in a notebook -- will be updated.
"""

import datetime
import os
import sys

import numpy as np
import pandas

import miplib.analysis.resolution.fourier_ring_correlation as frc
import miplib.processing.to_string as strutils
import miplib.ui.cli.miplib_entry_point_options as options
from miplib.data.io import read as imread


def main():
    # Get input arguments
    args = options.get_frc_script_options(sys.argv[1:])
    path = args.directory

    # Create output directory
    output_dir = args.directory
    date_now = datetime.datetime.now().strftime("%H-%M-%S")

    filename = f"{date_now}_miplib_{args.frc_mode}_frc_results.csv"
    filename = os.path.join(output_dir, filename)

    # Get image file names, sort in alphabetic order and complete.
    files_list = [
        i for i in os.listdir(path) if i.endswith((".jpg", ".tif", ".tiff", ".png"))
    ]
    files_list.sort()
    print(f"Number of images to analyze: {len(files_list)}")

    # df_main = pandas.DataFrame(0, index=np.arange(len(files_list)), columns=["Image", "Depth", "Kind", "Resolution"])
    df_main = pandas.DataFrame(
        0, index=np.arange(len(files_list)), columns=["Image", "Resolution"]
    )

    if args.frc_mode == "two-image":

        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a, strict=False)

        for idx, im1, im2 in enumerate(pairwise(files_list)):
            # Read images
            image1 = imread.get_image(os.path.join(path, im1))
            image2 = imread.get_image(os.path.join(path, im2))

            result = frc.calculate_two_image_frc(image1, image2, args)
            title = strutils.common_start(im1, im2)

            resolution = result.resolution["resolution"]
            df_main.iloc[idx] = title, resolution

    elif args.frc_mode == "one-image":
        for idx, im in enumerate(files_list):
            image = imread.get_image(os.path.join(path, im))

            print(f"Analyzing image {im}")

            result = frc.calculate_single_image_frc(image, args)

            title = im.split(".")[0]

            # I left these snippets here to show how one can add additional info
            # to the dataframes in particular use cases.

            # depth = title.split('um_')[0].split("_")[-1]

            # kind = None
            # for x in ("apr_ism", "apr_ism_bplus", "closed", "open", "static_ism", "ism_sim"):
            #     if x in title:
            #         kind = x
            # if kind is None:
            #     raise RuntimeError("Unknown image: {}".format(title))
            resolution = result.resolution["resolution"]
            # df_main.iloc[idx] = title, depth, kind, resolution
            df_main.iloc[idx] = title, resolution

    else:
        raise NotImplementedError()

    df_main.index = list(range(len(df_main)))
    df_main.to_csv(filename)


if __name__ == "__main__":
    main()
