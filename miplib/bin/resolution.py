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
from miplib.data.io import read as imread
import miplib.ui.cli.miplib_entry_point_options as options
import miplib.processing.to_string as strutils
import miplib.processing.image as imops

def main():

    # Get input arguments
    args = options.get_frc_script_options(sys.argv[1:])
    path = args.working_directory

    # Create output directory
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + '_MIPLIB_output'
    output_dir = os.path.join(args.working_directory, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    date_now = datetime.datetime.now().strftime("%H-%M-%S")

    filename = date_now + "_miplib_{}_frc_results.csv".format(args.mode)
    filename = os.path.join(output_dir, filename)

    # Get image file names, sort in alphabetic order and complete.
    files_list = list(i for i in os.listdir(path)
                      if i.endswith((".jpg", ".tif", ".tiff", ".png")))
    files_list.sort()
    print(('Number of images to analyze: ', len(files_list)))

    df_main = pandas.DataFrame()

    if args.frc_mode == "two-image":
       
        def pairwise(iterable):
            a = iter(iterable)
            return zip(a, a)

        for im1, im2 in pairwise(files_list):
            # Read images
            image1 = imread.get_image(os.path.join(path, im1))
            image2 = imread.get_image(os.path.join(path, im2))

            result = frc.calculate_two_image_frc(image1, image2, args)
            title = strutils.common_start(im1, im2)

            df_temp = pandas.DataFrame()
            df_temp['image'] = title
            df_temp['resolution'] = result.resolution['resolution']

            df_main = pandas.concat([df_main, df_temp])

    elif args.frc_mode == "one-image":
        for im in files_list:
            image = imread.get_image(os.path.join(path, im))

            result = frc.calculate_single_image_frc(image, args)

            title = im.split('.')[0]
            df_temp = pandas.DataFrame()
            df_temp['image'] = title
            df_temp['resolution'] = result.resolution['resolution']

            df_main = pandas.concat([df_main, df_temp])

    else:
        raise NotImplementedError()

    df_main.index = list(range(len(df_main)))
    df_main.to_csv(filename)

if __name__ == '__main__':
    main()
