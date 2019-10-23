#!/usr/bin/env python
# -*- python -*-
"""
File: power.py
Author: Sami Koho (sami.koho@gmail.com)

Description:

A utility script for extracting 1D power spectra of all images within
a defined input directory. The spectra are saved in a single csv
file, each column denoting a single image.
"""
import datetime
import os
import sys

import numpy
import pandas

from miplib.analysis.image_quality import filters
from miplib.data.io import read
from miplib.processing import image as improc
from miplib.ui.cli import miplib_entry_point_options


def main():
    options = miplib_entry_point_options.get_power_script_options(sys.argv[1:])
    path = options.working_directory

    assert os.path.isdir(path)

    # Create output directory
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d")+'_PyIQ_output'
    output_dir = os.path.join(options.working_directory, output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create output file
    date_now = datetime.datetime.now().strftime("%H-%M-%S")
    file_name = date_now + '_PyIQ_power_spectra' + '.csv'
    file_path = os.path.join(output_dir, file_name)

    csv_data = pandas.DataFrame()

    # Scan through images
    for image_in in os.listdir(path):
        if not image_in.endswith((".jpg", ".tif", ".tiff", ".png")):
            continue
        path_in = os.path.join(path, image_in)

        # Get image
        image = read.get_image(path_in, channel=options.rgb_channel)
        image = improc.crop_to_rectangle(image)

        for dim in image.shape:
            if dim != options.image_size:
                image = improc.resize(image, options.image_size)
                break

        task = filters.FrequencyQuality(image, options)
        task.calculate_power_spectrum()
        task.calculate_summed_power()

        power_spectrum = task.get_power_spectrum()

        csv_data[image_in] = power_spectrum[1]

    csv_data.insert(0, "Power", numpy.linspace(0, 1, num=len(csv_data)))
    csv_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()
