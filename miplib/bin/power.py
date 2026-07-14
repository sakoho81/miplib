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
import sys
from pathlib import Path

import numpy
import pandas

from miplib.analysis.image_quality import filters
from miplib.data.io import read
from miplib.processing import image as improc
from miplib.ui.cli import miplib_entry_point_options


def main():
    options = miplib_entry_point_options.get_power_script_options(sys.argv[1:])
    path = Path(options.working_directory)

    assert path.is_dir()

    # Create output directory
    output_dir = datetime.datetime.now().strftime("%Y-%m-%d") + "_PyIQ_output"
    output_dir = path / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file
    date_now = datetime.datetime.now().strftime("%H-%M-%S")
    file_name = date_now + "_PyIQ_power_spectra" + ".csv"
    file_path = output_dir / file_name

    csv_data = pandas.DataFrame()

    # Scan through images
    for image_in in path.iterdir():
        if image_in.suffix not in (".jpg", ".tif", ".tiff", ".png"):
            continue

        # Get image
        image = read.get_image(str(image_in), channel=options.rgb_channel)
        image = improc.crop_to_rectangle(image)

        for dim in image.shape:
            if dim != options.image_size:
                image = improc.resize(image, options.image_size)
                break

        task = filters.FrequencyQuality(image, options)
        task.calculate_power_spectrum()
        task.calculate_summed_power()

        power_spectrum = task.get_power_spectrum()

        csv_data[image_in.name] = power_spectrum[1]

    csv_data.insert(0, "Power", numpy.linspace(0, 1, num=len(csv_data)))
    csv_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()
