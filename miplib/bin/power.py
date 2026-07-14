#!/usr/bin/env python
# -*- python -*-

"""
File:        power.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
Extract 1D radial power spectra from microscopy images. Computes the
rotationally averaged power spectrum for each image in a directory and
exports the results to a CSV file.
"""

import datetime
import sys

import numpy as np
import pandas as pd

from miplib.data.io import read
from miplib.processing import fftutils
from miplib.processing import image as improc
from miplib.ui.cli import miplib_entry_point_options


def main():
    """
    Power spectrum extraction tool.
    """
    options = miplib_entry_point_options.get_power_options(sys.argv[1:])
    input_path = options.input

    if not input_path.is_dir():
        print(f"Error: {input_path} is not a directory")
        sys.exit(1)

    # Determine output path
    if options.output:
        output_path = options.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = input_path / f"{timestamp}_power_spectra.csv"

    # Collect all image files
    image_files = []
    for ext in ("*.jpg", "*.tif", "*.tiff", "*.png"):
        image_files.extend(input_path.glob(ext))
    image_files.sort()

    if not image_files:
        print(f"No images found in {input_path}")
        sys.exit(1)

    print(f"Extracting power spectra from {len(image_files)} images...")

    # Process each image
    csv_data = pd.DataFrame()
    for image_path in image_files:
        try:
            # Get image
            image = read.get_image(str(image_path), channel=options.rgb_channel)
            image = improc.crop_to_rectangle(image)

            # Resize if needed
            for dim in image.shape:
                if dim != options.image_size:
                    image = improc.resize(image, options.image_size)
                    break

            # Extract power spectrum
            freq, power = fftutils.power_spectrum_1d(image)
            csv_data[image_path.name] = power
            print(f"  ✓ {image_path.name}")
        except Exception as e:
            print(f"  ✗ {image_path.name}: {e}")

    # Add frequency axis and save
    csv_data.insert(0, "Frequency", np.linspace(0, 1, num=len(csv_data)))
    csv_data.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
