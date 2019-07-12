#!/usr/bin/env python
# -*- python -*-

"""
File: utils_image_subtractor.py
Author: Sami Koho (sami.koho@gmail.com)
Description:
I happened to receive a dataset, of 300+ HCS images in jpg to be analyzed.
The images contained phase contrast data (Red) and fluorescence data (Green)
The color channels in the jpg files were mixed so that the green channel
contained both red and green, whereas red contained only green. Therefore
in order to get the fluorescence only images, I needed to get rid of the
crosstalk. This is a small utility for doing that; it can
of course quite easily be modified into any kind of a batch processing task.
"""

import os
import sys

from miplib.data.io import read, write


def main():
    path = sys.argv[1]
    assert os.path.isdir(path), path

    # Create output directory
    output_dir = os.path.join(path, "Subtracted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(path):
        real_path = os.path.join(path, image_name)
        if not os.path.isfile(real_path) or not real_path.endswith(".jpg"):
            continue
        image_red = read.get_image(real_path, channel=0)
        image_green = read.get_image(real_path, channel=1)

        image_sub = image_green - image_red

        save_name = "sub_" + image_name
        save_path = os.path.join(path, output_dir)
        save_path = os.path.join(save_path, save_name)
        write.image(image_sub, save_path)
        print("Saved %s to %s" % (save_name, save_path))

if __name__ == "__main__":
    main()