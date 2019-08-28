#!/usr/bin/env python
# -*- python -*-

"""
File:        utils_extract_channel.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
A batch processing utility to extract a single channel from RGB
images. The results are saved into Extracted sub-folder of the
original path containing the images.
"""

import os
import sys

from miplib.data.io import read, write


def main():
    path = sys.argv[1]
    assert os.path.isdir(path), path
    channel = sys.argv[2]

    # Create output directory
    output_dir = os.path.join(path, "Extracted")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(path):
        real_path = os.path.join(path, image_name)
        if not os.path.isfile(real_path):
            continue
        image = read.get_image(real_path, channel=channel)

        save_name = "channel_" + channel + "_" + image_name
        save_path = os.path.join(path, output_dir)
        save_path = os.path.join(save_path, save_name)
        write.image(image, save_path)
        print("Saved %s to %s" % (save_name, save_path))

if __name__ == "__main__":
    main()