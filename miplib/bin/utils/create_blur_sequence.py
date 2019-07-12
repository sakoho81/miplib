"""
Sami Koho - 2015 - sami.koho@gmail.com

A small utility that generates a series of Gaussian blurred images from a
single base image. This utility was use to create an autofocus function test
dataset.

"""

import os
import sys

from scipy import ndimage, misc


def main():
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print("Please specify a path to an image file")
        sys.exit(1)

    original = misc.imread(sys.argv[1], mode='P')

    path_parts = os.path.split(sys.argv[1])
    output_dir = os.path.join(path_parts[0], "Blurred")
    extension = path_parts[1].split(".")[1]
    base_name = path_parts[1].split(".")[0]

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    for mul in range(0, 30):
        sigma = mul * 1
        blurred = ndimage.gaussian_filter(original, sigma)

        file_name = base_name + "_Gaussian_" + str(sigma) + "." + extension
        output_path = os.path.join(output_dir, file_name)

        misc.imsave(output_path, blurred)

if __name__ == "__main__":
    main()
