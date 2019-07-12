"""
Sami Koho - 2015 - sami.koho@gmail.com

A small utility that generates Gaussian blurred images from a
series of base images. This utility was use to create an autofocus function test
dataset.

"""

import os
import sys

from scipy import ndimage, misc


def main():

    # Check input parameters
    if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
        print("Please specify a path to a directory that contains the pictures")
        sys.exit(1)
    path = sys.argv[1]

    # Create output directory
    output_dir = os.path.join(path, "test_image_series")
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Process every image in the input path
    for image_name in os.listdir(path):
        real_path = os.path.join(path, image_name)
        if not real_path.endswith((".jpg", ".tif", ".tiff", ".png")):
            continue

        original = misc.imread(real_path, flatten=True)
        path_parts = os.path.split(real_path)
        extension = path_parts[1].split(".")[1]
        base_name = path_parts[1].split(".")[0]

        # Save original
        output_path = os.path.join(output_dir, image_name)
        misc.imsave(output_path, original)

        # Blur with Gaussian filter, Sigma=1
        file_name = base_name + "_gaussian_1." + extension
        output_path = os.path.join(output_dir, file_name)
        misc.imsave(output_path, ndimage.gaussian_filter(original, 1))

        # Blur with Gaussian filter, Sigma=2
        file_name = base_name + "_gaussian_2." + extension
        output_path = os.path.join(output_dir, file_name)
        misc.imsave(output_path, ndimage.gaussian_filter(original, 2))

if __name__ == "__main__":
    main()
