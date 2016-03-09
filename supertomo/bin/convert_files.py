"""
A program to convert image files into our HDF5 archive format
"""
import sys
import os
from ..io import utils, image_data


def main():
    # The user can give a directory from the command line or alternatively
    # the image are expected to reside in the current working directory.
    if len(sys.argv) == 2:
        directory = sys.argv[1]
        assert os.path.isdir(directory)
    else:
        directory = os.getcwd()

    # Create a new HDF5 file.
    file_name = raw_input("Give a name for the HDF5 file: ")
    file_name += ".hdf5"
    data_path = os.path.join(directory, file_name)
    data = image_data.ImageData(data_path)

    # Add every image in the directory to the new file. You should be
    # carefull to only include relevant tomography data into the
    # image directory.
    for image_name in os.listdir(directory):
        full_path = os.path.join(directory, image_name)

        if full_path.endswith((".tiff", ".tif")):
            images, spacing = utils.get_imagej_tiff(full_path)
        elif full_path.endswith((".mhd", ".mha")):
            images, spacing = utils.get_itk_image(full_path)
        elif full_path.endswith((".hdf5", ".raw")):
            continue
        else:
            print "Unknown image: %s. Skipping it." % image_name
            continue

        assert "angle_" in image_name, "Please specify rotation angle for all images"
        angle = image_name.split("angle_")[-1].split(".")[0]
        data.add_original_image(images, angle, spacing, image_name)

    data.set_active_image(0, "original")
    print "The spacing is: ", data.get_voxel_size()

    data.close()


if __name__ == "__main__":
    main()