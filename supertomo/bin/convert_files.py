"""
A program to convert image files into our HDF5 archive format
"""
import sys
from ..io import tiffio

def main():
    if len(sys.argv) < 2:
        print "Please specify a path to the image directory"
        sys.exit(1)

    images, tags = tiffio.get_tiff(sys.argv[1])

    for x in tags:
        print x, ' : ', tags[x]

    print "The shape of the images is: ", images.shape




# Read source images

# Save into HDF5 format


if __name__ == "__main__":
    main()