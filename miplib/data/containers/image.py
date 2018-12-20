"""
File:        image.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
This file contains a simple class for storing image data.
"""

import argparse

import numpy


class Image(numpy.ndarray):
    """
    A very simple extension to numpy.ndarray, to contain image data and
    metadata.
    """

    # region Initialization

    def __new__(cls, images, spacing, filename=None):
        obj = numpy.asarray(images).view(cls)

        obj.spacing = list(spacing)
        obj.filename = filename

        return obj

    def __array__finalize__(self, obj):

        self.spacing = getattr(obj, 'spacing')
        self.filename = getattr(obj, 'filename', None)
    # endregion

    # region Properties

    # @property
    # def spacing(self): return self._spacing
    #
    # @spacing.setter
    # def spacing(self, value):
    #     if len(value) != len(self.shape):
    #         raise ValueError("You should define spacing for every dimension")
    #     else:
    #         self._spacing = value

    # endregion

# region Command Line Arguments (refactor)
def get_options(parser):
    """
    Command-line options for the image I/O
    """
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Image I/O", "Options for image file I/O")
    # Parameters for controlling how image files are handled
    group.add_argument(
        "--imagej",
        help="Defines wheter the image are in ImageJ tiff format, "
             "and thus contain the pixel size info etc in the TIFF tags. "
             "By default true",
        action="store_true"
    )
    group.add_argument(
        "--rgb-channel",
        help="Select which channel in an RGB image is to be used for quality"
             " analysis",
        dest="rgb_channel",
        type=int,
        choices=[0, 1, 2],
        default=1
    )
     # File filtering for batch mode processing
    parser.add_argument(
        "--average-filter",
        dest="average_filter",
        type=int,
        default=0,
        help="Analyze only images with similar amount of detail, by selecting a "
             "grayscale average pixel value threshold here"
    )
    parser.add_argument(
        "--file-filter",
        dest="file_filter",
        default=None,
        help="Define a common string in the files to be analysed"
    )
    return parser
# endregion











