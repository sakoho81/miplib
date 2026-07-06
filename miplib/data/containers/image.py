"""
File:        image.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
This file contains a simple class for storing image data.
"""

import argparse
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


class Image(np.ndarray):
    """A numpy.ndarray subclass carrying image data with pixel spacing metadata.

    ``spacing`` (a list of floats) and optional ``filename`` are preserved
    through numpy operations like slicing and ufuncs via ``__array_finalize__``.
    """

    def __new__(
        cls,
        images: ArrayLike,
        spacing: Sequence[float],
        filename: str | None = None,
    ):
        obj = np.asarray(images).view(cls)
        obj.spacing = list(spacing)
        obj.filename = filename
        return obj

    def __array_finalize__(self, obj: np.ndarray | None):
        if obj is not None:
            self.spacing = getattr(obj, "spacing", None)
            self.filename = getattr(obj, "filename", None)


def get_options(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add command-line options for image file I/O."""
    if not isinstance(parser, argparse.ArgumentParser):
        raise TypeError(f"Expected ArgumentParser, got {type(parser).__name__}")
    group = parser.add_argument_group("Image I/O", "Options for image file I/O")
    group.add_argument(
        "--imagej",
        help="Defines wheter the image are in ImageJ tiff format, "
        "and thus contain the pixel size info etc in the TIFF tags. "
        "By default true",
        action="store_true",
    )
    group.add_argument(
        "--rgb-channel",
        help="Select which channel in an RGB image is to be used for quality analysis",
        dest="rgb_channel",
        type=int,
        choices=[0, 1, 2],
        default=1,
    )
    parser.add_argument(
        "--average-filter",
        dest="average_filter",
        type=int,
        default=0,
        help="Analyze only images with similar amount of detail, by selecting a "
        "grayscale average pixel value threshold here",
    )
    parser.add_argument(
        "--file-filter",
        dest="file_filter",
        default=None,
        help="Define a common string in the files to be analysed",
    )
    return parser
