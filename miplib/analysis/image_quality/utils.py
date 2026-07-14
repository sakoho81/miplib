"""
File:        utils.py
Author:      sami.koho@gmail.com

Description:
Various sorts of small utilities for the PyImageQuality software.
Contains all kinds of code snippets that did not find a home in
the main modules.
"""

import numpy
from numpy.typing import ArrayLike
from scipy import ndimage


def analyze_accumulation(x: ArrayLike, fraction: float) -> int:
    """Return the minimal number of tail elements that sum to *fraction* of total."""
    x = numpy.asarray(x)
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    final = fraction * x.sum()
    index = 1
    while x[-index:].sum() < final:
        index += 1
    return index


def calculate_entropy(data: ArrayLike) -> float:
    """Shannon entropy estimated from a 50-bin histogram."""
    data = numpy.asarray(data)
    histogram = ndimage.histogram(data, data.min(), data.max(), 50)
    histogram = histogram[numpy.nonzero(histogram)]
    histogram = histogram.astype(float) / histogram.sum()
    return -numpy.sum(histogram * numpy.log2(histogram))
