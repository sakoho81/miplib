"""
File:        utils.py
Author:      sami.koho@gmail.com

Description:
Various sorts of small utilities for the PyImageQuality software.
Contains all kinds of code snippets that did not find a home in
the main modules.
"""

import numpy
from scipy import ndimage


def analyze_accumulation(x, fraction):
    """
    Analyze the accumulation by starting from the end of the data.
    """
    assert 0.0 < fraction <= 1.0
    final = fraction * x.sum()
    index = 1
    while x[-index:].sum() < final:
        index += 1
    return index


def calculate_entropy(data):
    """
    Calculate the Shannon entropy for data
    """
    # Calculate histogram
    histogram = ndimage.histogram(
        data,
        data.min(),
        data.max(), 50)
    # Exclude zeros
    histogram = histogram[numpy.nonzero(histogram)]
    # Normalize histogram bins to sum to one
    histogram = histogram.astype(float) / histogram.sum()
    return -numpy.sum(histogram * numpy.log2(histogram))





