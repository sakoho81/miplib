"""
Sami Koho - IIT

This file contains classes for generating different kinds of complex
indexing structures (masks).
"""

import numpy as np


class SimplePolarIndexer(object):
    """
    Basic indexer for a polar/spherical coordinate system
    """
    def __init__(self, shape):
        assert isinstance(shape, tuple) or \
               isinstance(shape, list) or \
               isinstance(shape, np.ndarray)
        assert 1 < len(shape) < 4

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)

        meshgrid = np.meshgrid(*axes)
        self.r = np.sqrt(sum([axis**2 for axis in meshgrid]))

    def __getitem__(self, item):
        return self.r == item


class PolarLowPassIndexer(SimplePolarIndexer):
    """
    Generates a low-pass mask in the polar coordinate system, i.e. points
    closer than the specified distance will be selected.
    """
    def __getitem__(self, item):
        return self.r < item


class PolarHighPassIndexer(SimplePolarIndexer):
    """
    Generates a high-pass mask in the polar coordinate system, i.e. points
    farther than the specified distance will be selected.
    """
    def __getitem__(self, item):
        return self.r > item



