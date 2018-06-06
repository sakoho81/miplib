"""
Sami Koho 01/2017

Image resolution measurement by Fourier Ring Correlation.
re
"""

import numpy as np

import supertomo.data.iterators.fourier_ring_iterators as iterators
import supertomo.processing.image as ops_image
from supertomo.data.containers.fourier_correlation_data import FourierCorrelationData, \
    FourierCorrelationDataCollection
from supertomo.data.containers.image import Image


class FRC(object):
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, d_bin, normalize_power = False):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape or image1.spacing != image2.spacing:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.pixel_size = image1.spacing[0]

        # Expand to square
        image1 = ops_image.zero_pad_to_cube(image1)
        image2 = ops_image.zero_pad_to_cube(image2)

        self.iterator = iterators.FourierRingIterator(image1.shape, d_bin)
        # Calculate power spectra for the input images.
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1)).real
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2)).real

        if normalize_power:
            pixels = image1.shape[0] * image1.shape[1]
            self.fft_image1 /= (np.array(pixels * np.mean(image1)))
            self.fft_image2 /= (np.array(pixels * np.mean(image2)))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))


    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results.

        """
        radii = self.iterator.radii
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for ind_ring, idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2))
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Calculate FRC
        spatial_freq = radii.astype(np.float32) / self.freq_nyq
        n_points = np.array(points)
        frc = np.abs(c1) / np.sqrt(c2 * c3)

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set