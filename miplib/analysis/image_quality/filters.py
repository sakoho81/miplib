# coding=utf-8
"""
File:        filters.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
This file contains the filters that are used for calculating the
image quality parameters in the PyImageQuality software.
-   The LocalImageQuality class is used to run spatial domain
    analysis. It calculates the Shannon entropy value at a
    masked part of an image
-   The FrequencyQuality class is used to calculate statistical
    quality parameters in the frequency domain. The calculations
    are based on the analysis of the tail of the 1D power spect-
    rum.
-   Brenner and Spectral domain autofocus metrics were impelemnted
    as well, based on the two classes above.
"""

import argparse
from math import floor

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage, fftpack, stats

from . import utils
from miplib.data.containers.image import Image
from miplib.processing import image as imutils
from miplib.data.iterators.fourier_ring_iterators import FourierRingIterator


def get_common_options(parser):
    """
    Common command-line options for the image-quality filters
    """
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group(
        "Filters common", "Common options for the quality filters"
    )
    group.add_argument(
        "--power-averaging",
        dest="power_averaging",
        choices=["radial", "additive"],
        default="additive"
    )
    group.add_argument(
        "--normalize-power",
        dest="normalize_power",
        action="store_true"

    )
    group.add_argument(
        "--use-mask",
        dest="use_mask",
        action="store_true"
    )
    group.add_argument(
        "--invert-mask",
        dest="invert_mask",
        action="store_true"
    )
    group.add_argument(
        "--power-threshold",
        dest="power_threshold",
        type=float,
        default=0.4
    )
    group.add_argument(
        "--spatial-threshold",
        dest="spatial_threshold",
        type=int,
        default=80
    )
    
    return parser


class Filter(object):
    """
    A base class for a filter utilizing Image class object
    """
    def __init__(self, image, options, physical=False, verbal=False):

        assert isinstance(image, Image)
        self.options = options

        self.data = image

        self.spacing = self.data.spacing
        self.dimensions = self.data.shape
        self.physical = physical
        self.verbal = verbal

    def set_physical_coordinates(self):
        self.physical = True

    def set_pixel_coordinates(self):
        self.physical = False


class LocalImageQuality(Filter):
    """
    This is a filter for quantifying  image quality, based on the calculation
    of Shannon entropy at image neighborhoods that contain the highest amount
    of detail.
    """

    def __init__(self, image, options, physical=False, verbal=False):

        Filter.__init__(self, image, options, physical, verbal)

        self.data_temp = None
        self.kernel_size = []

    def set_smoothing_kernel_size(self, size):

        if isinstance(size, list):
            assert len(size) == len(self.spacing)
            sizes = size
        elif isinstance(size, float) or isinstance(size, int):
            sizes = [size, ] * len(self.spacing)
        else:
            print("Unknown size type")
            return

        if self.physical is True:
            for i in range(len(sizes)):
                self.kernel_size[i] = sizes[i]/self.spacing[i]
                assert self.kernel_size[i] < self.dimensions[i]
        else:
            self.kernel_size = sizes
            assert all(x < y for x, y in zip(self.kernel_size, self.dimensions)), \
                "Kernel can not be larger than image"

    def run_mean_smoothing(self, return_result=False):
        """
        Mean smoothing is used to create a mask for the entropy calculation
        """

        assert len(self.kernel_size) == len(self.dimensions)
        self.data_temp = ndimage.uniform_filter(self.data[:], size=self.kernel_size)

        if return_result:
            return Image(self.data_temp, self.spacing)

    def calculate_entropy(self):
        """
        Returns the Shannon entropy value of an image.
        """
        # Calculate histogram
        histogram = ndimage.histogram(
            self.data_temp,
            self.data_temp.min(),
            self.data_temp.max(), 50
        )
        # Exclude zeros
        histogram = histogram[np.nonzero(histogram)]
        # Normalize histogram bins to sum to one
        histogram = histogram.astype(float)/histogram.sum()
        return -np.sum(histogram*np.log2(histogram))

    def find_sampling_positions(self):
        """
        Create a mask by finding pixel positions in the smoothed image
        that have pixel values higher than 80% of the maximum value.
        """
        peaks = np.percentile(self.data_temp, self.options.spatial_threshold)
        mask = np.where(self.data_temp >= peaks, 1, 0)
        if self.options.invert_mask:
            return np.invert(mask.astype(bool))
        else:
            return mask

    def calculate_image_quality(self, kernel=None, show=False):
        """
        Calculate an estimate for image quality, based on the
        Shannon entropy measure. options.use_mask switch can
        be used to limit the entropy calculation to detailed
        parts of the image.
        """
        if self.options.use_mask:
            if kernel is not None:
                self.set_smoothing_kernel_size(kernel)

            assert len(self.kernel_size) != 0

            if self.data_temp is None:
                self.run_mean_smoothing()

            positions = self.find_sampling_positions()
            self.data_temp = self.data[:][np.nonzero(positions)]
            if show:
                Image(self.data[:]*positions, self.spacing).show()
        else:
            self.data_temp = self.data[:]

        return self.calculate_entropy()


class FrequencyQuality(Filter):
    """
    A filter for calculated image-quality related parameters in the frequency
    domain. First a one-dimensional power spectrum is calculated for an image,
    after which various types of statistics are calculated for the power
    spectrum tail (frequencies > 40% of Nyquist)
    """
    def __init__(self, image, options, physical=False, verbal=False):

        Filter.__init__(self, image, options, physical=physical, verbal=verbal)

        # Additive form of power spectrum calculation requires a square shaped
        # image
        if self.options.power_averaging == "additive":
            self.data = imutils.crop_to_largest_square(self.data)

        self.simple_power = None
        self.power = None
        self.kernel_size = []

    def set_image(self, image):
        self.data = image

    def calculate_power_spectrum(self):
        """
        A function that is used to calculate a centered 2D power spectrum.
        Additionally the power spectrum can be normalized by image dimensions
        and image intensity mean, if necessary.
        """
        self.power = np.abs(np.fft.fftshift(np.fft.fft2(self.data[:])))**2
        if self.options.normalize_power:
            dims = self.data[:].shape[0]*self.data[:].shape[1]
            mean = np.mean(self.data[:])
            self.power /= (dims*mean)

    def calculate_radial_average(self, bin_size=2):
        """
        Convert a 2D centered power spectrum into 1D by averaging spectral
        power at different radiuses from the zero frequency center
        """
        iterator = FourierRingIterator(self.power.shape, d_bin=bin_size)

        average = np.zeros(iterator.nbins)

        for idx, ring in enumerate(iterator):
            subset = self.power[ring]
            average[idx] = float(subset.sum())/subset.size

        dx = self.data.spacing[0]
        f_k = np.linspace(0, 1, iterator.nbins) * (1.0/(2*dx))

        self.simple_power = [f_k, average]

        if self.options.show_plots:
            plt.plot(np.log10(self.simple_power[0]))
            plt.ylabel("Average power")
            plt.xlabel("Frequency")
            plt.show()

    def calculate_summed_power(self):
        """
        Calculate a 1D power spectrum fro 2D power spectrum, by summing all rows and
        columns, and then summing negative and positive frequencies, to form a
        N/2+1 long 1D array. This approach is significantly faster to calculate
        than the radial average.
        """

        power_sum = np.zeros(self.power.shape[0])
        for i in range(len(self.power.shape)):
            power_sum += np.sum(self.power, axis=i)
        zero = floor(float(power_sum.size)/2)
        power_sum[zero+1:] = power_sum[zero+1:]+power_sum[:zero-1][::-1]
        power_sum = power_sum[zero:]
        dx = self.data.spacing[0]
        f_k = np.linspace(0, 1, power_sum.size)*(1.0/(2*dx))

        self.simple_power = [f_k, power_sum]

        if self.options.show_plots:
            plt.plot(self.simple_power[0], self.simple_power[1], linewidth=2, color="red")
            plt.ylabel("Total power")
            plt.yscale('log')
            plt.xlabel('Frequency')
            plt.show()

    def analyze_power_spectrum(self):
        """
        Run the image quality analysis on the power spectrum
        """
        assert self.data is not None, "Please set an image to process"
        self.calculate_power_spectrum()

        # Choose a method to calculate 1D power spectrum
        if self.options.power_averaging == "radial":
            self.calculate_radial_average()
        elif self.options.power_averaging == "additive":
            self.calculate_summed_power()
        else:
            raise NotImplementedError

        # Extract the power spectrum tail
        hf_sum = self.simple_power[1][self.simple_power[0] > self.options.power_threshold*self.simple_power[0].max()]

        # Calculate parameters
        f_th = self.simple_power[0][self.simple_power[0] > self.options.power_threshold*self.simple_power[0].max()][-utils.analyze_accumulation(hf_sum, .2)]
        mean = np.mean(hf_sum)
        std = np.std(hf_sum)
        entropy = utils.calculate_entropy(hf_sum)
        nm_th = 1.0e9/f_th
        pw_at_high_f = np.mean(self.simple_power[1][self.simple_power[0] > .9*self.simple_power[0].max()])
        skew = stats.skew(np.log(hf_sum))
        kurtosis = stats.kurtosis(hf_sum)
        mean_bin = np.mean(hf_sum[0:5])

        return [mean, std, entropy, nm_th, pw_at_high_f, skew, kurtosis, mean_bin]

    def show_all(self):
        """
        A small utility to show a plot of the 2D and 1D power spectra
        """
        fig, subplots = plt.subplots(1, 2)
        if self.power is not None:
            subplots[0].imshow(np.log10(self.power))
        if self.simple_power is not None:
            #index = int(len(self.simple_power[0])*.4)
            #subplots[1].plot(self.simple_power[0][index:], self.simple_power[1][index:], linewidth=1)
            subplots[1].plot(self.simple_power[0], self.simple_power[1], linewidth=1)
            subplots[1].set_yscale('log')
        plt.show()

    def get_power_spectrum(self):
        """
        Returns the calculated 1D power spectrum. Please make sure to create
        power spectrum before calling this.
        """
        assert self.simple_power is not None
        return self.simple_power


class SpectralMoments(FrequencyQuality):
    """
    Our implementation of the Spectral Moments autofocus metric
    Firestone, L. et al (1991). Comparison of autofocus methods for automated
    microscopy. Cytometry, 12(3), 195–206.
    http://doi.org/10.1002/cyto.990120302
    """

    def calculate_percent_spectrum(self):
        self.simple_power[1] /= (self.simple_power[1].sum()/100)

    def calculate_spectral_moments(self):
        """
        Run the image quality analysis on the power spectrum
        """
        assert self.data is not None, "Please set an image to process"
        self.calculate_power_spectrum()

        # Choose a method to calculate 1D power spectrum
        if self.options.power_averaging == "radial":
            self.calculate_radial_average()
        elif self.options.power_averaging == "additive":
            self.calculate_summed_power()
        else:
            raise NotImplementedError

        self.calculate_percent_spectrum()

        bin_index = np.arange(1, self.simple_power[1].shape[0]+1)

        return (self.simple_power[1]*np.log10(bin_index)).sum()


class BrennerImageQuality(Filter):
    """
    Our implementation of the Brenner autofocus metric
    Brenner, J. F. et al (1976). An automated microscope for cytologic research
    a preliminary evaluation. Journal of Histochemistry & Cytochemistry, 24(1),
    100–111. http://doi.org/10.1177/24.1.1254907
    """
    def __init__(self, image, options, physical=False, verbal=False):

        Filter.__init__(self, image, options, physical, verbal)
        # This is not really necessary. It was just added in order to compare
        # with the frequency domain measures.
        self.data = imutils.crop_to_largest_square(image)

    def calculate_brenner_quality(self):
        data = self.data
        rows = data.shape[0]
        columns = data.shape[1]-2
        temp = np.zeros((rows, columns))

        temp[:] = ((data[:, 0:-2] - data[:, 2:])**2)

        return temp.sum()















