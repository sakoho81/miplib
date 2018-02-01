"""
Sami Koho 01/2017

Image resolution measurement by Fourier Ring Correlation.

The code in this file was modified from FRC implementation by Filippo Arcadu

The original file header is shown below:

#######                                                                      #######
#######         FOURIER RING CORRELATION ANALYSIS FOR RESOLUTION             #######
#######                                                                      #######
#######  This routine evaluates the resol by means of the fourier            #######
#######  ring correlation (FRC). The inputs are two reconstructions made     #######
#######  with the same algorithm on a sinogram storing the odd-map_disted    #######
#######  projections and on an other one storing the even-map_disted projec.-#######
#######  tions. The two images are transformed with the FFT and their        #######
#######  transform centered. Then, rings of increasing radius R are selec-   #######
#######  ted, starting from the origin of the Fourier space, and the         #######
#######  Fourier coefficients lying inside the ring are used to calculate    #######
#######  the FRC at R, that is FRC(R), with the following formula:           #######
#######                                                                      #######
####### FRC(R)=(sum_{i in R}I_{1}(r_{i})*I_{2}(r_{i}))/sqrt((sum_{i in R}    #######
#######        ||I_{1}(r_{i})||^{2})*(sum_{i in R}||I_{2}(r_{i})||^{2}))     #######
#######                                                                      #######
#######  At the same time, the so-called '2*sigma' curve is calculated at    #######
#######  each step R:                                                        #######
#######                F_{2*sigma}(R) = 2/sqrt(N_{p}(R)/2)                   #######
#######  where N_{p} is the number of pixels in the ring of radius R.        #######
#######  Then, the crossing point between FRC(R) and 2*sigma(R) is found     #######
#######  as the first zero crossing point with negative slope of the dif-    #######
#######  ference curve:                                                      #######
#######                D(R) = FRC(R) - F_{2*sigma}(R)                        #######
#######  The resol is calculated as real space distance correspon-           #######
#######  to this intersection point.                                         ####### 
#######                                                                      #######
#######  Reference:                                                          #######
#######  "Fourier Ring Correlation as a resol criterion for super-           #######
#######  resol microscopy", N. Banterle et al., 2013, Journal of             #######
#######  Structural Biology, 183  363-367.                                   #######
#######                                                                      #######                                    
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 16/09/2013        #######
#######                                                                      #######
####################################################################################
####################################################################################
####################################################################################
"""

import numpy as np
from supertomo.data.containers.image import Image
from supertomo.data.containers.fourier_correlation import FourierCorrelationData

import fourier_shape_iterators as iterators
import supertomo.processing.image as ops_image


class FRC(object):
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, args):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape or image1.spacing != image2.spacing:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.args = args
        self.pixel_size = image1.spacing[0]

        # Expand to square
        image1 = ops_image.zero_pad_to_cube(image1)
        image2 = ops_image.zero_pad_to_cube(image2)

        self.iterator = iterators.FourierRingIterator(original_shape,
                                                      d_bin=args.width_ring)
        # FFT transforms of the input images
        self.fft_image1 = np.fft.fftshift(np.fft.rfft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.rfft2(image2))

        if args.normalize_power:
            pixels = image1.shape[0] * image1.shape[1]
            self.fft_image1 /= (np.array(pixels * np.mean(image1)))
            self.fft_image2 /= (np.array(pixels * np.mean(image2)))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))

        self._result = None

    @property
    def result(self):
        """
        Get the FRC points. In case they have not been calculated already,
        the FRC calculation will be run first.

        :return: Returns a dictionary {y:frc_values, x:frequencies,
                 fit:curve fit to the y values, equation:the equation for the
                 fitted function.
        """
        if self._result is None:
            return self.execute()
        else:
            return self._result

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """
        width_ring = self.args.width_ring
        radii = np.arange(0, self.freq_nyq, width_ring)
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

        self._result = FourierCorrelationData()
        self._result.correlation["correlation"] = frc
        self._result.correlation["frequency"] = spatial_freq
        self._result.correlation["point-x-bin"] = n_points

        return self._result
