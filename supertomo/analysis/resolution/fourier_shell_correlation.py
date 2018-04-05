# coding=utf-8
"""
Sami Koho 01/2018

Directional Fourier Shell Correlation for complex resolution analysis
in 3D images.

The code is based on:

Tan et al: “Addressing Preferred Specimen Orientation in Single-Particle
Cryo-EM through Tilting.” Nature Methods 14, no. 8 (July 3, 2017):
793–96. doi:10.1038/nmeth.4347.
"""

import numpy as np

import supertomo.processing.image as ops_myimage
import supertomo.processing.ndarray as ndarray
import fourier_shape_iterators as iterators

from supertomo.data.containers.image import Image
import supertomo.data.containers.fourier_correlation_data as containers
from supertomo.analysis.resolution.analysis import FourierCorrelationAnalysis


class DirectionalFSC(object):
    def __init__(self, image1, image2, args):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("You should provide a stack for FSC analysis")

        if image1.shape != image2.shape:
            print "The size of the images does not match. Zero padding will be applied" \
                  "to match the shapes."
            image1, image2 = ops_myimage.zero_pad_to_matching_shape(image1, image2)

        # # Zoom to isotropic spacing if necessary
        # image1 = ops_myimage.zoom_to_isotropic_spacing(image1, order=1)
        # image2 = ops_myimage.zoom_to_isotropic_spacing(image2, order=1)
        #
        # # Pad to uniform shape
        # image1 = ops_myimage.zero_pad_to_cube(image1)
        # image2 = ops_myimage.zero_pad_to_cube(image2)

        # Create an Iterator
        self.iterator = iterators.FourierShellIterator(image1.shape,
                                                       d_bin=args.d_bin,
                                                       d_angle=args.d_angle)

        # FFT transforms of the input images
        self.fft_image1 = np.fft.fftshift(np.fft.fftn(image1)).real
        self.fft_image2 = np.fft.fftshift(np.fft.fftn(image2)).real

        if args.normalize_power:
            pixels = image1.shape[0]**3
            self.fft_image1 /= (np.array(pixels * np.mean(image1)))
            self.fft_image2 /= (np.array(pixels * np.mean(image2)))

        self._result = None

        self.args = args
        self.pixel_size = image1.spacing[0]

    @property
    def result(self):
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

        data_structure = containers.FourierCorrelationDataCollection()
        radii, angles = self.iterator.steps
        freq_nyq = self.iterator.nyquist
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        # Iterate through the sphere and calculate initial values
        for ind_ring, shell_idx, rotation_idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2))
            c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
            c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)

            points[rotation_idx, shell_idx] = len(subset1)

        # Finish up FRC calculation for every rotation angle and sav
        # results to the data structure.
        for i in range(angles.shape[0]):

            # Calculate FRC for every orientation
            spatial_freq = radii.astype(np.float32) / freq_nyq
            n_points = np.array(points[i])
            frc = ndarray.safe_divide(np.abs(c1[i]), np.sqrt(c2[i] * c3[i]))

            result = containers.FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            # Save result to the structure and move to next
            # angle
            data_structure[angles[i]] = result

        self._result = FourierCorrelationAnalysis(
            data_structure, self.args).calculate_resolution(self.pixel_size)

        return self._result
