import numpy as np

import supertomo.processing.image as ops_myimage
import fourier_shape_iterators as iterators

from supertomo.data.containers.image import Image
import supertomo.data.containers.fourier_correlation as containers


class DirectionalFSC():
    def __init__(self, image1, image2, args):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("You should provide a stack for FSC analysis")

        # Zoom to isotropic spacing if necessary
        image1 = ops_myimage.zoom_to_isotropic_spacing(image1)
        image2 = ops_myimage.zoom_to_isotropic_spacing(image2)

        # Pad to uniform shape
        image1 = ops_myimage.zero_pad_to_cube(image1)
        image2 = ops_myimage.zero_pad_to_cube(image2)

        # Create an Iterator
        if args.d_theta != 0:
            self.iterator = iterators.FourierShellIterator(image1.shape,
                                                           d_bin=args.width_ring,
                                                           d_theta=args.d_theta)
            d_angle = args.d_theta
        else:
            self.iterator = iterators.FourierShellIterator(image1.shape,
                                                           d_bin=args.width_ring,
                                                           d_phi=args.d_phi)
            d_angle = args.d_phi

        # FFT transforms of the input images
        self.fft_image1 = np.fft.fftshift(np.fft.rfftn(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.rfftn(image2))

        if args.normalize_power:
            pixels = image1.shape[0]**3
            self.fft_image1 /= (np.array(pixels * np.mean(image1)))
            self.fft_image2 /= (np.array(pixels * np.mean(image2)))

        # Get the Nyquist frequency
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))
        self._data = None

        self.radii = np.arange(0, self.freq_nyq, args.width_ring)
        self.angles = np.arange(0, 360, d_angle)

    @property
    def result(self):
        if self._data is None:
            return self.execute()
        else:
            return self._data

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        self._data = containers.FourierCorrelationDataCollection()
        shape = (self.angles.shape[0], self.radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(self.radii.shape, dtype=np.float32)

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
        for i in range(self.angles.shape[0]):

            # Calculate FRC for every orientation
            spatial_freq = self.radii.astype(np.float32) / self.freq_nyq
            n_points = np.array(points[i])
            frc = np.abs(c1[i]) / np.sqrt(c2[i] * c3[i])

            result = containers.FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            # Save result to the structure and move to next
            # angle
            self._data[self.angles[i]] = result

        return self._data
