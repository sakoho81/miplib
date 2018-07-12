import numpy as np
from supertomo.data.containers.image import Image
import supertomo.data.iterators.fourier_ring_iterators as iterators
import supertomo.data.containers.fourier_correlation_data as containers
import supertomo.processing.itk as itkutils


class RotatingFRC(object):

    def __init__(self, image1, image2, d_bin, d_angle, normalize_power=False):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("You should provide a stack for FSC analysis")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        shape = image1.shape[1], image1.shape[2]

        # Create an Iterator
        self.iterator = iterators.FourierRingIterator(shape, d_bin)

        # FFT transforms of the input images
        fft_image1 = np.fft.fftshift(np.fft.fftn(image1)).real
        fft_image2 = np.fft.fftshift(np.fft.fftn(image2)).real

        if normalize_power:
            pixels = image1.shape[0] ** 3
            fft_image1 /= (np.array(pixels * np.mean(image1)))
            fft_image2 /= (np.array(pixels * np.mean(image2)))

        self.fft_image1 = itkutils.convert_from_numpy(fft_image1, (1, 1, 1))
        self.fft_image2 = itkutils.convert_from_numpy(fft_image2, (1, 1, 1))

        self.pixel_size = image1.spacing[0]

        self.d_angle = d_angle
        self.d_bin = d_bin

        self.freq_nyq = int(np.floor(image1.shape[1] / 2.0))
        self.center = np.floor(image1.shape[0]/2.0)

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        data_structure = containers.FourierCorrelationDataCollection()
        radii = self.iterator.radii
        angles = np.arange(0, 360, self.d_angle, dtype=int)
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        for rotation_idx, angle in enumerate(angles):
            fft_plane1 = itkutils.convert_from_itk_image(
                itkutils.rotate_image(self.fft_image1, angle,
                                      interpolation='nearest'))[self.center, :, :]
            fft_plane2 = itkutils.convert_from_itk_image(
                itkutils.rotate_image(self.fft_image2, angle,
                                      interpolation='nearest'))[self.center, :, :]

            # Iterate through the sphere and calculate initial values
            for ind_ring, ring_idx in self.iterator:
                subset1 = fft_plane1[ind_ring]
                subset2 = fft_plane2[ind_ring]

                c1[rotation_idx, ring_idx] = np.sum(subset1 * np.conjugate(subset2))
                c2[rotation_idx, ring_idx] = np.sum(np.abs(subset1) ** 2)
                c3[rotation_idx, ring_idx] = np.sum(np.abs(subset2) ** 2)

                points[rotation_idx, ring_idx] = len(subset1)

            # Calculate FRC
            spatial_freq = radii.astype(np.float32) / self.freq_nyq
            n_points = np.array(points)
            frc = np.abs(c1) / np.sqrt(c2 * c3)

            data_set = containers.FourierCorrelationData()
            data_set.correlation["correlation"] = frc
            data_set.correlation["frequency"] = spatial_freq
            data_set.correlation["points-x-bin"] = n_points

            data_structure[angle] = data_set

        return data_structure
