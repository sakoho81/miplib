"""
Sami Koho 01/2017

Image resolution measurement by Fourier Ring Correlation.

"""

import numpy as np
import os
import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.processing.image as imops
from miplib.data.containers.fourier_correlation_data import FourierCorrelationData, \
    FourierCorrelationDataCollection
from miplib.data.containers.image import Image
from . import analysis as fsc_analysis
from miplib.processing import windowing
import miplib.data.io.read as imread

def calculate_single_image_frc(image, args, average=True, trim=True, z_correction=1):
    """
    A simple utility to calculate a regular FRC with a single image input

    :param image: the image as an Image object
    :param args:  the parameters for the FRC calculation. See *miplib.ui.frc_options*
                  for details
    :return:      returns the FRC result as a FourierCorrelationData object

    """
    assert isinstance(image, Image)

    frc_data = FourierCorrelationDataCollection()

    # Hamming Windowing
    if not args.disable_hamming:
        spacing = image.spacing
        image = Image(windowing.apply_hamming_window(image), spacing)

    # Split and make sure that the images are the same siz
    image1, image2 = imops.checkerboard_split(image)
    #image1, image2 = imops.reverse_checkerboard_split(image)
    image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)

    # Run FRC
    iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
    frc_task = FRC(image1, image2, iterator)
    frc_data[0] = frc_task.execute()

    if average:
        # Split and make sure that the images are the same size
        image1, image2 = imops.reverse_checkerboard_split(image)
        image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)
        iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
        frc_task = FRC(image1, image2, iterator)

        frc_data[0].correlation["correlation"] *= 0.5
        frc_data[0].correlation["correlation"] += 0.5*frc_task.execute().correlation["correlation"]

    freqs = frc_data[0].correlation["frequency"].copy()
    
    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d
  
    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    # Analyze results
    analyzer = fsc_analysis.FourierCorrelationAnalysis(frc_data, image1.spacing[0], args)

    result = analyzer.execute(z_correction=z_correction)[0]
    point = result.resolution["resolution-point"][1]

    log_correction = func(point, *params)
    result.resolution["spacing"] /= log_correction
    result.resolution["resolution"] /= log_correction

    return result

def calculate_two_image_frc(image1, image2, args, z_correction=1):
    """
    A simple utility to calculate a regular FRC with a two image input

    :param image: the image as an Image object
    :param args:  the parameters for the FRC calculation. See *miplib.ui.frc_options*
                  for details
    :return:      returns the FRC result as a FourierCorrelationData object
    """
    assert isinstance(image1, Image)
    assert isinstance(image2, Image)

    assert image1.shape == image2.shape

    frc_data = FourierCorrelationDataCollection()

    spacing = image1.spacing

    if not args.disable_hamming:

        image1 = Image(windowing.apply_hamming_window(image1), spacing)
        image2 = Image(windowing.apply_hamming_window(image2), spacing)

    # Run FRC
    iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
    frc_task = FRC(image1, image2, iterator)
    frc_data[0] = frc_task.execute()

    # Analyze results
    analyzer = fsc_analysis.FourierCorrelationAnalysis(frc_data, image1.spacing[0], args)

    return analyzer.execute(z_correction=z_correction)[0]

def calculate_single_image_sectioned_frc(image, args, rotation=45, orthogonal=True, trim=True):
    """
    A function utility to calculate a single image FRC on a Fourier ring section. The section
    is defined by the section size d_angle (in args) and the section rotation.
    :param image: the image as an Image object
    :param args:  the parameters for the FRC calculation. See *miplib.ui.frc_options*
                  for details
    :param rotation: defines the orientation of the fourier ring section
    :param orthogonal: if True, FRC is calculated from two sections, oriented at rotation
    and rotation + 90 degrees
    :return:      returns the FRC result as a FourierCorrelationData object

    """
    assert isinstance(image, Image)

    frc_data = FourierCorrelationDataCollection()

    # Hamming Windowing
    if not args.disable_hamming:
        spacing = image.spacing
        image = Image(windowing.apply_hamming_window(image), spacing)
   

    # Run FRC
    def frc_helper(image1, image2, args, rotation):
        iterator = iterators.SectionedFourierRingIterator(image1.shape, args.d_bin, args.d_angle)
        iterator.angle = rotation
        frc_task = FRC(image1, image2, iterator)
        return frc_task.execute()
    
    image1, image2 = imops.checkerboard_split(image)
    image1, image2 = imops.zero_pad_to_matching_shape(image1, image2)

    image1_r, image2_r = imops.reverse_checkerboard_split(image)
    image1_r, image2_r = imops.zero_pad_to_matching_shape(image1_r, image2_r)

    pair_1 = frc_helper(image1, image2, args, rotation)
    pair_2 = frc_helper(image1_r, image2_r, args, rotation)
    
    pair_1.correlation["correlation"] * 0.5
    pair_1.correlation["correlation"] += 0.5 * pair_2.correlation["correlation"]
   
    if orthogonal:
        pair_1_o = frc_helper(image1, image2, args, rotation+90)
        pair_2_o = frc_helper(image1_r, image2_r, args, rotation+90)
    
        pair_1_o.correlation["correlation"] * 0.5
        pair_1_o.correlation["correlation"] += 0.5 * pair_2_o.correlation["correlation"]

        pair_1.correlation["correlation"] += 0.5 * pair_1_o.correlation["correlation"]
    
    frc_data[0] = pair_1

    freqs = frc_data[0].correlation["frequency"].copy()
    
    def func(x, a, b, c, d):
        return a * np.exp(c * (x - b)) + d
  
    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]

    # Analyze results
    analyzer = fsc_analysis.FourierCorrelationAnalysis(frc_data, image1.spacing[0], args)

    result = analyzer.execute()[0]
    point = result.resolution["resolution-point"][1]

    log_correction = func(point, *params)
    result.resolution["spacing"] /= log_correction
    result.resolution["resolution"] /= log_correction

    return result

def batch_evaluate_frc(path, options):
    """
    Batch calculate FRC resolution for files placed in a directory
    :param options: options for the FRC
    :parame path:   directory that contains the images to be analyzed
    """
    assert os.path.isdir(path)

    measures = FourierCorrelationDataCollection()
    image_names = []

    for idx, image_name in enumerate(sorted(os.listdir(path))):
    
        real_path = os.path.join(path, image_name)
        # Only process images. The bioformats reader can actually do many more file formats
        # but I was a little lazy here, as we usually have tiffs.
        if not os.path.isfile(real_path) or not real_path.endswith((".tiff", ".tif")):
            continue
        # ImageJ files have particular TIFF tags that can be processed correctly
        # with the options.imagej switch
        image = imread.get_image(real_path)

        # Only grayscale images are processed. If the input is an RGB image,
        # a channel can be chosen for processing.
        measures[idx] = calculate_single_image_frc(image, options)

        image_names.append(image_name)

    return measures, image_names
        



class FRC(object):
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, iterator):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        if image1.shape != image2.shape or tuple(image1.spacing) != tuple(image2.spacing):
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.pixel_size = image1.spacing[0]

        # Expand to square
        image1 = imops.zero_pad_to_cube(image1)
        image2 = imops.zero_pad_to_cube(image2)

        self.iterator = iterator
        # Calculate power spectra for the input images.
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2))

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
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Calculate FRC
        spatial_freq = radii.astype(np.float32) / self.freq_nyq
        n_points = np.array(points)

        with np.errstate(divide="ignore", invalid="ignore"):
            frc = np.abs(c1) / np.sqrt(c2 * c3)
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)


        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set