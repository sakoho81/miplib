"""
itkutils.py

Copyright (C) 2014 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains several utilities & filters for simplified
usage of ITK (www.itk.org) modules in Python. Most of the ITK classes
have been implemented in similar manner, so it should be rather
easy to include additional filters.

"""
import SimpleITK as sitk
import numpy

def convert_to_numpy(itk_image):
    """
    A simple conversion function from ITK:Image to a Numpy array. Please notice
    that the pixel size information gets lost in the conversion. If you want
    to conserve image information, rather use ImageStack class method in
    iocbio.io.image_stack module
    """
    assert isinstance(itk_image, sitk.Image)
    return sitk.GetArrayFromImage(itk_image)


def convert_from_numpy(array):
    assert isinstance(array, numpy.ndarray)
    return sitk.GetImageFromArray(array)

def make_itk_transform(type, parameters, fixed_parameters):
    transform = getattr(sitk, type)
    assert issubclass(transform, sitk.Transform)

    transform.SetParameters(parameters)
    transform.SetFixedParameters(fixed_parameters)

    return transform

def resample_image(image, transform, reference=None):
    """
    Resampling filter for manipulating data volumes. This function can be
    used to transform an image module or to perform up or down sampling
    for example.

    image       =   input image object itk::Image
    transform   =   desired transform itk::Transform
    image_type  =   pixel type of the image data
    reference   =   a reference image, which can be used in resizing
                    applications, when different dimensions and or
                    spacing are desired to the output image
    """
    assert isinstance(image, sitk.Image)
    if reference is None:
        reference = image
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInput(image)
    region = reference.GetLargestPossibleRegion()

    resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetSize(region.GetSize())
    resampler.SetOutputOrigin(reference.GetOrigin())
    resampler.SetOutputSpacing(reference.GetSpacing())
    resampler.SetOutputDirection(reference.GetDirection())
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute()


def rotate_psf(psf, transform, return_numpy=False):
    """
    In case, only one point-spread-function (PSF) is to be used in the image
    fusion, it needs to be rotated with the transform of the moving_image.
    The transform is generated during the registration process.

    psf             = A Numpy array, containing PSF data
    transform       = itk::VersorRigid3DTransform object
    return_numpy    = it is possible to either return the result as an
                      itk:Image, or a ImageStack.

    """
    assert isinstance(transform, sitk.VersorRigid3DTransform)

    if isinstance(psf, numpy.ndarray):
        image = convert_from_numpy(psf)
    else:
        image = psf

    assert isinstance(image, sitk.Image)

    # We don't want to translate, but only rotate
    parameters = transform.GetParameters()
    for i in range(3, 6):
        parameters[i] = 0.0
    transform.SetParameters(parameters)

    # Find  and set center of rotation This assumes that the PSF is in
    # the centre of the volume, which should be expected, as otherwise it
    # will cause translation of details in the final image.
    imdims = image.GetDimension()
    imspacing = image.GetSpacing()

    center = map(
        lambda size, spacing: spacing * size / 2, imdims, imspacing
    )

    transform.SetFixedParameters(center)

    # Rotate
    image = resample_image(image, transform)

    if return_numpy:
        return convert_to_numpy(image)
    else:
        return image


def resample_to_isotropic(itk_image):
    """
    This function can be used to rescale or upsample a confocal stack,
    which generally has a larger spacing in the z direction.

    :param itk_image:   an ITK:Image object
    :param image_type:  an ITK image type string (e.g. 'IUC3')
    :return:            returns a new ITK:Image object with rescaled
                        axial dimension
    """
    assert isinstance(itk_image, sitk.Image)

    filter = sitk.ResampleImageFilter()
    transform = sitk.Transform()
    transform.SetIdentity()

    filter.SetInterpolator(sitk.sitkBSpline)
    filter.SetDefaultPixelValue(0)

    # Set output spacing
    spacing = itk_image.GetSpacing()

    if len(spacing) != 3:
        print "The function resample_to_isotropic(itk_image, image_type) is" \
              "intended for processing 3D images. The input image has %d " \
              "dimensions" % len(spacing)
        return

    scaling = spacing[2]/spacing[0]

    spacing[:] = spacing[0]
        
    filter.SetOutputSpacing(spacing)
    filter.SetOutputDirection(itk_image.GetDirection())
    filter.SetOutputOrigin(itk_image.GetOrigin())

    # Set Output Image Size
    region = itk_image.GetLargestPossibleRegion()
    size = region.GetSize()
    size[2] = int(size[2]*scaling)
    filter.SetSize(size)

    transform.SetIdentity()
    filter.SetTransform(transform)
    filter.SetInput(itk_image)

    return filter.Execute()


def rescale_intensity(image, input_type, output_type='IUC3'):
    """
    A filter to scale the intensities of the input image to the full range
    allowed by the pixel type

    Inputs:
        image       = an itk.Image() object
        input_type  = pixel type string of the input image. Must be an ITK
                      recognized pixel type
        output_type = same as above, for the output image
    """
    rescaling = getattr(
        itk.RescaleIntensityImageFilter, input_type+output_type).New()
    rescaling.SetInput(image)
    rescaling.Update()

    return rescaling.GetOutput()


def gaussian_blurring_filter(image, image_type, variance):
    """
    Gaussian blur filter
    """

    filter = getattr(
        itk.DiscreteGaussianImageFilter, image_type+image_type).New()
    filter.SetInput(image)
    filter.SetVariance(variance)
    filter.Update()

    return filter.GetOutput()


def grayscale_dilate_filter(image, image_type, kernel_radius):
    """
    Grayscale dilation filter for 2D/3D datasets
    """

    if '2' in image_type:
        image_type = image_type + image_type + 'SE2'
    else:
        image_type = image_type + image_type + 'SE3'

    filter = getattr(itk.GrayscaleDilateImageFilter, image_type).New()

    kernel = filter.GetKernel()
    kernel.SetRadius(kernel_radius)
    kernel = kernel.Ball(kernel.GetRadius())

    filter.SetKernel(kernel)
    filter.SetInput(image)

    filter.Update()

    return filter.GetOutput()

def mean_filter(image, image_type, kernel_radius):
    """
    Uniform Mean filter for itk.Image objects
    """
    filter = getattr(itk.MeanImageFilter, image_type+image_type).New()

    filter.SetRadius(kernel_radius)
    filter.SetInput(image)

    filter.Update()

    return filter.GetOutput()

def median_filter(image, image_type, kernel_radius):
    """
    Median filter for itk.Image objects

    :param image:           an itk.Image object
    :param image_type:      image type string (e.g. IUC2, IF3)
    :param kernel_radius:   median kernel radius
    :return:                filtered image
    """
    filter = getattr(itk.MedianImageFilter, image_type+image_type).New()
    #radius = filter.GetRadius()
    #radius.Fill(kernel_radius)
    filter.SetRadius(kernel_radius)
    filter.SetInput(image)

    filter.Update()

    return filter.GetOutput()



def normalize_image_filter(image, image_type):
    """
    Normalizes the pixel values in an image to Mean of zero and Variance
    of one. A floating point image_type is expected. For integer pixel
    type, casting to a float is recommended before using this.
    """

    filter = getattr(itk.NormalizeImageFilter, image_type+image_type).New()
    filter.SetInput(image)
    filter.Update()

    return filter.GetOutput()

def threshold_image_filter(image, threshold, image_type, th_value=0,
                           th_method="below"):
    """
    Thresholds an image by setting pixel values above or below "threshold"
    to "th_value". The result is not a binary image, but a thresholded
    grayscale image.
    """

    filter = getattr(itk.ThresholdImageFilter, image_type).New()
    filter.SetInput(image)

    if th_method is "above":
        filter.ThresholdAbove(threshold)
    elif th_method is "below":
        filter.ThresholdBelow(threshold)

    filter.SetOutsideValue(th_value)

    filter.Update()

    return filter.GetOutput()


def get_image_statistics(image, image_type, verbal=True):
    """
    A utility to calculate basic image statistics (Mean and Variance here)

    :param image:       an ITK:Image object
    :param image_type:  a string describing the image type (e.g. IUC3). The
                        naming convention as in ITK
    :param verbal:      print results on screen (ON/OFF)
    :return:            returns the image mean and variance in a tuple
    """
    filter = getattr(itk.StatisticsImageFilter, image_type).New()
    filter.SetInput(image)
    filter.Update()

    mean = filter.GetMean()
    variance = filter.GetVariance()

    if verbal is True:
        print "Mean: %s" % mean
        print "Variance: %s" % variance



    return mean, variance


def type_cast(image, input_type, output_type):
    """
    A utility for changing the image pixel container type

    :param image:       An ITK:Image
    :param input_type:  input image type as a string (e.g. IUC3)
    :param output_type: output image type as a string (e.g. IF2)
    :return:            returns the image with new pixel type
    """
    filter = getattr(itk.CastImageFilter, input_type+output_type).New()
    filter.SetInput(image)
    filter.Update()

    return filter.GetOutput()

# THIS ONE DOES NOT WORK WITH SITK YET
# def calculate_center_of_image(image, center_of_mass=False):
#     """
#     Center of an image can be defined either geometrically or statistically,
#     as a Center-of-Gravity measure.
#
#     Based on itk::ImageMomentsCalculator
#     http://www.itk.org/Doxygen/html/classitk_1_1ImageMomentsCalculator.html
#     """
#     if center_of_mass:
#         calculator = getattr(itk.ImageMomentsCalculator, image_type).New()
#         calculator.SetImage(image)
#         calculator.Compute()
#
#         center = tuple(calculator.GetCenterOfGravity())
#     else:
#         imdims = itkio.get_dimensions(image)
#         imspacing = tuple(image.GetSpacing())
#
#         center = map(
#             lambda size, spacing: spacing * size / 2,
#             imdims, imspacing
#         )
#
#     return center


def get_image_subset(image, image_type, multiplier):
    """

    A simple utility for extracting a subset of an ItkImage, based
    on a single floating point multiplier.

    Based on itk::ExtractImageFilter
    http://www.itk.org/Doxygen/html/classitk_1_1ExtractImageFilter.html

    Two parameters: output_size and output_origin define the subset.
    Here the parameters are calculated with a single multiplier.

    """

    filter = getattr(itk.ExtractImageFilter, image_type+image_type).New()

    input_region = image.GetLargestPossibleRegion()
    input_origin = input_region.GetIndex()
    input_size = input_region.GetSize()

    output_region = input_region
    output_size = input_size
    output_origin = input_origin

    for i in range(len(output_size)):
        output_size[i] = int(multiplier*input_size[i])
        output_origin[i] = int(0.5*(input_size[i]-output_size[i]))

    output_region.SetSize(output_size)
    output_region.SetIndex(output_origin)

    filter.SetInput(image)
    filter.SetExtractionRegion(output_region)
    filter.Update()

    return filter.GetOutput()

























