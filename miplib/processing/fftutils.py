import numpy as np
from math import floor
from miplib.data.containers.image import Image
from miplib.data.coordinates import polar as indexers
from miplib.processing import windowing, ndarray


def fft(array, interpolation=1.0, window='tukey', *kwargs):
    """ A n-dimensional Forward Discrete Fourier transform with some extra bells and whistles
    added on top of the standard Numpy method.

    :param array: the image to be transformed
    :type array: np.ndarray
    :param interpolation: Add "interpolation" to the FFT by zero-padding prior to transform.
    This is expressed as a multiple of the image size.
    :type interpolation: float
    :param window: a window function to apply. 'tukey' or 'hamming'
    :type window: str or None
    :return: the complex Fourier transform of the input array
    """

    # Apply a Window if requested
    if window is None:
        pass
    elif window == 'tukey':
        array = windowing.apply_tukey_window(array, *kwargs)
    elif window == 'hamming':
        array = windowing.apply_hamming_window(array)

    # Add extra padding
    if interpolation > 1.0:
        new_shape = tuple(int(interpolation * i) for i in array.shape)
        array = ndarray.expand_to_shape(array, new_shape)

    # Transform forward
    array = np.fft.fftshift(np.fft.fftn(array))

    return array


def ifft(array_f, interpolation=1.0):
    """ A n-dimensional Inverse Discrete Fourier transform with some extra bells and whistles
    added on top of the standard Numpy method. Assumes a FFT shifted Fourier domain image.

    :param array_f: the image to be transformed
    :type array_f: np.ndarray
    :param interpolation: add interpolation, by defining a value > 1.0. Corresponds to
    enlargement of the result image.
    :type interpolation: float
    :return: returns the iFFTd array
    """

    # Add  padding
    if interpolation > 1.0:
        new_shape = tuple(int(interpolation * i) for i in array_f.shape)
        array_f = ndarray.expand_to_shape(array_f, new_shape)

    # Transform back
    iarray_f = np.fft.ifftn(np.fft.fftshift(array_f))

    return iarray_f


def ideal_fft_filter(image, threshold, kind='low'):
    """
    An ideal high/low pass frequency domain noise filter.
    :param image: an Image object
    :param threshold: threshold value [0,1], where 1 corresponds
    to the maximum frequency.
    :param kind: filter type 'low' for low-pass, 'high' for high pass
    :return: returns the filtered Image.
    """
    assert isinstance(image, Image)

    spacing = image.spacing

    fft_image = np.fft.fftshift(np.fft.fftn(image))

    if kind == 'low':
        indexer = indexers.PolarLowPassIndexer(image.shape)
    elif kind == 'high':
        indexer = indexers.PolarHighPassIndexer(image.shape)
    else:
        raise ValueError("Unknown filter kind: {}".format(kind))

    r_max = floor(min(image.shape) / 2)

    fft_image *= indexer[threshold*r_max]

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)


def butterworth_fft_filter(image, threshold, n=3):
    """Create low-pass 2D Butterworth filter.
    :Parameters:
       size : tuple
           size of the filter
       cutoff : float
           relative cutoff frequency of the filter (0 - 1.0)
       n : int, optional
           order of the filter, the higher n is the sharper
           the transition is.
    :Returns:
       numpy.ndarray
         filter kernel in 2D centered
   """
    if not 0 < threshold <= 1.0:
        raise ValueError('Cutoff frequency must be between 0 and 1.0')

    if not isinstance(n, int):
        raise ValueError('n must be an integer >= 1')

    assert isinstance(image, Image)

    spacing = image.spacing

    # Create Fourier grid
    r = indexers.SimplePolarIndexer(image.shape).r

    threshold *= image.shape[0]

    butter = 1.0 / (1.0 + (r / threshold) ** (2 * n))  # The filter

    fft_image = np.fft.fftshift(np.fft.fftn(image))
    fft_image *= butter

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)


def gaussian_fft_filter(image, threshold):
    """
    Create low-pass 2D Gaussian filter.
    :Parameters:
       size : tuple
           size of the filter
       cutoff : float
           relative cutoff frequency of the filter (0 - 1.0)
       n : int, optional
           order of the filter, the higher n is the sharper
           the transition is.
    :Returns:
       numpy.ndarray:  filter kernel in 2D centered
   """
    if not 0 < threshold <= 1.0:
        raise ValueError('Cutoff frequency must be between 0 and 1.0')

    assert isinstance(image, Image)

    spacing = image.spacing

    # Create Fourier grid
    r = indexers.SimplePolarIndexer(image.shape).r

    r /= image.shape[0]

    gauss = np.exp(-(r**2/(2*(threshold**2))))
    fft_image = np.fft.fftshift(np.fft.fftn(image))
    fft_image *= gauss

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)

