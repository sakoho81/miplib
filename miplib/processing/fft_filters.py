import numpy as np
from math import floor
from miplib.data.containers.image import Image
from miplib.data.indexers import polar_indexers as indexers


def fft_filter(image, threshold, kind='low'):
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

