import numpy as np
from math import floor
from supertomo.data.containers.image import Image
from supertomo.data.indexers import polar_indexers as indexers


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
