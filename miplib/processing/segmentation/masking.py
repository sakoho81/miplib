import numpy as np
from scipy import ndimage


from miplib.data.containers.image import Image


def make_local_intensity_based_mask(image, threshold, kernel_size=40, invert=False):
    assert isinstance(image, Image)

    blurred_image = ndimage.uniform_filter(image, size=kernel_size)

    peaks = np.percentile(blurred_image, threshold)
    mask = np.where(blurred_image >= peaks, 1, 0)
    if invert:
        return np.invert(mask.astype(bool))
    else:
        return mask
