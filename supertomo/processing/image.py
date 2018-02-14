import numpy as np
from supertomo.data.containers.image import Image
import ndarray


def zoom_to_isotropic_spacing(image):  # type: (Image) -> Image
    assert isinstance(image, Image)

    spacing = image.spacing
    shape = image.shape
    min_spacing = min(spacing)
    zoom = tuple(pixel_spacing/min_spacing for pixel_spacing in spacing)
    if all(zoom) == 1:
        return image
    else:
        new_shape = tuple(int(pixels*dim_zoom)for (pixels, dim_zoom) in zip(shape, zoom))
        return resize(image, new_shape)


def resize(image, size, order=3):  # type: (Image, tuple) -> Image
    """
    Resize the image, using interpolation.

    :param order: The interpolation type defined as order of the b-spline
    :param image: The MyImage object.
    :param size: A tuple of new image dimensions.

    """
    assert isinstance(size, tuple)
    assert isinstance(image, Image)

    zoom = [float(a)/b for a, b in zip(size, image.shape)]
    print "The zoom is %s" % zoom

    image = np.zoom(image, tuple(zoom), order=order)
    image.spacing(i/j for i, j in zip(image.get_spacing(), zoom))

    return image


def apply_hanning(image):  # type: (Image) -> Image
    """
    Apply Hanning window to the image.

    :return:
    """
    npix = image.shape[0]
    window = np.hanning(npix)
    window = np.outer(window, window)

    return image*window


def zero_pad_to_shape(image, shape):  # type: (Image, tuple) -> Image
    """

    :param image: an image
    :param shape: a shape tuple
    :return:
    """

    return ndarray.expand_to_shape(image, shape)


def checkerboard_split(image):
    """
    Splits an image in two, by using a checkerboard pattern.

    :param image: a SuperTomo Image
    :return: two Supertomo Images
    """
    assert isinstance(image, Image)

    # Make an index chess board structure
    shape = image.shape
    odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape)))
    even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))

    # Create the two pseudo images
    image1 = image[odd_index[0], :][:, odd_index[1]]
    image2 = image[even_index[0], :][:, even_index[1]]

    image1.spacing = image.spacing
    image2.spacing = image1.spacing

    return image1, image2


def zero_pad_to_cube(image):
    assert isinstance(image, Image)

    original_shape = image.shape
    nmax = max(original_shape)
    square_shape = (nmax,) * image.ndim
    if square_shape != original_shape:
        return zero_pad_to_shape(image, square_shape)
    else:
        return image
