import numpy as np
from supertomo.data.containers.image import Image
import ndarray
from scipy.ndimage import interpolation

def zoom_to_isotropic_spacing(image, order=3):
    """
    Resize an Image to isotropic pixel spacing.

    :param image:   a Image object
    :param order:   the spline interpolation type
    :return:        a isotropically spaced Image
    """
    assert isinstance(image, Image)

    spacing = image.spacing
    old_shape = image.shape
    min_spacing = min(spacing)
    zoom = tuple(pixel_spacing/min_spacing for pixel_spacing in spacing)
    new_shape = tuple(int(pixels * dim_zoom) for (pixels, dim_zoom) in zip(old_shape, zoom))

    if new_shape == old_shape:
        return image
    else:
        return resize(image, new_shape, order)


def resize(image, size, order=3):  # type: (Image, tuple) -> Image
    """
    Resize the image, using interpolation.

    :param order:   The interpolation type defined as order of the b-spline
    :param image:   The MyImage object.
    :param size:    A tuple of new image dimensions.

    """
    assert isinstance(size, tuple)
    assert isinstance(image, Image)

    zoom = [float(a)/b for a, b in zip(size, image.shape)]
    print "The zoom is %s" % zoom

    array = interpolation.zoom(image, tuple(zoom), order=order)
    spacing = tuple(i/j for i, j in zip(image.spacing, zoom))

    return Image(array, spacing)


def apply_hanning(image):  # type: (Image) -> Image
    """
    Apply Hanning window to the image.

    :return:
    """
    npix = image.shape[0]
    window = np.hanning(npix)
    window = np.outer(window, window)

    return image*window


def zero_pad_to_shape(image, shape):
    """
    Apply zero padding to cast an Image into the given shape. The zero padding
    will be applied evenly on all sides of the image.

    :param image: an Image object
    :param shape: a shape tuple
    :return:      the zero padded Image
    """
    assert isinstance(image, Image)

    if image.shape == shape:
        return image
    else:
        return Image(ndarray.expand_to_shape(image, shape), image.spacing)


def zero_pad_to_matching_shape(image1, image2):
    """
    Apply zero padding to make the size of two Images match.
    :param image1: an Image object
    :param image2: an Image object
    :return:       zero padded image1 and image2
    """

    assert isinstance(image1, Image)
    assert isinstance(image2, Image)

    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape))

    image1 = zero_pad_to_shape(image1, shape)
    image2 = zero_pad_to_shape(image2, shape)

    return image1, image2


def checkerboard_split(image):
    """
    Splits an image in two, by using a checkerboard pattern.

    :param image:   a SuperTomo Image
    :return:        two Supertomo Images
    """
    assert isinstance(image, Image)

    # Make an index chess board structure
    shape = image.shape
    odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape)))
    even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))

    # Create the two pseudo images
    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, odd_index[1]]
        image2 = image[even_index[0], :][:, even_index[1]]
    else:
        image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
        image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]

    image1.spacing = image.spacing
    image2.spacing = image1.spacing

    return image1, image2


def zero_pad_to_cube(image):
    """
    Apply zero padding to cast an image into a cube shape (to match the number
    of pixels in all dimensions)
    :param image: an Image object
    :return:      zero padded input image, or the original, if already a cube
    """
    assert isinstance(image, Image)

    original_shape = image.shape
    nmax = max(original_shape)
    square_shape = (nmax,) * image.ndim
    if square_shape != original_shape:
        return zero_pad_to_shape(image, square_shape)
    else:
        return image


def crop_to_shape(image, shape):
    """
    Crop image to shape. The new image will be centered at the geometric center
    of the original
    image
    :param image:   An N-dimensional Image to be cropped
    :type image:    Image
    :param shape:   The new, cropped size; should be greater or equal than
                    the original
    :type shape:    tuple
    :return:        Returns the cropped image as an Image object
    """
    assert isinstance(image, Image)
    assert image.ndim == len(shape)
    assert all((x >= y for x, y in zip(image.shape, shape)))

    return Image(ndarray.contract_to_shape(image,shape), image.spacing)
