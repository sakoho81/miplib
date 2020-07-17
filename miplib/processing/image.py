import numpy as np
from scipy.ndimage import interpolation

from . import ndarray
from miplib.data.containers.image import Image


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
    zoom = tuple(pixel_spacing / min_spacing for pixel_spacing in spacing)
    new_shape = tuple(int(pixels * dim_zoom) for (pixels, dim_zoom) in zip(old_shape, zoom))

    if new_shape == old_shape:
        return image
    else:
        return resize(image, new_shape, order)

def zoom_to_spacing(image, spacing, order=3, verbose=False):

    assert isinstance(image, Image)
    assert image.ndim == len(spacing)

    zoom = tuple(i/j for i, j in zip(image.spacing, spacing))
    if verbose:
        print("The zoom is ", zoom)

    array = interpolation.zoom(image, zoom, order=order)

    return Image(array, spacing)


def resize(image, size, order=3, verbose=False):  # type: (Image, tuple) -> Image
    """
    Resize the image, using interpolation.

    :param order:   The interpolation type defined as order of the b-spline
    :param image:   The MyImage object.
    :param size:    A tuple of new image dimensions.

    """
    assert isinstance(size, tuple)
    assert isinstance(image, Image)

    zoom = [float(a) / b for a, b in zip(size, image.shape)]
    if verbose:
        print("The zoom is %s" % zoom)

    array = interpolation.zoom(image, tuple(zoom), order=order)
    spacing = tuple(i / j for i, j in zip(image.spacing, zoom))

    return Image(array, spacing)


def apply_hanning(image):  # type: (Image) -> Image
    """
    Apply Hanning window to the image.

    :return:
    """

    windows = (np.hanning(i) for i in image.shape)

    result = Image(image.astype('float64'), image.spacing)
    for window in windows:
        result *= window

    return result


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

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = zero_pad_to_shape(image1, shape)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = zero_pad_to_shape(image2, shape)

    return image1, image2


def remove_zero_padding(image, shape):
    """

    :param image: The zero padded image
    :param shape: The original image size (before padding)
    :return:
    """

    assert isinstance(image, Image)
    assert len(shape) == image.ndim

    return Image(ndarray.contract_to_shape(image, shape), image.spacing)


def checkerboard_split(image, disable_3d_sum = False):
    """
    Splits an image in two, by using a checkerboard pattern.

    :param image:   a miplib Image
    :return:        two miplib Images
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
        if disable_3d_sum:
            image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
            image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]

        else:
            image1 = image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]] + \
                     image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]

            image2 = image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]] + \
                     image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]

    # image1.spacing = tuple(i * np.sqrt(2) for i in image.spacing)
    image1.spacing = image.spacing
    image2.spacing = image1.spacing

    return image1, image2


def reverse_checkerboard_split(image, disable_3d_sum = False):
    """
    Splits an image in two, by using a checkerboard pattern.

    :param image:   a miplib Image
    :return:        two miplib Images
    """
    assert isinstance(image, Image)

    # Make an index chess board structure
    shape = image.shape
    odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape)))
    even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))

    # Create the two pseudo images
    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, even_index[1]]
        image2 = image[even_index[0], :][:, odd_index[1]]
    else:
        if disable_3d_sum:
            image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]
            image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]

        else:
            image1 = image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]] + \
                     image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]

            image2 = image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]] + \
                     image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]

    #image1.spacing = tuple(i * np.sqrt(2) for i in image.spacing)
    image1.spacing = image.spacing
    image2.spacing = image1.spacing

    return image1, image2

def summed_checkerboard_split(image):
    """
    Splits an image in two, by using a checkerboard pattern and diagonal pixels
    in each 4 pixel group (2D) case and orthogonal diagonal groups (never adjacent)
    in 3D case.

    :param image:   a miplib Image
    :return:        two miplib Images
    """
    assert isinstance(image, Image)

    # Make an index chess board structure
    shape = image.shape
    odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape)))
    even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))

    # Create the two pseudo images
    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, odd_index[1]] + image[even_index[0], :][:, even_index[1]]
        image2 = image[odd_index[0], :][:, even_index[1]] + image[even_index[0], :][:, odd_index[1]]

        image1.spacing = tuple(i * 2 for i in image.spacing)
        image2.spacing = image1.spacing
    else:
        image1 = image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]] + \
                 image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]] + \
                 image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]] + \
                 image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, even_index[2]]

        image2 = image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]] + \
                 image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]] + \
                 image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]] +\
                 image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]

        image1.spacing = tuple(i * 2 for i in image.spacing)
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


def crop_to_largest_square(image, physical_dims=False):
    """
    Crops an image into a largest square shape that fits inside the image area in all
    dimensions. The cropping can bone either in physical units or in pixels (typically pixels)
    :param image: an image Object
    :return: the cropped image
    """
    assert isinstance(image, Image)

    if physical_dims:
        shape_real = list(x*y for x, y in zip(image.shape, image.spacing))
        min_shape_real = (min(*shape_real), ) * image.ndim
        min_shape_px = list(x / y for x, y in zip(min_shape_real, image.spacing))
    else:
        min_shape_px = (min(*image.shape),) * image.ndim

    return remove_zero_padding(image, min_shape_px)


def crop_to_shape(image, shape, offset):
    """
    Crop image to shape.

    :param image:   An N-dimensional Image to be cropped
    :type image:    Image
    :param shape:   The new, cropped size; should be greater or equal than
                    the original
    :type shape:    tuple
    :return:        Returns the cropped image as an Image object
    """
    assert isinstance(image, Image)
    assert image.ndim == len(shape) == len(offset)
    assert all((v + x <= y for v, x, y in zip(offset, shape, image.shape)))

    crop_idx = tuple(slice(start, size + start) for start, size in zip(offset, shape))

    return Image(image[crop_idx], image.spacing)


def noisy(image, noise_type):
    """
    Parameters
    ----------
    image :
        Input image data. Will be converted to float.
    noise_type : str
        One of the following strings, selecting the type of noise to add:

        'gauss'     Gaussian-distributed additive noise.
        'poisson'   Poisson-distributed noise generated from the data.
        's&p'       Replaces random pixels with 0  or 1.
        'speckle'   Multiplicative noise using out = image + n*image,where
                    n is uniform noise with specified mean & variance.
    """
    assert isinstance(image, Image)
    assert image.ndim < 4
    spacing = image.spacing

    if noise_type == "gauss":
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        return Image(image + gauss, spacing)
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return Image(out, spacing)
    elif noise_type == "poisson":
        vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
        return Image(np.random.poisson(image * vals) / float(vals), spacing)
    elif noise_type == "speckle":
        gauss = np.random.standard_normal(image.shape).reshape(image.shape)
        return Image(image + image * gauss, spacing)


def enhance_contrast(image, percent_saturated=0.3, out_type=np.uint8):
    """
    Performs historgram stretching (not equalization), with a given percentage
    :param percent_saturated of pixels saturated in the output.

    :param image: an Image object
    :param percent_saturated: Percentage value of saturated pixels. Defaults to 0.3
    :param out_type: The type of the output image. The default is 8-bit uint
    :return: an Image with intensity values rescaled to the whole dynamic range
    """

    assert isinstance(image, Image)

    percent_saturated /= 100

    spacing = image.spacing

    if out_type == np.uint8:
        out_max = 255
        out_min = 0
    else:
        raise ValueError("Not supported output type {}".format(out_type))

    # Get Input Image Min/Max from histogram
    histogram, bin_edges = np.histogram(image, bins=250, density=True)
    cumulative = np.cumsum(histogram * np.diff(bin_edges))

    in_max = bin_edges[1:][cumulative >= 1.0 - percent_saturated].min()

    to_zero = cumulative <= percent_saturated
    if not np.any(to_zero):
        in_min = image.min()
    else:
        in_min = bin_edges[1:][to_zero].max()

    # Trim and rescale
    image = np.clip(image, in_min, in_max)
    image *= (out_max-out_min)/image.max()
    return Image(image.astype(out_type), spacing)


def rescale_to_8_bit(image):
    """
    Converts an Image into 8-bit (typically for saving)
    :param image: an Image object
    :return: a 8-bit version of the Image
    """
    assert isinstance(image, Image)
    return Image((image*(255.0/image.max())).astype(np.uint8), image.spacing)




def flip_image(image):

    assert isinstance(image, Image)

    indexer = (np.s_[::-1],) * image.ndim

    return Image(image[indexer], image.spacing)


def translate_image(image, shift):
    """
    Apply a circular shift to an image

    :param image: An Image object
    :param shift: The shift as a single numeric value
    :return: returns the translated image.
    """
    fft_image = np.fft.fftshift(np.fft.fft2(image))

    shape = fft_image.shape
    axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
    axes = (i / (2 * i.max()) for i in axes)
    y, x = np.meshgrid(*axes)

    xx = np.zeros(fft_image.shape, dtype=np.complex64)
    xx.real[:] = np.cos(2 * np.pi * shift * x)
    xx.imag[:] = np.sin(-2 * np.pi * shift * x)

    yy = np.zeros(fft_image.shape, dtype=np.complex64)
    yy.real[:] = np.cos(2 * np.pi * shift * y)
    yy.imag[:] = np.sin(-2 * np.pi * shift * y)

    multiplier = xx * yy

    result = np.abs(np.fft.ifftn(fft_image * multiplier).real)

    return Image(result, image.spacing)

def maximum_projection(image, axis=0):
    """ Generate a maximum projection image along an axis
    
    :param image: an image
    :type image: Image
    :param axis: the axis on which the projeciton is to be calculated, defaults to 0
    :type axis: int, optional
    :return: a maximum projection image, with one dimension less thatn the input image
    :rtype: Image
    """
    assert isinstance(image, Image)
    spacing = (image.spacing[s] for s in filter(lambda x : x != axis, range(image.ndim)))
    return  Image(np.amax(image, axis=axis), spacing)
