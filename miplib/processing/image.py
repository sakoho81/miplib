import logging

import numpy as np
from scipy.ndimage import interpolation

from miplib.data.containers.image import Image
from miplib.processing import fftutils

from . import ndarray

logger = logging.getLogger(__name__)


def zoom_to_isotropic_spacing(image: Image, order: int = 3) -> Image:
    """Resize an Image to isotropic pixel spacing."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    spacing = image.spacing
    old_shape = image.shape
    min_spacing = min(spacing)
    zoom = tuple(pixel_spacing / min_spacing for pixel_spacing in spacing)
    new_shape = tuple(
        round(pixels * dim_zoom)
        for (pixels, dim_zoom) in zip(old_shape, zoom, strict=False)
    )

    if new_shape == old_shape:
        return image
    else:
        return resize(image, new_shape, order)


def zoom_to_spacing(image: Image, spacing: tuple[float, ...], order: int = 3) -> Image:
    """Resample an Image to the given pixel spacing."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if image.ndim != len(spacing):
        raise ValueError(
            f"spacing length {len(spacing)} does not match image ndim {image.ndim}"
        )

    zoom = tuple(i / j for i, j in zip(image.spacing, spacing, strict=False))
    logger.debug("The zoom is %s", zoom)

    array = interpolation.zoom(image, zoom, order=order)

    return Image(array, spacing)


def resize(image: Image, size: tuple[int, ...], order: int = 3) -> Image:
    """Resize the image using spline interpolation.

    :param image:   The Image object.
    :param size:    A tuple of new image dimensions.
    :param order:   The interpolation type defined as order of the b-spline.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    zoom = [float(a) / b for a, b in zip(size, image.shape, strict=False)]
    logger.debug("The zoom is %s", zoom)

    array = interpolation.zoom(image, tuple(zoom), order=order)
    spacing = tuple(i / j for i, j in zip(image.spacing, zoom, strict=False))

    return Image(array, spacing)


def zero_pad_to_shape(image: Image, shape: tuple[int, ...]) -> Image:
    """Apply zero padding to cast an Image into the given shape.

    Padding is applied evenly on all sides of the image.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    if image.shape == shape:
        return image
    else:
        return Image(ndarray.expand_to_shape(image, shape), image.spacing)


def zero_pad_to_matching_shape(image1: Image, image2: Image) -> tuple[Image, Image]:
    """Apply zero padding to make the size of two Images match."""
    if not isinstance(image1, Image):
        raise TypeError(f"Expected Image for image1, got {type(image1).__name__}")
    if not isinstance(image2, Image):
        raise TypeError(f"Expected Image for image2, got {type(image2).__name__}")

    shape = tuple(max(x, y) for x, y in zip(image1.shape, image2.shape, strict=False))

    if any(map(lambda x, y: x != y, image1.shape, shape)):
        image1 = zero_pad_to_shape(image1, shape)
    if any(map(lambda x, y: x != y, image2.shape, shape)):
        image2 = zero_pad_to_shape(image2, shape)

    return image1, image2


def remove_zero_padding(image: Image, shape: tuple[int, ...]) -> Image:
    """Remove zero padding to restore an Image to the given shape."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if len(shape) != image.ndim:
        raise ValueError(
            f"shape length {len(shape)} does not match image ndim {image.ndim}"
        )

    return Image(ndarray.contract_to_shape(image, shape), image.spacing)


def checkerboard_split(
    image: Image, disable_3d_sum: bool = False
) -> tuple[Image, Image]:
    """Split an image in two using a checkerboard subsampling pattern.

    For 2D images the even/odd pixel pairs are separated. For 3D the default
    behaviour sums spatially adjacent pairs to reduce noise (set
    ``disable_3d_sum=True`` to get a pure subsampling instead).
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, odd_index[1]]
        image2 = image[even_index[0], :][:, even_index[1]]
    else:
        if disable_3d_sum:
            image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]]
            image2 = image[even_index[0], :, :][:, even_index[1], :][
                :, :, even_index[2]
            ]
        else:
            image1 = (
                image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][
                    :, :, odd_index[2]
                ]
                + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][
                    :, :, odd_index[2]
                ]
            )

            image2 = (
                image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][
                    :, :, even_index[2]
                ]
                + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][
                    :, :, even_index[2]
                ]
            )

    return image1, image2


def reverse_checkerboard_split(
    image: Image, disable_3d_sum: bool = False
) -> tuple[Image, Image]:
    """Split an image in two using the reverse checkerboard pattern.

    Like :func:`checkerboard_split` but with odd/even index roles swapped.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    if image.ndim == 2:
        image1 = image[odd_index[0], :][:, even_index[1]]
        image2 = image[even_index[0], :][:, odd_index[1]]
    else:
        if disable_3d_sum:
            image1 = image[odd_index[0], :, :][:, odd_index[1], :][:, :, even_index[2]]
            image2 = image[even_index[0], :, :][:, even_index[1], :][:, :, odd_index[2]]
        else:
            image1 = (
                image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][
                    :, :, even_index[2]
                ]
                + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][
                    :, :, odd_index[2]
                ]
            )

            image2 = (
                image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][
                    :, :, odd_index[2]
                ]
                + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][
                    :, :, even_index[2]
                ]
            )

    return image1, image2


def summed_checkerboard_split(image: Image) -> tuple[Image, Image]:
    """Split an image using diagonal pixel pairs from each 2×2 block.

    In 2D each output contains the sum of one diagonal pair per 2×2 block.
    In 3D the split uses orthogonal diagonal groups (never adjacent pixels).
    The output spacing is doubled relative to the input to reflect the
    effective pixel pitch after subsampling.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    shape = image.shape
    odd_index = [np.arange(1, shape[i], 2) for i in range(len(shape))]
    even_index = [np.arange(0, shape[i], 2) for i in range(len(shape))]

    if image.ndim == 2:
        image1 = (
            image[odd_index[0], :][:, odd_index[1]]
            + image[even_index[0], :][:, even_index[1]]
        )
        image2 = (
            image[odd_index[0], :][:, even_index[1]]
            + image[even_index[0], :][:, odd_index[1]]
        )

        image1.spacing = tuple(i * 2 for i in image.spacing)
        image2.spacing = image1.spacing
    else:
        image1 = (
            image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][
                :, :, odd_index[2]
            ]
            + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][
                :, :, odd_index[2]
            ]
            + image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][
                :, :, even_index[2]
            ]
            + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][
                :, :, even_index[2]
            ]
        )

        image2 = (
            image.astype(np.uint32)[even_index[0], :, :][:, odd_index[1], :][
                :, :, even_index[2]
            ]
            + image.astype(np.uint32)[odd_index[0], :, :][:, odd_index[1], :][
                :, :, even_index[2]
            ]
            + image.astype(np.uint32)[even_index[0], :, :][:, even_index[1], :][
                :, :, odd_index[2]
            ]
            + image.astype(np.uint32)[odd_index[0], :, :][:, even_index[1], :][
                :, :, odd_index[2]
            ]
        )

        image1.spacing = tuple(i * 2 for i in image.spacing)
        image2.spacing = image1.spacing

    return image1, image2


def zero_pad_to_cube(image: Image) -> Image:
    """Apply zero padding to cast an image into a cubic shape.

    Returns the original image unchanged if it is already cubic.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    original_shape = image.shape
    nmax = max(original_shape)
    square_shape = (nmax,) * image.ndim
    if square_shape != original_shape:
        return zero_pad_to_shape(image, square_shape)
    else:
        return image


def crop_to_largest_square(image: Image, physical_dims: bool = False) -> Image:
    """Crop an image to the largest square shape that fits inside it.

    :param image:         an Image object
    :param physical_dims: if True, compute the square in physical units
                          rather than pixels
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    if physical_dims:
        shape_real = [x * y for x, y in zip(image.shape, image.spacing, strict=False)]
        min_shape_real = (min(*shape_real),) * image.ndim
        min_shape_px = tuple(
            x / y for x, y in zip(min_shape_real, image.spacing, strict=False)
        )
    else:
        min_shape_px = (min(*image.shape),) * image.ndim

    return remove_zero_padding(image, min_shape_px)


def crop_to_shape(
    image: Image, shape: tuple[int, ...], offset: tuple[int, ...]
) -> Image:
    """Crop image to the given shape starting at offset.

    :param image:   An N-dimensional Image to be cropped.
    :param shape:   The desired output shape; each dimension must fit within
                    the image at the given offset.
    :param offset:  Per-axis start indices for the crop.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if image.ndim != len(shape) or image.ndim != len(offset):
        raise ValueError("image.ndim, shape, and offset must all have the same length")
    if not all(v + x <= y for v, x, y in zip(offset, shape, image.shape, strict=False)):
        raise ValueError("crop region (offset + shape) extends beyond image bounds")

    crop_idx = tuple(
        slice(start, size + start) for start, size in zip(offset, shape, strict=False)
    )

    return Image(image[crop_idx], image.spacing)


def noisy(image: Image, noise_type: str) -> Image:
    """Add synthetic noise to an image.

    Parameters
    ----------
    image :
        Input image data. Will be converted to float.
    noise_type : str
        One of ``'gauss'``, ``'poisson'``, ``'s&p'``, or ``'speckle'``.
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if image.ndim >= 4:
        raise ValueError(f"noisy() supports up to 3D images, got {image.ndim}D")
    spacing = image.spacing

    if noise_type == "gauss":
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, image.shape)
        gauss = gauss.reshape(image.shape)
        return Image(image + gauss, spacing)
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return Image(out, spacing)
    elif noise_type == "poisson":
        vals = 2 ** np.ceil(np.log2(len(np.unique(image))))
        return Image(np.random.poisson(image * vals) / float(vals), spacing)
    elif noise_type == "speckle":
        gauss = np.random.standard_normal(image.shape).reshape(image.shape)
        return Image(image + image * gauss, spacing)
    else:
        raise ValueError(f"Unknown noise_type {noise_type!r}")


def enhance_contrast(
    image: Image,
    percent_saturated: float = 0.3,
    out_type: type = np.uint8,
) -> Image:
    """Perform histogram stretching with a given saturation percentage.

    :param image:              an Image object
    :param percent_saturated:  percentage of pixels to saturate (default 0.3)
    :param out_type:           output dtype (only np.uint8 supported)
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")

    percent_saturated /= 100

    spacing = image.spacing

    if out_type == np.uint8:
        out_max = 255
        out_min = 0
    else:
        raise ValueError(f"Not supported output type {out_type}")

    histogram, bin_edges = np.histogram(image, bins=250, density=True)
    cumulative = np.cumsum(histogram * np.diff(bin_edges))

    in_max = bin_edges[1:][cumulative >= 1.0 - percent_saturated].min()

    to_zero = cumulative <= percent_saturated
    if not np.any(to_zero):
        in_min = image.min()
    else:
        in_min = bin_edges[1:][to_zero].max()

    image = np.clip(image, in_min, in_max)
    image *= (out_max - out_min) / image.max()
    return Image(image.astype(out_type), spacing)


def rescale_to_8_bit(image: Image) -> Image:
    """Convert an Image to 8-bit by scaling to the full [0, 255] range."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    return Image((image * (255.0 / image.max())).astype(np.uint8), image.spacing)


def translate_image(image: Image, shift: tuple[float, ...]) -> Image:
    """Apply a circular shift to an image using Fourier phase multiplication.

    The shift is applied along all axes simultaneously. Sub-pixel shifts are
    supported. The image is assumed to be periodic (circular boundary).

    :param image: an Image object (2D or 3D)
    :param shift: per-axis shift in pixels, e.g. ``(dy, dx)`` for 2D
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if len(shift) != image.ndim:
        raise ValueError(
            f"shift length {len(shift)} does not match image ndim {image.ndim}"
        )

    F = fftutils.fft(image, window=None)

    # Build the phase ramp in the DC-centred frequency domain.
    # fftfreq returns frequencies in cycles/sample; after fftshift the DC
    # component is at the centre, matching the layout produced by fftutils.fft.
    freqs = [np.fft.fftshift(np.fft.fftfreq(s)) for s in image.shape]
    mesh = np.meshgrid(*freqs, indexing="ij")
    phase_ramp = sum(s * f for s, f in zip(shift, mesh, strict=True))
    F *= np.exp(-2j * np.pi * phase_ramp)

    result = fftutils.ifft(F)
    return Image(result.real, image.spacing)


def maximum_projection(image: Image, axis: int = 0) -> Image:
    """Generate a maximum intensity projection along the given axis.

    :param image: an Image
    :param axis:  the axis along which the projection is calculated (default 0)
    :return:      a projection image with one fewer dimension than the input
    """
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    spacing = tuple(
        image.spacing[s] for s in filter(lambda x: x != axis, range(image.ndim))
    )
    return Image(np.amax(image, axis=axis), spacing)
