import numpy as np
import cupy as cp
from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
from numpy.fft import fftshift

from miplib.data.containers.image import Image
import miplib.processing.image as imops
import miplib.processing.ndarray as arrayops


def wiener_deconvolution(image, psf, snr=30, add_pad=0):
    """ A GPU accelerated implementation of a linear Wiener filter. Some effort is made
    to allow processing even relatively large images, but some kind of block-based processing
     (as in the RL implementation) may be required in some cases."""
    assert isinstance(image, Image)
    assert isinstance(psf, Image)

    image_s = Image(image.copy(), image.spacing)
    orig_shape = image.shape

    if image.ndim != psf.ndim:
        raise ValueError("Image and psf dimensions do not match")

    if psf.spacing != image.spacing:
        psf = imops.zoom_to_spacing(psf, image.spacing)

    if add_pad != 0:
        new_shape = list(i + 2 * add_pad for i in image_s.shape)
        image_s = imops.zero_pad_to_shape(image_s, new_shape)

    if psf.shape != image_s.shape:
        psf = imops.zero_pad_to_shape(psf, image_s.shape)

    psf /= psf.max()
    psf = fftshift(psf)

    psf_dev = cp.asarray(psf.astype(np.complex64))
    with get_fft_plan(psf_dev):
        psf_dev = fftn(psf_dev, overwrite_x=True)

    below = cp.asnumpy(psf_dev)
    psf_abs = cp.abs(psf_dev) ** 2
    psf_abs /= (psf_abs + snr)
    above = cp.asnumpy(psf_abs)
    psf_abs = None
    psf_dev = None

    image_dev = cp.asarray(image_s.astype(np.complex64))
    with get_fft_plan(image_dev):
        image_dev = fftn(image_dev, overwrite_x=True)

    wiener_dev = cp.asarray(arrayops.safe_divide(above, below))

    image_dev *= wiener_dev

    result = cp.asnumpy(cp.abs(ifftn(image_dev, overwrite_x=True)).real)
    result = Image(result, image.spacing)

    return imops.remove_zero_padding(result, orig_shape)

