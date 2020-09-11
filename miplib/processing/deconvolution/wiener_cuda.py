import numpy as np
import cupy as cp
from cupyx.scipy.fftpack import fftn, ifftn, get_fft_plan
from numpy.fft import fftshift

from miplib.data.containers.image import Image
import miplib.processing.image as imops
import miplib.processing.ndarray as arrayops


def wiener_deconvolution(image, psf, snr=30, add_pad=0):
    assert isinstance(image, Image)
    assert isinstance(psf, Image)

    image_s = Image(image.copy(), image.spacing)
    orig_shape = image.shape

    if image.ndim != psf.ndim:
        raise ValueError("Image and psf dimensions do not match")

    if psf.spacing != image.spacing:
        psf = imops.zoom_to_spacing(psf, image.spacing)

    if add_pad != 0:
        new_shape = list(i + 2*add_pad for i in image_s.shape)
        image_s = imops.zero_pad_to_shape(image_s, new_shape)

    if psf.shape != image_s.shape:
        psf = imops.zero_pad_to_shape(psf, image_s.shape)

    psf /= psf.max()
    psf = fftshift(psf)

    psf_dev = fftn(cp.asarray(psf), overwrite_x=True)

    stream = cp.cuda.stream.Stream()
    with stream:
        psf_abs = cp.abs(psf_dev)**2
        below = cp.asnumpy(psf_dev)
    stream.synchronize()

    with stream:
        psf_abs /= (psf_abs + snr)
        image_dev = fftn(cp.asarray(image_s), overwrite_x=True)
    stream.synchronize()

    above = cp.asnumpy(psf_abs)
    wiener = arrayops.safe_divide(above, below)

    psf_dev = cp.asarray(wiener)
    image_dev *= psf_dev

    result = np.abs(cp.asnumpy(ifftn(image_dev, overwrite_x=True)).real)
    result = Image(result, image.spacing)

    return imops.remove_zero_padding(result, orig_shape)


