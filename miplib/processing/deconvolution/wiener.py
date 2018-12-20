import numpy as np

from numpy.fft import fftn, ifftn, fftshift

from miplib.data.containers.image import Image
import miplib.processing.image as imops
import miplib.processing.ndarray as arrayops

#todo: Speed up with CUDA/Multithreading. Functions are ready in the ufuncs.py

def wiener_deconvolution(image, psf, snr=1e-3, add_pad=0, cuda=False):
    assert isinstance(image, Image)
    assert isinstance(psf, Image)

    image_s = Image(image.copy(), image.spacing)

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

    psf_f = fftn(fftshift(psf))

    wiener = arrayops.safe_divide(np.abs(psf_f)**2/(np.abs(psf_f)**2 + snr), psf_f)

    image_f = fftn(image_s)

    result = Image(np.abs(ifftn(image_f * wiener).real), image.spacing)

    return imops.remove_zero_padding(result, image.shape)


