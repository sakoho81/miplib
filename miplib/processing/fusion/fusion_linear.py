import numpy as np
import miplib.processing.deconvolution.wiener as wiener

from miplib.data.containers.image_data import ImageData


def wiener_fusion(data, options, gate=0, scale=100, views=None):
    assert isinstance(data, ImageData)

    if views is None:
        views = list(range(data.get_number_of_images("registered")))

    result = np.zeros(data.get_image_size(), dtype=np.float32)
    for idx in views:
        data.set_active_image(idx, gate, scale, "registered")
        image = data.get_image()
        data.set_active_image(idx, gate, scale, "psf")
        psf = data.get_image()

        result += wiener.wiener_deconvolution(image, psf,
                                              snr=options.wiener_snr, add_pad=options.block_pad)
    return result
