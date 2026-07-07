import numpy as np

import miplib.processing.deconvolution.wiener_cuda as wiener
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType


def wiener_fusion(
    data: ImageData,
    options,
    gate: int = 0,
    scale: int = 100,
    views: list[int] | None = None,
) -> np.ndarray:
    """Fuse registered views by Wiener-deconvolving each with its PSF.

    Returns the sum of individually deconvolved views.
    """
    if not isinstance(data, ImageData):
        raise TypeError(f"Expected ImageData, got {type(data).__name__}")

    if views is None:
        views = list(range(data.get_number_of_images(ImageType.REGISTERED)))

    key0 = ImageKey(ImageType.REGISTERED, 0, gate, scale)
    result = np.zeros(data.get_image_size(key0), dtype=np.float32)
    for idx in views:
        key_reg = ImageKey(ImageType.REGISTERED, idx, gate, scale)
        image = data.get_image(key_reg)
        key_psf = ImageKey(ImageType.PSF, idx, gate, scale)
        psf = data.get_image(key_psf)

        result += wiener.wiener_deconvolution(
            image, psf, snr=options.wiener_snr, add_pad=options.block_pad
        )
    return result
