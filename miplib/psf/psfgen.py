import logging
import math

from psf import _psf, psf  # type: ignore[attr-defined]

from miplib.analysis.resolution import fourier_ring_correlation as frc
from miplib.data.containers.image import Image

logger = logging.getLogger(__name__)


class PsfFromFwhm:
    """Generate a Gaussian PSF from FWHM values.

    FWHM in µm is converted to sigma in pixels using the pixel spacing
    computed from the field-of-view dimensions and image shape.
    """

    def __init__(
        self,
        fwhm: list[float],
        shape: tuple[int, int] = (128, 128),
        dims: tuple[float, float] = (4.0, 4.0),
    ) -> None:
        if not isinstance(fwhm, list):
            raise TypeError(f"fwhm must be a list, got {type(fwhm).__name__}")

        if len(fwhm) == 1:
            logger.info(
                "Only one resolution value given. Assuming the same "
                "resolution for the axial direction."
            )
            fwhm = [fwhm[0], fwhm[0]]

        self.shape = int(shape[0]), int(shape[1])
        self.dims = psf.Dimensions(px=shape, um=(float(dims[0]), float(dims[1])))

        self.spacing = [x / y for x, y in zip(self.dims.um, self.dims.px, strict=False)]
        self.sigma_px = [
            x / (2 * math.sqrt(2 * math.log(2)) * y)
            for x, y in zip(fwhm, self.spacing, strict=False)
        ]

        self.data = _psf.gaussian2d(self.dims.px, self.sigma_px)

    def xy(self):
        """Return a z slice of the PSF with rotational symmetries applied."""
        data = psf.mirror_symmetry(_psf.zr2zxy(self.data))
        spacing = (self.spacing[1], self.spacing[1])

        center = self.shape[0] - 1
        return Image(data[center], spacing)

    def volume(self):
        """Return a 3D volume of the PSF with all symmetries applied.

        The shape of the returned array is
            (2*self.shape[0]-1, 2*self.shape[1]-1, 2*self.shape[1]-1)

        """
        data = psf.mirror_symmetry(_psf.zr2zxy(self.data))
        spacing = (self.spacing[0], self.spacing[1], self.spacing[1])

        return Image(data, spacing)


def generate_frc_based_psf(image, args):
    fwhm = [
        frc.calculate_single_image_frc(image, args).resolution["resolution"],
    ] * 2
    psf_generator = PsfFromFwhm(fwhm)

    if image.ndim == 2:
        return psf_generator.xy()
    else:
        return psf_generator.volume()
