import math
from psf import psf, _psf

from miplib.analysis.resolution import fourier_ring_correlation as frc
from miplib.data.containers.image import Image


class PsfFromFwhm(object):

    def __init__(self, fwhm, shape=(128, 128), dims=(4., 4.)):
        assert isinstance(fwhm, list)

        if len(fwhm) == 1:
            print ("Only one resolution value given. Assuming the same"
                   " resolution for the axial direction.")
            fwhm = [fwhm, ] * 2

        self.shape = int(shape[0]), int(shape[1])
        self.dims = psf.Dimensions(px=shape, um=(float(dims[0]), float(dims[1])))

        self.spacing = list(x/y for x, y in zip(self.dims.um, self.dims.px))
        self.sigma_px = list(x/(2*math.sqrt(2*math.log(2))*y) for x, y in zip(fwhm, self.spacing))

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
    fwhm = [frc.calculate_single_image_frc(image, args).resolution["resolution"], ] * 2
    psf_generator = PsfFromFwhm(fwhm)

    if image.ndim == 2:
        return psf_generator.xy()
    else:
        return psf_generator.volume()