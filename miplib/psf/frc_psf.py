from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection
import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.processing.image as imops
import miplib.analysis.resolution.analysis as frc_analysis
import miplib.analysis.resolution.fourier_ring_correlation as frc

from .psfgen import PsfFromFwhm


def generate_frc_based_psf(image, args):
    fwhm = [frc.calculate_single_image_frc(image, args).resolution["resolution"], ] * 2
    psf_generator = PsfFromFwhm(fwhm)

    if image.ndim == 2:
        return psf_generator.xy()
    else:
        return psf_generator.volume()

