from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection
import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.processing.image as imops
import miplib.analysis.resolution.analysis as frc_analysis
import miplib.analysis.resolution.fourier_ring_correlation as frc

from .psfgen import PsfFromFwhm


def generate_frc_based_psf(image, args):
    data = FourierCorrelationDataCollection()

    image1, image2 = imops.checkerboard_split(image)
    iterator = iterators.FourierRingIterator(image1.shape, args.d_bin)
    frc_task = frc.FRC(image1, image2, iterator)
    data[0] = frc_task.execute()

    analyzer = frc_analysis.FourierCorrelationAnalysis(data, image1.spacing[0], args)
    fwhm = [analyzer.execute()[0].resolution['resolution'], ] * 2

    psf_generator = PsfFromFwhm(fwhm)

    if image.ndim == 2:
        return psf_generator.xy()
    else:
        return psf_generator.volume()

