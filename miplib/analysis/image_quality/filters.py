from dataclasses import dataclass

import numpy as np
from scipy import stats

from miplib.data.containers.image import Image
from miplib.processing import fftutils
from miplib.processing import image as imutils
from miplib.processing.segmentation import masking

from . import utils


@dataclass
class QualityFilterOptions:
    """Options for image quality filter functions."""

    normalize_power: bool = False
    use_mask: bool = True
    invert_mask: bool = False
    power_threshold: float = 0.4
    spatial_threshold: int = 80


@dataclass
class PowerSpectrumStats:
    """Statistical measures computed from the high-frequency tail of a power spectrum."""

    mean: float
    std: float
    entropy: float
    threshold_freq: float
    power_at_high_freq: float
    skew: float
    kurtosis: float
    mean_bin: float


def local_image_quality(
    image: Image,
    options: QualityFilterOptions | None = None,
    kernel_size: int | list[int] = 100,
) -> float:
    """Shannon entropy of the image (optionally restricted to detailed regions).

    Args:
        image: Input image
        options: Quality filter options (controls masking behavior)
        kernel_size: Smoothing kernel size for mask generation

    Returns:
        Shannon entropy value
    """
    if options is None:
        options = QualityFilterOptions()

    data = image[:]

    if options.use_mask:
        if isinstance(kernel_size, list):
            sizes = kernel_size
        else:
            sizes = [kernel_size] * image.ndim

        mask = masking.make_local_intensity_based_mask(
            image,
            threshold=options.spatial_threshold,
            kernel_size=sizes[0] if len(sizes) == 1 else sizes,
            invert=options.invert_mask,
        )
        data = data[np.nonzero(mask)]

    return utils.calculate_entropy(data)


def frequency_quality(
    image: Image,
    options: QualityFilterOptions | None = None,
) -> PowerSpectrumStats:
    """Statistical quality parameters from the high-frequency tail of the power spectrum.

    Args:
        image: Input image
        options: Quality filter options (controls normalization, masking, etc.)

    Returns:
        PowerSpectrumStats with mean, std, entropy, threshold frequency, etc.
    """
    if options is None:
        options = QualityFilterOptions()

    data = image[:]
    spacing = image.spacing[0]

    ps = fftutils.power_spectrum(data)

    if options.normalize_power:
        dims = data.shape[0] * data.shape[1]
        mean = np.mean(data)
        ps /= dims * mean

    profile = fftutils.radial_average(ps)
    freq = fftutils.frequency_axis(len(profile), spacing)

    hf_mask = freq > options.power_threshold * freq.max()
    hf_sum = profile[hf_mask]

    if len(hf_sum) == 0:
        return PowerSpectrumStats(
            mean=0.0,
            std=0.0,
            entropy=0.0,
            threshold_freq=0.0,
            power_at_high_freq=0.0,
            skew=0.0,
            kurtosis=0.0,
            mean_bin=0.0,
        )

    f_th = freq[hf_mask][-utils.analyze_accumulation(hf_sum, 0.2)]
    mean_val = float(np.mean(hf_sum))
    std_val = float(np.std(hf_sum))
    entropy_val = utils.calculate_entropy(hf_sum)
    nm_th = 1.0e9 / f_th if f_th > 0 else 0.0
    pw_at_high_f = float(np.mean(profile[freq > 0.9 * freq.max()]))
    skew_val = float(stats.skew(np.log(hf_sum))) if np.all(hf_sum > 0) else 0.0
    kurtosis_val = float(stats.kurtosis(hf_sum))
    mean_bin_val = float(np.mean(hf_sum[0:5]))

    return PowerSpectrumStats(
        mean=mean_val,
        std=std_val,
        entropy=entropy_val,
        threshold_freq=nm_th,
        power_at_high_freq=pw_at_high_f,
        skew=skew_val,
        kurtosis=kurtosis_val,
        mean_bin=mean_bin_val,
    )


def spectral_moments(
    image: Image,
    options: QualityFilterOptions | None = None,
) -> float:
    """Spectral moments autofocus metric (Firestone et al., 1991).

    Args:
        image: Input image
        options: Quality filter options

    Returns:
        Spectral moments value
    """
    if options is None:
        options = QualityFilterOptions()

    data = image[:]

    ps = fftutils.power_spectrum(data)
    profile = fftutils.radial_average(ps)

    profile_pct = profile / (profile.sum() / 100)
    bin_index = np.arange(1, len(profile_pct) + 1)

    return float((profile_pct * np.log10(bin_index)).sum())


def brenner_quality(image: Image) -> float:
    """Brenner autofocus metric (Brenner et al., 1976).

    Args:
        image: Input image

    Returns:
        Brenner quality value (sum of squared differences 2 pixels apart)
    """
    data = imutils.crop_to_largest_square(image)[:]
    rows = data.shape[0]
    columns = data.shape[1] - 2
    temp = np.zeros((rows, columns))
    temp[:] = (data[:, 0:-2] - data[:, 2:]) ** 2
    return float(temp.sum())
