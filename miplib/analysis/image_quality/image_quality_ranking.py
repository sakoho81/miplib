from dataclasses import dataclass
from pathlib import Path

from miplib.analysis.resolution import fourier_ring_correlation as frc
from miplib.data.containers.image import Image
from miplib.data.io import read

from . import filters


@dataclass
class ImageQualityMetrics:
    """Complete set of image quality metrics."""

    entropy: float
    brenner: float
    spectral_moments: float
    power_stats: filters.PowerSpectrumStats


def evaluate_image_quality(
    image: Image, options: filters.QualityFilterOptions | None = None
) -> ImageQualityMetrics:
    """Calculate quality features of a single image.

    Args:
        image: Input image
        options: Quality filter options

    Returns:
        ImageQualityMetrics with entropy, brenner, spectral moments, and power spectrum stats
    """
    if options is None:
        options = filters.QualityFilterOptions()

    entropy = filters.local_image_quality(image, options)
    power_stats = filters.frequency_quality(image, options)
    moments = filters.spectral_moments(image, options)
    brenner = filters.brenner_quality(image)

    return ImageQualityMetrics(
        entropy=entropy,
        brenner=brenner,
        spectral_moments=moments,
        power_stats=power_stats,
    )


def batch_evaluate_image_quality(
    path: str | Path,
    options: filters.QualityFilterOptions | None = None,
    frc_options=None,
):
    """Batch calculate quality features for images in a directory.

    Args:
        path: Directory containing images to analyze
        options: Quality filter options
        frc_options: Options for FRC resolution calculation (optional)

    Returns:
        pandas DataFrame with quality metrics for each image
    """
    import pandas as pd

    if options is None:
        options = filters.QualityFilterOptions()

    df = pd.DataFrame(
        columns=[
            "Filename",
            "tEntropy",
            "tBrenner",
            "fMoments",
            "fMean",
            "fSTD",
            "fEntropy",
            "fTh",
            "fMaxPw",
            "Skew",
            "Kurtosis",
            "MeanBin",
            "Resolution",
        ]
    )

    path_obj = Path(path) if not isinstance(path, Path) else path
    image_files: list[Path] = []
    for ext in ("*.jpg", "*.tif", "*.tiff"):
        image_files.extend(path_obj.glob(ext))
    image_files.sort()

    for idx, image_entry in enumerate(image_files):
        image = read.get_image(image_entry)
        metrics = evaluate_image_quality(image, options)

        resolution = None
        if frc_options is not None:
            resolution = frc.calculate_single_image_frc(image, frc_options).resolution[
                "resolution"
            ]

        df.loc[idx] = [
            str(image_entry),
            metrics.entropy,
            metrics.brenner,
            metrics.spectral_moments,
            metrics.power_stats.mean,
            metrics.power_stats.std,
            metrics.power_stats.entropy,
            metrics.power_stats.threshold_freq,
            metrics.power_stats.power_at_high_freq,
            metrics.power_stats.skew,
            metrics.power_stats.kurtosis,
            metrics.power_stats.mean_bin,
            resolution,
        ]

    return df
