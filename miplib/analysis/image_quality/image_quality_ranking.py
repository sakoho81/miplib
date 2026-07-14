from pathlib import Path

import pandas as pd

import miplib.analysis.resolution.fourier_ring_correlation as frc
from miplib.data.io import read

from . import filters


def evaluate_image_quality(image, options):
    """
    Calculate quality features sof a single image
    """

    # Run spatial domain analysis
    task = filters.LocalImageQuality(image, options)
    task.set_smoothing_kernel_size(100)
    entropy = task.calculate_image_quality()

    # Run frequency domain analysis
    task2 = filters.FrequencyQuality(image, options)
    results = task2.analyze_power_spectrum()

    task3 = filters.SpectralMoments(image, options)
    moments = task3.calculate_spectral_moments()

    task4 = filters.BrennerImageQuality(image, options)
    brenner = task4.calculate_brenner_quality()

    # Save results
    results.insert(0, moments)
    results.insert(0, brenner)
    results.insert(0, entropy)

    return results


def batch_evaluate_image_quality(path, options):
    """
    Batch calculate quality features for images in a directory
    :param options: options for the quality ranking scripts, as in miplib/ui/image_quality_options.py
    :parame path:   directory that contains the images to be analyzed
    """

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

    for idx, image_entry in enumerate(Path(path).iterdir()):
        if not image_entry.is_file():
            continue
        image_name = image_entry.name
        if options.file_filter is not None and options.file_filter not in image_name:
            continue
        if image_entry.suffix not in (".jpg", ".tif", ".tiff"):
            continue
        # ImageJ files have particular TIFF tags that can be processed correctly
        # with the options.imagej switch
        image = read.get_image(image_entry, channel=options.rgb_channel)

        # Only grayscale images are processed. If the input is an RGB image,
        # a channel can be chosen for processing.
        results = evaluate_image_quality(image, options)
        results.insert(0, str(image_entry))

        # Add resolution value to the end
        results.append(
            frc.calculate_single_image_frc(image, options).resolution["resolution"]
        )

        df.loc[idx] = results

        print(f"Done analyzing {image_name}")

    return df
