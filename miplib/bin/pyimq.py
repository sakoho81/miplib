#!/usr/bin/env python
# -*- python -*-

"""
File:        pyimq.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
Image quality ranking tool for microscopy datasets. Computes multiple quality
metrics (spatial entropy, Brenner gradient, spectral moments, power spectrum
statistics) to rank images by focus quality and detail content.
"""

import datetime
import json
import sys

import pandas as pd

from miplib.analysis.image_quality.filters import QualityFilterOptions
from miplib.analysis.image_quality.image_quality_ranking import evaluate_image_quality
from miplib.data.io import read
from miplib.ui.cli import miplib_entry_point_options
from miplib.utils.dataclasses import options_from_dict


def main():
    """
    Image quality ranking tool.
    """
    options = miplib_entry_point_options.get_quality_options(sys.argv[1:])
    input_path = options.input

    filter_options = options_from_dict(
        QualityFilterOptions,
        json.loads(options.options) if options.options else None,
    )

    # Auto-detect file vs directory
    if input_path.is_file():
        # Single file mode: print to stdout
        image = read.get_image(str(input_path), channel=options.rgb_channel)
        metrics = evaluate_image_quality(image, filter_options)

        print(f"Image: {input_path.name}")
        print(f"Shape: {image.shape}")
        print("\nSpatial Measures:")
        print(f"  Entropy: {metrics.entropy:.4f}")
        print(f"  Brenner: {metrics.brenner:.2f}")
        print("\nFrequency Domain Measures:")
        print(f"  Spectral Moments: {metrics.spectral_moments:.4f}")
        print(f"  Power Mean: {metrics.power_stats.mean:.2e}")
        print(f"  Power Std: {metrics.power_stats.std:.2e}")
        print(f"  Power Entropy: {metrics.power_stats.entropy:.4f}")
        print(f"  Threshold Freq: {metrics.power_stats.threshold_freq:.2e} Hz")
        print(f"  High Freq Power: {metrics.power_stats.power_at_high_freq:.2e}")
        print(f"  Skew: {metrics.power_stats.skew:.4f}")
        print(f"  Kurtosis: {metrics.power_stats.kurtosis:.4f}")

    elif input_path.is_dir():
        # Directory mode: write CSV
        # Determine output path
        if options.output:
            output_path = options.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = input_path / f"{timestamp}_quality_metrics.csv"

        # Collect all image files
        image_files = []
        for ext in ("*.jpg", "*.tif", "*.tiff", "*.png"):
            image_files.extend(input_path.glob(ext))
        image_files.sort()

        if not image_files:
            print(f"No images found in {input_path}")
            sys.exit(1)

        print(f"Analyzing {len(image_files)} images...")

        # Analyze each image
        results = []
        for image_path in image_files:
            try:
                image = read.get_image(str(image_path), channel=options.rgb_channel)
                metrics = evaluate_image_quality(image, filter_options)
                results.append(
                    {
                        "Filename": image_path.name,
                        "Entropy": metrics.entropy,
                        "Brenner": metrics.brenner,
                        "SpectralMoments": metrics.spectral_moments,
                        "PowerMean": metrics.power_stats.mean,
                        "PowerStd": metrics.power_stats.std,
                        "PowerEntropy": metrics.power_stats.entropy,
                        "ThresholdFreq": metrics.power_stats.threshold_freq,
                        "HighFreqPower": metrics.power_stats.power_at_high_freq,
                        "Skew": metrics.power_stats.skew,
                        "Kurtosis": metrics.power_stats.kurtosis,
                    }
                )
                print(f"  ✓ {image_path.name}")
            except Exception as e:
                print(f"  ✗ {image_path.name}: {e}")

        # Write results
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

    else:
        print(f"Error: {input_path} is neither a file nor a directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
