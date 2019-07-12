import os
import pandas as pd

from . import filters
from miplib.data.io import read
import miplib.analysis.resolution.fourier_ring_correlation as frc

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

    df = pd.DataFrame(columns=["Filename", "tEntropy", "tBrenner", "fMoments", "fMean", "fSTD", "fEntropy",
               "fTh", "fMaxPw", "Skew", "Kurtosis", "MeanBin", "Resolution"])

    for idx, image_name in enumerate(os.listdir(path)):
        if options.file_filter is None or options.file_filter in image_name:
            real_path = os.path.join(path, image_name)
            # Only process images
            if not os.path.isfile(real_path) or not real_path.endswith((".jpg", ".tif", ".tiff", ".tif")):
                continue
            # ImageJ files have particular TIFF tags that can be processed correctly
            # with the options.imagej switch
            image = read.get_image(real_path, channel=options.rgb_channel)

            # Only grayscale images are processed. If the input is an RGB image,
            # a channel can be chosen for processing.
            results = evaluate_image_quality(image, options)
            results.insert(0, real_path)

            # Add resolution value to the end
            results.append(frc.calculate_single_image_frc(image, options).resolution["resolution"])

            df.loc[idx] = results

            print ("Done analyzing {}".format(image_name))

    return df
        