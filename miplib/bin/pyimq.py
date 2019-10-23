#!/usr/bin/env python
# -*- python -*-

"""
File:        pyimq.py
Author:      Sami Koho (sami.koho@gmail.com)

Description:
This is the main program of PyImageQualityRanking software. The
purpose of the software is to sort large microscopy image datasets
according to calculated image quality parameters. Possible
applications can be: (1) finding the best images in a dataset, or
(2) finding and discarding out-of-focus images before quantitative
analysis, for example.

The behavior of the program can be controlled by a rich command
line options interface. Please run "python pyimq.py -h" for details.

The program works in four main modes, that can be controlled by
the --mode parameter:
- file:        A single file is analyzed and the results are
            printed on the terminal screen
- directory:   All the images in a directory are analyzed and
            the results are saved in a file
- analyze:     Variables are calculated from the analysis results.
- plot:        The analysis results are ordered according to a
            selected image quality variable.
License:
The PyImageQuality software is licensed under BSD open-source license.

Copyright (c) 2015, Sami Koho, Laboratory of Biophysics, University of Turku.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided
with the distribution.

THIS SOFTWARE IS PROVIDED BY COPYRIGHT HOLDER AND CONTRIBUTORS ''AS
IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER
OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In addition to the terms of the license, we ask to acknowledge the use
of packages in scientific articles by citing the corresponding papers:

**citation here**
"""

import csv
import datetime
import os
import sys

import pandas

from miplib.analysis.image_quality import filters
from miplib.data.io import read
from miplib.ui.cli import miplib_entry_point_options
from miplib.ui.plots.image import show_pics_from_disk


def main():
    """
    The Main program of the PyImageQualityRanking software.
    """
    options = miplib_entry_point_options.get_quality_script_options(sys.argv[1:])
    path = options.working_directory
    file_path = None
    csv_data = None

    print("Mode option is %s" % options.mode)

    if "file" in options.mode:
        # In "file" mode a single file is analyzed and the various parameter
        # values are printed on screen. This functionality is provided mainly
        # for debugging purposes.
        assert options.file is not None, "You have to specify a file with a " \
                                         "--file option"
        path = os.path.join(path, options.file)
        assert os.path.isfile(path)
        image = read.get_image(path, channel=options.rgb_channel)

        print("The shape is %s" % str(image.shape))

        task = filters.LocalImageQuality(image, options)
        task.set_smoothing_kernel_size(100)
        entropy = task.calculate_image_quality()
        task2 = filters.FrequencyQuality(image, options)
        finfo = task2.analyze_power_spectrum()

        print("SPATIAL MEASURES:")
        print("The entropy value of %s is %f" % (path, entropy))
        print("ANALYSIS OF THE POWER SPECTRUM TAIL")
        print("The mean is: %e" % finfo[0])
        print("The std is: %e" % finfo[1])
        print("The entropy is %e" % finfo[2])
        print("The threshold frequency is %f Hz" % finfo[3])
        print("Power at high frequencies %e" % finfo[4])
        print("The skewness is %f" % finfo[5])
        print("The kurtosis is %f" % finfo[6])

    if "directory" in options.mode:
        # In directory mode every image in a given directory is analyzed in a
        # single run. The analysis results are saved into a csv file.

        assert os.path.isdir(path), path

        # Create output directory
        output_dir = datetime.datetime.now().strftime("%Y-%m-%d")+'_PyIQ_output'
        output_dir = os.path.join(options.working_directory, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create output file
        date_now = datetime.datetime.now().strftime("%H-%M-%S")
        file_name = date_now + '_PyIQ_out' + '.csv'
        file_path = os.path.join(output_dir, file_name)
        output_file = open(file_path, 'wt')
        output_writer = csv.writer(
            output_file, quoting=csv.QUOTE_NONNUMERIC, delimiter=",")
        output_writer.writerow(
            ("Filename", "tEntropy", "tBrenner", "fMoments", "fMean", "fSTD", "fEntropy",
             "fTh", "fMaxPw", "Skew", "Kurtosis", "MeanBin"))

        for image_name in os.listdir(path):
            if options.file_filter is None or options.file_filter in image_name:
                real_path = os.path.join(path, image_name)
                # Only process images
                if not os.path.isfile(real_path) or not real_path.endswith((".jpg", ".tif", ".tiff", ".png")):
                    continue
                # ImageJ files have particular TIFF tags that can be processed correctly
                # with the options.imagej switch
                image = read.get_image(path, channel=options.rgb_channel)

                # Only grayscale images are processed. If the input is an RGB image,
                # a channel can be chosen for processing.

                # Time series sometimes contain images of very different content: the start
                # of the series may show nearly empty (black) images, whereas at the end
                # of the series the whole field-of-view may be full of cells. Ranking such
                # dataset in a single piece may be challenging. Therefore the beginning of
                # the dataset can be separated from the end, by selecting a minimum value
                # for average grayscale pixel value here.
                if options.average_filter > 0 and image.average() < options.average_filter:
                    continue

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
                results.insert(0, os.path.join(path, image_name))
                output_writer.writerow(results)

                print("Done analyzing %s" % image_name)

        output_file.close()
        print("The results were saved to %s" % file_path)

    if "analyze" in options.mode:
    # In analyze mode the previously created quality ranking variables are
    # normalized to the highest value of every given variable. In addition
    # some new parameters are calculated. The results are saved into a new
    # csv file.
        if file_path is None:
            assert options.file is not None, "You have to specify a data file" \
                                             "with the --file option"
            path = os.path.join(options.working_directory, options.file)
            print(path)
            file_path = path
            assert os.path.isfile(path), "Not a valid file %s" % path
            assert path.endswith(".csv"), "Unknown suffix %s" % path.split(".")[-1]

        csv_data = pandas.read_csv(file_path)
        csv_data["cv"] = csv_data.fSTD/csv_data.fMean
        csv_data["SpatEntNorm"] = csv_data.tEntropy/csv_data.tEntropy.max()
        csv_data["SpectMean"] = csv_data.fMean/csv_data.fMean.max()
        csv_data["SpectSTDNorm"] = csv_data.fSTD/csv_data.fSTD.max()
        csv_data["InvSpectSTDNorm"] = 1- csv_data.SpectSTDNorm
        csv_data["SpectEntNorm"] = csv_data.fEntropy/csv_data.fEntropy.max()
        csv_data["SkewNorm"] = 1 - abs(csv_data.Skew)/abs(csv_data.Skew).max()
        csv_data["KurtosisNorm"] = abs(csv_data.Kurtosis)/abs(csv_data.Kurtosis).max()
        csv_data["SpectHighPowerNorm"] = csv_data.fMaxPw/csv_data.fMaxPw.max()
        csv_data["MeanBinNorm"] = csv_data.MeanBin/csv_data.MeanBin.max()
        csv_data["BrennerNorm"] = csv_data.tBrenner/csv_data.tBrenner.max()
        csv_data["SpectMomentsNorm"] = csv_data.fMoments/csv_data.fMoments.max()

        # Create output directory
        output_dir = datetime.datetime.now().strftime("%Y-%m-%d")+'_PyIQ_output'
        output_dir = os.path.join(options.working_directory, output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        date_now = datetime.datetime.now().strftime("%H-%M-%S")
        file_name = date_now + '_PyIQ_analyze_out' + '.csv'
        file_path = os.path.join(output_dir, file_name)

        csv_data.to_csv(file_path)
        print("The results were saved to %s" % file_path)

    if "plot" in options.mode:
    # With the plot option the dataset is sorted according to the desired ranking variable.
    # The changes are saved to the original csv file. In addition a plot is created to show
    # a subset of highest and lowest ranked images (the amount of images to show is
    # controlled by the options.npics parameter
        if csv_data is None:
            file_path = os.path.join(options.working_directory, options.file)
            assert os.path.isfile(file_path), "Not a valid file %s" % path
            assert path.endswith(".csv"), "Unknown suffix %s" % path.split(".")[-1]
            csv_data = pandas.read_csv(file_path)
        if options.result == "average":
            csv_data["Average"] = csv_data[["InvSpectSTDNorm", "SpatEntNorm"]].mean(axis=1)
            csv_data.sort(columns="Average", ascending=False, inplace=True)
        elif options.result == "fskew":
            csv_data.sort(columns="SkewNorm", ascending=False, inplace=True)
        elif options.result == "fentropy":
            csv_data.sort(columns="SpectEntNorm", ascending=False, inplace=True)
        elif options.result == "ientropy":
            csv_data.sort(columns="SpatEntNorm", ascending=False, inplace=True)
        elif options.result == "icv":
            csv_data.sort(columns="SpatEntNorm", ascending=False, inplace=True)
        elif options.result == "fstd":
            csv_data.sort(columns="SpectSTDNorm", ascending=False, inplace=True)
        elif options.result == "fkurtosis":
            csv_data.sort(columns="KurtosisNorm", ascending=False, inplace=True)
        elif options.result == "fpw":
            csv_data.sort(columns="SpectHighPowerNorm", ascending=False, inplace=True)
        elif options.result == "fmean":
            csv_data.sort(columns="SpectHighPowerNorm", ascending=False, inplace=True)
        elif options.result == "meanbin":
            csv_data.sort(columns="MeanBinNorm", ascending=False, inplace=True)
        else:
            print("Unknown results sorting method %s" % options.result)
            sys.exit()

        best_pics = csv_data["Filename"].head(options.npics).as_matrix()
        worst_pics = csv_data["Filename"].tail(options.npics).as_matrix()
        show_pics_from_disk(best_pics, title="BEST PICS")
        show_pics_from_disk(worst_pics, title="WORST PICS")

        csv_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()