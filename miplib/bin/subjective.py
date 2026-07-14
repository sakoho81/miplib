#!/usr/bin/env python
# -*- python -*-

"""
File: subjective.py
Author: Sami Koho (sami.koho@gmail.com)

Description:

A utility script for performing subjective image rankings. One image is
displayed at a time, after which it is ranked on 1-5 scale, 5 being the
best and 1 the worst. The script can be run multiple times to collect
several ranking results in a single csv file. At every run the data
is shuffled in order to not repeat the same image sequence twice.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas

import miplib.ui.cli.miplib_entry_point_options as script_options


def main():
    options = script_options.get_subjective_ranking_options(sys.argv[1:])
    path = Path(options.working_directory)
    index = 0
    assert path.is_dir(), str(path)

    # Create or open a csv file
    output_dir = path
    file_name = "subjective_ranking_scores.csv"
    file_path = output_dir / file_name

    if file_path.exists():
        csv_data = pandas.read_csv(file_path)
        # Append a new result column
        for column in csv_data:
            if "Result" in column:
                index += 1
    else:
        csv_data = pandas.DataFrame()
        file_names = []
        # Get valid file names
        for image_entry in path.iterdir():
            if not image_entry.is_file() or image_entry.suffix not in (
                ".jpg",
                ".tif",
                ".tiff",
                ".png",
            ):
                continue
            file_names.append(image_entry.name)
        csv_data["Filename"] = file_names

    result_name = "Result_" + str(index)
    results = []

    # Plot settings
    plt.ion()
    plt.axis("off")

    # Shuffle the data frame so that the order of the displayed images is mixed every time.
    csv_data = csv_data.sample(frac=1)
    print(
        "Images are graded on a scale 1-5, where 1 denotes a very bad image "
        "and 5 an excellent image"
    )

    for image_name in csv_data["Filename"]:
        real_path = path / image_name
        image = plt.imread(real_path)

        plt.imshow(image, cmap="hot", vmax=image.max(), vmin=image.min())

        success = False
        while not success:
            user_input = input("Give grade: ")

            if user_input.isdigit():
                result = int(user_input)
            else:
                print("Please give a numeric grade 1-5.")
                continue

            if 1 <= result <= 5:
                success = True
                results.append(result)
            else:
                print("Please give a numeric grade 1-5.")

    csv_data[result_name] = results
    csv_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    main()
