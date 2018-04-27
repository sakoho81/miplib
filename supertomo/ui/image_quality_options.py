"""
File:        image_quality_options.py
Author:      Sami Koho (sami.koho@gmail.com

Description:
This file contains the various generic command line options
for controlling the behaviour of the PyImageQuality software.
In addition, some specific parameters are defined in the
filters.py file
"""

import argparse

import supertomo.analysis.image_quality.filters as filters


def get_quality_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
                    "image quality ranking software"
    )

    parser.add_argument(
        "--file",
        help="Defines a path to the image files",
        default=None
    )
    parser.add_argument(
        "--file-filter",
        dest="file_filter",
        default=None,
        help="Define a common string in the files to be analysed"
    )
    parser.add_argument(
        "--rgb-channel",
        help="Select which channel in an RGB image is to be used for quality"
             " analysis",
        dest="rgb_channel",
        type=int,
        choices=[0, 1, 2],
        default=1
    )
    # File filtering for batch mode processing
    parser.add_argument(
        "--average-filter",
        dest="average_filter",
        type=int,
        default=0,
        help="Analyze only images with similar amount of detail, by selecting a "
             "grayscale average pixel value threshold here"
    )
    parser.add_argument(
        "--working-directory",
        dest="working_directory",
        help="Defines the location of the working directory",
        default="/home/sami/Pictures/Quality"
    )
    parser.add_argument(
        "--mode",
        choices=["file", "directory", "analyze", "plot"],
        action="append",
        help="The argument containing the functionality of the main program"
             "You can concatenate actions by defining multiple modes in a"
             "single command, e.g. --mode=directory --mode=analyze"
    )
    # Parameters for controlling the way plot functionality works.
    parser.add_argument(
        "--result",
        default="average",
        choices=["average", "fskew", "ientropy", "fentropy", "fstd",
                 "fkurtosis", "fpw", "fmean", "icv", "meanbin"],
        help="Tell how you want the results to be calculated."
    )
    parser.add_argument(
        "--npics",
        type=int,
        default=9,
        help="Define how many images are shown in the plots"
    )

    parser = filters.get_common_options(parser)
    return parser.parse_args(arguments)


def get_power_script_options(arguments):
    """
    Command line arguments for the power.py script that is used to calculate
    1D power spectra of images within a directory.
    """
    parser = argparse.ArgumentParser(
        description="Command line options for the power.py script that can be"
                    "used to save the power spectra of images within a "
                    "directory"
    )
    parser.add_argument(
        "--working-directory",
        dest="working_directory",
        help="Defines the location of the working directory",
        default="/home/sami/Pictures/Quality"
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        default=512
    )
    parser = filters.get_common_options(parser)
    return parser.parse_args(arguments)


def get_subjective_ranking_options(arguments):
    """
    Command line arguments for the subjective.py script that can be used
    to obtain subjective opinion scores for image quality.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
                    "subjective image quality ranking"
                    "script."
    )
    parser.add_argument(
        "--working-directory",
        dest="working_directory",
        help="Defines the location of the working directory",
        default="/home/sami/Pictures/Quality"
    )

    return parser.parse_args(arguments)

def get_resolution_script_options(arguments):

    parser = argparse.ArgumentParser(description='Fourier ring correlation analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('directory')

    parser.add_argument('--outdir', dest='pathout',
                        help='Select output folder where to save the log file'
                             + ' and the plots')

    parser.add_argument('--ring', dest='width_ring', type=float, default=5,
                        help='Set thickness of the ring for FRC calculation')

    parser.add_argument('--square', dest='resol_square', action='store_true',
                        help='Enable analysis only in the resolution square')

    parser.add_argument('--hanning', dest='hanning', action='store_true',
                        help='Enable multiplication of the images with a hanning window')

    parser.add_argument('--labels', dest='labels',
                        help='Enable specific labels for plots, one for each pair of images;'
                             + ' e.g.: -l EST:GRIDREC:IFBPTV')

    parser.add_argument('--plot', dest='plot', action='store_true',
                        help='Display check plot')

    parser.add_argument("--normalize-power", dest="normalize_power", action="store_true")

    parser.add_argument('--polynomial', dest='polynomial_degree', type=int,
                        default=8)
    parser.add_argument('--resolution', dest='resolution_criterion',
                        choices=['one-bit', 'half-bit', 'half-height'],
                        default='half-bit')

    return parser.parse_args(arguments)


