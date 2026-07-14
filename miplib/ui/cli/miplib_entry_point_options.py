"""
File: miplib_entry_point_options.py

In this file the command line argument interface is configured
for the various *miplib* entry points, that can be found in the
/bin directory.
"""

import argparse
from pathlib import Path

import miplib.ui.cli.argparse_helpers as helpers
from miplib.ui.cli.deconvolution_options import get_deconvolution_options_group
from miplib.ui.cli.frc_options import get_frc_options_group
from miplib.ui.cli.fusion_options import get_fusion_options_group
from miplib.ui.cli.ism_options import get_ism_reconstruction_options_group
from miplib.ui.cli.psf_estimation_options import get_psf_estimation_options_group
from miplib.ui.cli.registration_options import get_registration_options_group

# region Fourier Ring Correlation scripts


def get_frc_script_options(arguments):
    """Command line options for the Fourier ring correlation script

    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Fourier ring correlation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("directory", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--frc-mode", choices=["two-image", "one-image"], default="one-image"
    )
    parser.add_argument(
        "--outdir",
        dest="pathout",
        type=Path,
        help="Select output folder where to save the log file" + " and the plots",
    )
    parser = get_common_options_group(parser)
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


# endregion


# region Deconvolution scripts
def get_deconvolve_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for theimage Deconvolution script"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("psf", type=Path)
    parser = get_common_options_group(parser)
    parser = get_deconvolution_options_group(parser)
    parser = get_psf_estimation_options_group(parser)
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


# endregion

# region Image Scanning Microscopy reconstruction scripts


def get_ism_script_options(arguments):
    """Command line options for the ISM reconstruction script

    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for theISM image reconstruction script"
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "ism_mode",
        choices=["adaptive", "static", "wiener", "rl", "all"],
        default="reassign",
        help="Indicate the reassignment approach",
    )
    parser = get_common_options_group(parser)
    parser = get_registration_options_group(parser)
    parser = get_deconvolution_options_group(parser)
    parser = get_psf_estimation_options_group(parser)
    parser = get_frc_options_group(parser)
    parser = get_ism_reconstruction_options_group(parser)
    return parser.parse_args(arguments)


# endregion

# region Multi-View Reconstruction scripts


def get_import_script_options(arguments):
    """Import script is used in *miplib* to import data to the internal
    HDF5 file structure.


    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for themiplib data import script."
    )
    parser.add_argument("data_dir_path", type=Path)
    parser.add_argument("--scales", type=helpers.parse_int_tuple, action="store")
    parser.add_argument("--calculate-psfs", dest="calculate_psfs", action="store_true")

    parser.add_argument(
        "--copy-registration-result",
        dest="copy_registration_result",
        type=helpers.parseFromToString,
        default=-1,
    )
    parser.add_argument("--normalize-inputs", action="store_true")

    return parser.parse_args(arguments)


def get_register_script_options(arguments):
    """Command line options for the multi-view image registration script


    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for the miplib image registration script"
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Give a path to a HDF5 file that contains the images",
    )

    parser = get_common_options_group(parser)
    parser = get_registration_options_group(parser)

    return parser.parse_args(arguments)


def get_fusion_script_options(arguments):
    """Command line options for the multi-view image fusion script


    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """

    parser = argparse.ArgumentParser(
        description="Multi-view image fusion using Richardson-Lucy deconvolution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fuse all views with 100 iterations\n"
            "  miplib-fuse data.hdf5 --max-nof-iterations 100\n\n"
            "  # Fuse specific views with CUDA\n"
            "  miplib-fuse data.hdf5 --fuse-views 0,1,2 --enable-cuda "
            "--max-nof-iterations 200\n\n"
            "  # Fuse with multiplicative fusion mode\n"
            "  miplib-fuse data.hdf5 --fusion-method multiplicative "
            "--max-nof-iterations 100\n"
        ),
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Give a path to a HDF5 file that contains the images",
    )
    parser = get_common_options_group(parser)
    parser = get_fusion_options_group(parser)

    return parser.parse_args(arguments)


# endregion

# region Correlative Microscopy scripts


def get_tem_correlation_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    group = parser.add_argument_group(
        "TEM Correlation", "Options for STED-TEM correlation"
    )

    # Image file path prefix

    group.add_argument(
        "--emfile",
        "--em",
        dest="em_image_path",
        metavar="PATH",
        type=Path,
        default=None,
        help="Specify PATH to Electro microscope Image",
    )
    # STED image path
    group.add_argument(
        "--stedfile",
        "--st",
        dest="sted_image_path",
        metavar="PATH",
        type=Path,
        default=None,
        help="Specify PATH to STED Image",
    )
    group.add_argument("--register", action="store_true")
    group.add_argument("--transform", action="store_true")
    group.add_argument(
        "--transform-path",
        "-t",
        dest="transform_path",
        metavar="PATH",
        type=Path,
        help="Specify PATH to transform file",
    )
    group.add_argument(
        "--tfm-type",
        dest="tfm_type",
        choices=["rigid", "similarity"],
        default="rigid",
        help="Define the spatial transform type to be used with registration",
    )

    return parser


def get_correlate_tem_script_options(arguments):
    """This script is used to correlate fluoresence microscope (STED) and
    TEM images


    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
        "miplib correlative STED-TEM image registration script"
    )
    parser = get_common_options_group(parser)
    parser = get_tem_correlation_options(parser)
    parser = get_registration_options_group(parser)

    return parser.parse_args(arguments)


def get_transform_script_options(arguments):
    """A utility script that can be used to apply a saved spatial transform
    to  an image.


    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings,
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.

    Returns:
        [Namespace object] -- Simple class used by default by parse_args()
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for the miplib image transform script"
    )
    parser = get_common_options_group(parser)

    parser.add_argument("moving_image")
    parser.add_argument("fixed_image")
    parser.add_argument("transform")
    parser.add_argument("--hdf", action="store_true")

    return parser.parse_args(arguments)


# endregion

# region Image Quality Ranking


def get_quality_options(arguments):
    """Command line options for the image quality ranking script.

    Minimal API: positional input (file or directory), optional output path,
    and RGB channel selection. All other filter options use sensible defaults.

    Args:
        arguments: Command line parameters as a list of strings

    Returns:
        Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Image quality ranking and analysis tool for microscopy datasets. "
        "Computes multiple quality metrics (spatial entropy, Brenner gradient, spectral moments, "
        "power spectrum statistics) to rank images by focus quality and detail content. "
        "Useful for finding the best-focused images in large datasets or filtering out "
        "out-of-focus images before quantitative analysis."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input image file or directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV file path (default: auto-generated in input directory)",
        default=None,
    )
    parser.add_argument(
        "--rgb-channel",
        dest="rgb_channel",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="RGB channel to use for analysis (0=R, 1=G, 2=B, default: 1)",
    )

    return parser.parse_args(arguments)


def get_power_options(arguments):
    """Command line options for the power spectrum extraction script.

    Minimal API: positional input directory, optional output path, image size,
    and RGB channel selection.

    Args:
        arguments: Command line parameters as a list of strings

    Returns:
        Namespace object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Extract 1D radial power spectra from microscopy images. "
        "Computes the rotationally averaged power spectrum for each image in a directory "
        "and exports the results to a CSV file. Useful for analyzing frequency content "
        "and comparing resolution characteristics across image datasets."
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input directory containing images",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV file path (default: auto-generated in input directory)",
        default=None,
    )
    parser.add_argument(
        "--image-size",
        dest="image_size",
        type=int,
        default=512,
        help="Resize images to this size before analysis (default: 512)",
    )
    parser.add_argument(
        "--rgb-channel",
        dest="rgb_channel",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="RGB channel to use for analysis (0=R, 1=G, 2=B, default: 1)",
    )

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
        type=Path,
        help="Defines the location of the working directory",
        default=Path("/home/sami/Pictures/Quality"),
    )

    return parser.parse_args(arguments)


# endregion


# region Common options
def get_common_options_group(parser):
    """Common options for all the above scripts

    Arguments:
        parser {argparse.ArgumentParser} -- An argument parser to which
        the common options group is to be added.

    Returns:
        [argparse.ArgumentParser] -- The parser instance augmented with
        the new options group.
    """
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Common", "Common Options for miplib scripts")
    group.add_argument("--verbose", action="store_true")
    group.add_argument(
        "--dir",
        dest="working_directory",
        type=Path,
        default=Path("/home/sami/Data"),
        help="Path to image files",
    )
    group.add_argument(
        "--show-plots",
        dest="show_plots",
        action="store_true",
        help="Show summary plots of registration/fusion variables",
    )
    group.add_argument(
        "--show-image",
        dest="show_image",
        action="store_true",
        help="Show a 3D image of the fusion/registration result upon completion",
    )
    group.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Define the size of images to use. By default the full size "
        "originals"
        "will be used, but it is possible to select resampled images as "
        "well",
    )

    group.add_argument(
        "--channel", type=int, default=0, help="Select the active color channel."
    )

    group.add_argument(
        "--jupyter",
        action="store_true",
        help="A switch to enable certain functions that only work when using"
        "Jupyter notebook to run the code.",
    )
    group.add_argument(
        "--test-drive",
        dest="test_drive",
        action="store_true",
        help="Enable certain sections of code that are used for debugging or "
        "tuning parameter values with new images",
    )

    group.add_argument(
        "--evaluate",
        dest="evaluate_results",
        action="store_true",
        help="Indicate whether you want to evaluate the registration/fusion "
        "results by eye before they are saved"
        "to the data structure.",
    )

    group.add_argument(
        "--temp-dir",
        type=Path,
        help="Specify a custom directory for Temp data. By default it will"
        "be saved into an automatically generated directory in the "
        "system's temp file directory (/temp on *nix)",
        default=None,
    )

    group.add_argument(
        "--carma-gate-idx",
        type=int,
        default=0,
        help="Carma files contain several images from various detector/laser gate"
        "combinations. Some scripts only work with single images, so one can"
        "specify a certain image in the file structure with the --carma-gate-idx"
        "and --carma-det-idx keywords.",
    )

    group.add_argument(
        "--carma-det-idx",
        type=int,
        default=0,
        help="Carma files contain several images from various detector/laser gate"
        "combinations. Some scripts only work with single images, so one can"
        "specify a certain image in the file structure with the --carma-gate-idx"
        "and --carma-det-idx keywords.",
    )

    group.add_argument(
        "--plot-size",
        type=helpers.parse_float_tuple,
        default=(2.5, 2.5),
        help="Size of the generated plots (in)",
    )
    group.add_argument(
        "--save-plots",
        action="store_true",
        help="Save some extra plots that a script may generate",
    )

    group.add_argument(
        "--enhance-contrast-on-save",
        action="store_true",
        help="Enhance contrast of the output images, by allowing a small percentage "
        "of the pixels to saturate.",
    )

    return parser


# endregion
