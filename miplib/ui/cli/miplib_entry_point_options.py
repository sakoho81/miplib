"""
File: miplib_entry_point_options.py

In this file the command line argument interface is configured
for the various *miplib* entry points, that can be found in the
/bin directory.
"""

import argparse

import miplib.analysis.image_quality.filters as filters
import miplib.ui.cli.argparse_helpers as helpers
from miplib.ui.cli.deconvolution_options import get_deconvolution_options_group
from miplib.ui.cli.frc_options import get_frc_options_group
from miplib.ui.cli.fusion_options import get_fusion_options_group
from miplib.ui.cli.psf_estimation_options import get_psf_estimation_options_group
from miplib.ui.cli.registration_options import get_registration_options_group
from miplib.ui.cli.ism_options import get_ism_reconstruction_options_group



# region Fourier Ring Correlation scripts

def get_frc_script_options(arguments):
    """ Command line options for the Fourier ring correlation script
    
    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings, 
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.
    
    Returns:
        [Namespace object] -- Simple class used by default by parse_args() 
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(description='Fourier ring correlation analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('directory')
    parser.add_argument('--debug',
                        action='store_true')
    parser.add_argument('--frc-mode', choices=["two-image", "one-image"], default="one-image")
    parser.add_argument('--outdir', dest='pathout',
                        help='Select output folder where to save the log file'
                             + ' and the plots')
    parser = get_common_options_group(parser)
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


# endregion

# region Deconvolution scripts
def get_deconvolve_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "image Deconvolution script"
    )
    parser.add_argument('image')
    parser.add_argument('psf')
    parser = get_common_options_group(parser)
    parser = get_deconvolution_options_group(parser)
    parser = get_psf_estimation_options_group(parser)
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


# endregion

# region Image Scanning Microscopy reconstruction scripts

def get_ism_script_options(arguments):

    """ Command line options for the ISM reconstruction script
    
    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings, 
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.
    
    Returns:
        [Namespace object] -- Simple class used by default by parse_args() 
        to create an object holding attributes and return it.
    """
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "ISM image reconstruction script"
    )
    parser.add_argument('directory', type=helpers.parse_is_dir)
    parser.add_argument('ism_mode',
                        choices=["adaptive", "static", "wiener", "rl", "all"],
                        default="reassign",
                        help="Indicate the reassignment approach"
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
    """ Import script is used in *miplib* to import data to the internal
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
        description="Command line arguments for the"
                    "miplib data import script."
    )
    parser.add_argument('data_dir_path')
    parser.add_argument(
        '--scales',
        type=helpers.parse_int_tuple,
        action='store'
    )
    parser.add_argument(
        '--calculate-psfs',
        dest='calculate_psfs',
        action='store_true'
    )

    parser.add_argument(
        '--copy-registration-result',
        dest='copy_registration_result',
        type=helpers.parseFromToString,
        default=-1,
    )
    parser.add_argument(
        '--normalize-inputs',
        action='store_true'
    )

    return parser.parse_args(arguments)


def get_register_script_options(arguments):
    """ Command line options for the multi-view image registration script

    
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
                    "miplib image registration script"
    )
    parser.add_argument(
        'data_file',
        help="Give a path to a HDF5 file that contains the images")

    parser = get_common_options_group(parser)
    parser = get_registration_options_group(parser)

    return parser.parse_args(arguments)


def get_fusion_script_options(arguments):
    """ Command line options for the multi-view image fusion script

    
    Arguments:
        arguments {tuple} -- Command line parameters as a tuple of strings, 
        typically obtained as sys.argv[1:]. But one can of course just use
        string.split(" "), if using in a notebook for example.
    
    Returns:
        [Namespace object] -- Simple class used by default by parse_args() 
        to create an object holding attributes and return it.
    """

    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "miplib image fusion script"
    )
    parser.add_argument(
        'data_file',
        help="Give a path to a HDF5 file that contains the images")
    parser = get_common_options_group(parser)
    parser = get_fusion_options_group(parser)

    return parser.parse_args(arguments)


# endregion

# region Correlative Microscopy scripts

def get_tem_correlation_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    group = parser.add_argument_group("TEM Correlation",
                                      "Options for STED-TEM correlation")

    # Image file path prefix

    group.add_argument(
        '--emfile', '--em',
        dest='em_image_path',
        metavar='PATH',
        default=None,
        help='Specify PATH to Electro microscope Image'
    )
    # STED image path
    group.add_argument(
        '--stedfile', '--st',
        dest='sted_image_path',
        metavar='PATH',
        default=None,
        help='Specify PATH to STED Image'
    )
    group.add_argument(
        '--register',
        action='store_true'
    )
    group.add_argument(
        '--transform',
        action='store_true'
    )
    group.add_argument(
        '--transform-path', '-t',
        dest='transform_path',
        metavar='PATH',
        help='Specify PATH to transform file'
    )
    group.add_argument(
        '--tfm-type',
        dest='tfm_type',
        choices=['rigid', 'similarity'],
        default='rigid',
        help='Define the spatial transform type to be used with registration'
    )

    return parser


def get_correlate_tem_script_options(arguments):
    """ This script is used to correlate fluoresence microscope (STED) and
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
    """ A utility script that can be used to apply a saved spatial transform
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
        description="Command line arguments for the"
                    " miplib image transform script"
    )
    parser = get_common_options_group(parser)

    parser.add_argument('moving_image')
    parser.add_argument('fixed_image')
    parser.add_argument('transform')
    parser.add_argument('--hdf', action='store_true')

    return parser.parse_args(arguments)


# endregion

# region Image Quality Ranking

def get_quality_script_options(arguments):
    """ Command line options for the image quality ranking script

    
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
                    "image quality ranking software"
    )

    parser.add_argument(
        "--file",
        help="Defines a path to the image files",
        default=None
    )
    parser.add_argument('--debug',
                        action='store_true')
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
    parser = get_common_options_group(parser)
    parser = get_frc_options_group(parser)
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


# endregion

# region Common options
def get_common_options_group(parser):
    """ Common options for all the above scripts
    
    Arguments:
        parser {argparse.ArgumentParser} -- An argument parser to which
        the common options group is to be added.
    
    Returns:
        [argparse.ArgumentParser] -- The parser instance augmented with 
        the new options group.
    """
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Common",
                                      "Common Options for miplib scripts")
    group.add_argument(
        '--verbose',
        action='store_true'
    )
    group.add_argument(
        '--dir',
        dest='working_directory',
        default='/home/sami/Data',
        help='Path to image files'
    )
    group.add_argument(
        '--show-plots',
        dest='show_plots',
        action='store_true',
        help='Show summary plots of registration/fusion variables'
    )
    group.add_argument(
        '--show-image',
        dest='show_image',
        action='store_true',
        help='Show a 3D image of the fusion/registration result upon '
             'completion'
    )
    group.add_argument(
        '--scale',
        type=int,
        default=100,
        help="Define the size of images to use. By default the full size "
             "originals"
             "will be used, but it is possible to select resampled images as "
             "well"
    )

    group.add_argument(
        '--channel',
        type=int,
        default=0,
        help="Select the active color channel."
    )

    group.add_argument(
        '--jupyter',
        action='store_true',
        help='A switch to enable certain functions that only work when using'
             'Jupyter notebook to run the code.'
    )
    group.add_argument(
        '--test-drive',
        dest="test_drive",
        action='store_true',
        help="Enable certain sections of code that are used for debugging or "
             "tuning parameter values with new images"
    )

    group.add_argument(
        '--evaluate',
        dest='evaluate_results',
        action='store_true',
        help='Indicate whether you want to evaluate the registration/fusion '
             'results by eye before they are saved'
             'to the data structure.'
    )

    group.add_argument(
        '--temp-dir',
        help="Specify a custom directory for Temp data. By default it will"
             "be saved into an automatically generated directory in the "
             "system's temp file directory (/temp on *nix)",
        default=None
    )

    group.add_argument(
        '--carma-gate-idx',
        type=int,
        default=0,
        help='Carma files contain several images from various detector/laser gate'
             'combinations. Some scripts only work with single images, so one can'
             'specify a certain image in the file structure with the --carma-gate-idx'
             'and --carma-det-idx keywords.'
    )

    group.add_argument(
        '--carma-det-idx',
        type=int,
        default=0,
        help='Carma files contain several images from various detector/laser gate'
             'combinations. Some scripts only work with single images, so one can'
             'specify a certain image in the file structure with the --carma-gate-idx'
             'and --carma-det-idx keywords.'
    )

    group.add_argument(
        '--plot-size',
        type=helpers.parse_float_tuple,
        default=(2.5, 2.5),
        help='Size of the generated plots (in)'
    )
    group.add_argument(
        '--save-plots',
        action='store_true',
        help='Save some extra plots that a script may generate'
    )

    group.add_argument(
        '--enhance-contrast-on-save',
        action='store_true',
        help='Enhance contrast of the output images, by allowing a small percentage '
             'of the pixels to saturate.'
    )

    return parser
# endregion
