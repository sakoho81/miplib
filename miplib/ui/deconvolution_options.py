import argparse

from miplib.psf import psfgen
from miplib.ui.frc_options import get_frc_options_group


def get_deconvolve_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "image Deconvolution script"
    )
    parser.add_argument('image')
    parser.add_argument('psf')
    parser = get_common_options(parser)
    parser = get_deconvolution_options(parser)
    parser = get_psf_estimation_options(parser)
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


def get_common_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Common",
                                      "Common Options for miplib scripts")
    group.add_argument(
        '--verbose',
        action='store_true'
    )
    group.add_argument('--debug',
                        action='store_true')
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

    parser.add_argument(
        '--carma-gate-idx',
        type=int,
        default=0,
        help='Carma files contain several images from various detector/laser gate'
             'combinations. Some scripts only work with single images, so one can'
             'specify a certain image in the file structure with the --carma-gate-idx'
             'and --carma-det-idx keywords.'
    )

    parser.add_argument(
        '--carma-det-idx',
        type=int,
        default=0,
        help='Carma files contain several images from various detector/laser gate'
             'combinations. Some scripts only work with single images, so one can'
             'specify a certain image in the file structure with the --carma-gate-idx'
             'and --carma-det-idx keywords.'
    )
    return parser


def get_deconvolution_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Deconvolution", "Options for controlling the deconvolution algorithm")

    group.add_argument(
        '--max-nof-iterations',
        type=int,
        default=100,
        help='Specify maximum number of iterations.'
    )
    group.add_argument(
        '--update-blind-psf',
        action='store_true',
        help='Updates the blind psf during deconvolution iteration'
    )
    group.add_argument(
        '--convergence-epsilon',
        dest='convergence_epsilon',
        type=float,
        default=0.05,
        help='Specify small positive number that determines '
             'the window for convergence criteria.'
    )

    group.add_argument(
        '--wiener-snr',
        type=float,
        default=100.0
    )

    group.add_argument(
        '--first-estimate',
        choices=['image',
                 'image_mean',
                 'constant'],
        default='image',
        help='Specify first estimate for iteration.'
    )

    group.add_argument(
        '--estimate-constant',
        dest='estimate_constant',
        type=float,
        default=1.0
    )

    group.add_argument(
        '--save-intermediate-results',
        action='store_true',
        help='Save intermediate results.'
    )

    group.add_argument(
        '--output-cast',
        dest='output_cast',
        action='store_true',
        help='By default the fusion output is returned as a 32-bit image'
             'This switch can be used to enable 8-bit unsigned output'
    )

    group.add_argument(
        '--blocks',
        dest='num_blocks',
        type=int,
        default=1,
        help="Define the number of blocks you want to break the images into"
             "for the image fusion. This argument defaults to 1, which means"
             "that the entire image will be used -- you should define a larger"
             "number to optimize memory consumption"
    )
    group.add_argument(
        '--stop-tau',
        type=float,
        default=0.002,
        help='Specify parameter for tau-stopping criteria.'
    )
    group.add_argument(
        '--tv-lambda',
        type=float,
        default=0,
        help="Enable Total Variation regularization by selecting value > 0"
    )

    group.add_argument(
        '--pad',
        dest='block_pad',
        type=int,
        default=0,
        help='The amount of padding to apply to a fusion block.'
    )

    group.add_argument(
        '--memmap-estimates',
        action='store_true'
    )

    group.add_argument(
        '--disable-tau1',
        action='store_true'
    )

    group.add_argument(
        '--disable-fft-psf-memmap',
        action='store_true'
    )

    group.add_argument(
        '--rl-background',
        type=float,
        default=0.0,
        help="Background correction term for RL"
    )

    return parser


def parse_psf_type(args):
    if args == "confocal" or args == "sted":
        return psfgen.GAUSSIAN | psfgen.CONFOCAL
    elif args == "widefield":
        return psfgen.GAUSSIAN | psfgen.WIDEFIELD


def parse_int_tuple(string):
    return (int(i) for i in string.split(','))


def parse_float_tuple(string):
    return (float(i) for i in string.split(','))


def get_psf_estimation_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("PSF estimation", "Options for controlling the PSF estimation algorithm")

    group.add_argument(
        '--psf-type',
        type=parse_psf_type,
        default=psfgen.GAUSSIAN | psfgen.CONFOCAL
    )

    group.add_argument(
        '--psf-shape',
        type=parse_int_tuple,
        default=(256,256)

    )
    group.add_argument(
        '--psf-size',
        type=parse_float_tuple,
        default=(4., 4.)
    )
    group.add_argument(
        '--ex-wl',
        type=float,
        default=488
    )
    group.add_argument(
        '--em-wl',
        type=float,
        default=550
    )
    group.add_argument(
        '--na',
        type=float,
        default=1.4
    )
    group.add_argument(
        '--refractive-index',
        type=float,
        default=1.414
    )
    group.add_argument(
        '--magnification',
        type=float,
        default=1.0
    )
    group.add_argument(
        '--pinhole-radius',
        type=float,
        default=None

    )

    return parser


