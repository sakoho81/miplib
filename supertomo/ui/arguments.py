import argparse
from itertools import chain

# CUSTOM PARSER TYPES
# =============================================================================


def parseRangeList(rngs):
    """
    This parser type was created to enable the input of numeric ranges, such
    as "2, 5, 7-11, 26". It returns a sorted list of integers.
    """
    def parse_range(rng):
        parts = rng.split('-')
        if 1 > len(parts) > 2:
            raise ValueError("Bad range: '%s'" % (rng,))
        parts = [int(i) for i in parts]
        start = parts[0]
        end = start if len(parts) == 1 else parts[1]
        if start > end:
            end, start = start, end
        return range(start, end + 1)

    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))

def parseFromToString(string):
    return list(int(i) for i in string.split("to"))

def parseCommaSeparatedList(string):
    return sorted(int(i)for i in string.split(','))


# CLI SCRIPT ARGUMENTS
# The supertomo scripts located in the /bin each have their specific
# argparse functions. Common arguments are added as argument groups.

def get_import_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "SuperTomo2 data import script."
    )
    parser.add_argument('data_dir_path')
    parser.add_argument(
        '--scales',
        type=parseCommaSeparatedList,
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
        type=parseFromToString,
        default=-1,
    )
    parser.add_argument(
        '--normalize-inputs',
        action='store_true'
    )

    return parser.parse_args(arguments)


def get_register_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
                    "SuperTomo2 image registration script"
    )
    parser.add_argument('data_file')
    parser = get_common_options(parser)
    parser = get_registration_options(parser)

    return parser.parse_args(arguments)


def get_fusion_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    "SuperTomo2 image fusion script"
    )
    parser.add_argument('data_file')
    parser = get_common_options(parser)
    parser = get_fusion_options(parser)

    return parser.parse_args(arguments)


def get_correlate_tem_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the "
                    "SuperTomo2 correlative STED-TEM image registration script"
    )
    parser = get_common_options(parser)
    parser = get_tem_correlation_options(parser)
    parser = get_registration_options(parser)

    return parser.parse_args(arguments)


def get_transform_script_options(arguments):
    parser = argparse.ArgumentParser(
        description="Command line arguments for the"
                    " SuperTomo2 image transform script"
    )
    parser = get_common_options(parser)

    parser.add_argument('moving_image')
    parser.add_argument('fixed_image')
    parser.add_argument('transform')
    parser.add_argument('--hdf', action='store_true')

    return parser.parse_args(arguments)


# Argument groups

def get_common_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Common",
                                      "Common Options for SuperTomo2 scripts")
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
    return parser


def get_fusion_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Fusion", "Options for image fusion")

    group.add_argument(
        '--disable-cuda',
        action='store_true'
    )
    group.add_argument(
        '--max-nof-iterations',
        type=int,
        default=100,
        help='Specify maximum number of iterations.'
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
        '--first-estimate',
        choices=['first_image',
                 'sum_of_all',
                 'simple_fusion',
                 'first_image_mean',
                 'average_of_all',
                 'constant'],
        default='first_image_mean',
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
        '--fusion-method',
        dest='fusion_method',
        choices=['multiplicative', 'multiplicative-opt', 'summative',
                 'summative-opt'],
        default='summative'
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
        '--rltv-stop-tau',
        type=float,
        default=0.002,
        help='Specify parameter for tau-stopping criteria.'
    )
    group.add_argument(
        '--rltv-lambda',
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
        '--fuse-views',
        dest='fuse_views',
        type=parseRangeList,
        default=-1
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
    return parser


def get_registration_options(parser):

    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Registration",
                                      "Options for image registration")
    group.add_argument(
        '--initializer-off',
        dest='initializer',
        action='store_false'
    )
    group.add_argument(
        '--reg-method',
        dest='registration_method',
        choices=['mattes', 'mean-squared-difference', 'viola-wells',
                 'correlation'],
        default='mattes',
        help='Specify registration method'
    )
    group.add_argument(
        '--two-step',
        dest='two_step_registration',
        action='store_true',
        help='Select if you want to do a two phase registration, '
             'the first being with a degraded image and the second'
             'with the high-resolution original'
    )
    group.add_argument(
        '--normalize',
        action='store_true',
        help='Choose this option if you want to normalize the intensity values'
             'before registration. Some registration methods work better with'
             'normalized intensities.'
    )
    # It is possible to degrade the input images before registration
    # This sometimes helps with the registrations, as differences
    # at pixel level may get the registration stuck at the wrong
    # position
    group.add_argument(
        '--gaussian',
        dest='gaussian_variance',
        type=float,
        default=0.0,
        help='Define variance for Gaussian blur'
    )
    group.add_argument(
        '--dilation',
        dest='dilation_size',
        type=int,
        default=0,
        help='Define size for Grayscale dilation'
    )
    group.add_argument(
        '--mean',
        dest='mean_kernel',
        type=int,
        default=0,
        help='In case you would like to use a mean filter to smoothen the '
             'images'
             'before registration, define a kernel here'
    )
    group.add_argument(
        '--median',
        dest='median_size',
        type=int,
        default=0,
        help='Enable median filtering before registering by a non-zero kernel '
             'size'
    )

    # Mattes mutual information metric specific options
    group.add_argument(
        '--mattes-histogram-bins',
        dest='mattes_histogram_bins',
        type=int,
        default=15,
        help='Specify the number of histogram bins for Mattes '
             'Mutual Information sampling'
    )
    group.add_argument(
        '--sampling-percentage',
        dest='sampling_percentage',
        type=float,
        default=1.0,
        help='Specify the number of samples to take from each '
             'histogram bin'
    )

    # Viola Wells mutual information specific parameters
    group.add_argument(
        '--vw-fixed-sd',
        dest='vw_fixed_sd',
        type=float,
        default=0.4,
        help='Specify the fixed image SD value in Viola-Wells mutual '
             'information registration'
    )
    group.add_argument(
        '--vw-moving-sd',
        dest='vw_moving_sd',
        type=float,
        default=0.4,
        help='Specify the fixed image SD value in Viola-Wells mutual '
             'information registration'
    )
    group.add_argument(
        '--vw-samples-multiplier',
        dest='vw_samples_multiplier',
        type=float,
        default=0.2,
        help='Specify the amount of spatial samples to be used in '
             'mutual information calculations. The amount is given'
             'as a proportion of the total number of pixels in the'
             'fixed image.'
    )

    # Initializer options
    group.add_argument(
        '--set-rot-axis',
        dest='rot_axis',
        type=int,
        default=0,
        help='Specify the axis for initial rotation of the '
             'moving image'
    )
    group.add_argument(
        '--set-rotation',
        dest='set_rotation',
        type=float,
        default=1.0,
        help='Specify an estimate for initial rotation angle'
    )
    group.add_argument(
        '--set-scale',
        dest='set_scale',
        type=float,
        default=1.0,
        help='Specify the initial scale for similarity transform'
    )
    # Optimizer options
    group.add_argument(
        '--set-translation-scale',
        dest='translation_scale',
        type=float,
        default=1.0,
        help='A scaling parameter to adjust optimizer behavior'
             'effect on rotation and translation. By default'
             'the translation scale is 1000 times that of rotation'
    )
    group.add_argument(
        '--set-scaling-scale',
        dest='scaling_scale',
        type=float,
        default=10.0
    )
    group.add_argument(
        '--max-step',
        dest='max_step_length',
        type=float,
        default=0.2,
        help='Specify an estimate for initial rotation angle'
    )
    group.add_argument(
        '--min-step',
        dest='min_step_length',
        type=float,
        default=0.000001,
        help='Specify an estimate for initial rotation angle'
    )
    group.add_argument(
        '--x-offset',
        dest='x_offset',
        type=float,
        default=0.0
    )
    group.add_argument(
        '--y-offset',
        dest='y_offset',
        type=float,
        default=0.0
    )
    group.add_argument(
        '--z-offset',
        dest='z_offset',
        type=float,
        default=0.0
    )

    group.add_argument(
        '--reg-max-iterations',
        dest='registration_max_iterations',
        type=int,
        default=200,
        help='Specify an estimate for initial rotation angle'
    )
    group.add_argument(
        '--reg-relax-factor',
        dest='relaxation_factor',
        type=float,
        default=0.5,
        help='Defines how quickly optmizer shortens the step size'
    )
    group.add_argument(
        '--reg-print-prog',
        dest='print_registration_progress',
        action='store_true'
    )
    group.add_argument(
        '--use-internal-type',
        dest='use_internal_type',
        action='store_true'
    )
    group.add_argument(
        '--disable-init-moments',
        dest='moments',
        action='store_false'
    )
    group.add_argument(
        '--mask-threshold',
        dest='mask_threshold',
        type=int,
        default=30,
        help='Intensity threshold for the registration spatial mask. It '
             'defaults to 30, which works with most images.'
    )
    group.add_argument(
        '--learning-rate',
        dest='learning_rate',
        type=float,
        default=1.0
    )



    return parser


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
