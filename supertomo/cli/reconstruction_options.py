import argparse

def get_rltv_options_group(parser):
    group = parser.add_argument_group(
        "RLTV",
        "Specify options for Richardson-Lucy deconvolution algorithm with "
        "total variation term"
    )
    group.add_option ('--rltv-estimate-lambda', dest='rltv_estimate_lambda',
                      action='store_true',
                      help = 'Enable estimating RLTV parameter lambda.')
    group.add_option ('--no-rltv-estimate-lambda',
                      dest='rltv_estimate_lambda', action='store_false',
                      help = 'See ``--rltv-estimate-lambda`` option.')
    group.add_option ('--rltv-lambda-lsq-coeff',
                      type = 'float',
                      default = 0.0,
                      help = 'Specify coefficient for RLTV regularization parameter. If set '
                             'to 0 then the coefficent will be chosed such that lambda_lsq_0==50/SNR.'
                             '')
    group.add_option ('--rltv-lambda',
                      type = 'float',
                      default = 0.0,
                      help = 'Specify RLTV regularization parameter.')
    group.add_option ('--rltv-compute-lambda-lsq',
                      action='store_true',
                      help = 'Compute RLTV parameter estimation lambda_lsq.')
    group.add_option ('--rltv-algorithm-type',
                      choices = ['multiplicative', 'additive'], default='multiplicative',
                      help = 'Specify algorithm type. Use multiplicative with Poisson noise '
                             'and additive with Gaussian noise.')
    group.add_option ('--rltv-alpha',
                      type = 'float',
                      help = 'Specify additive RLTV regularization parameter.')
    group.add_option ('--rltv-stop-tau',
                      type = 'float',
                      default = 0.002,
                      help = 'Specify parameter for tau-stopping criteria.')
    return group


def set_fusion_options(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Fusion", "Options for image fusion")
    group.add_argument(
        '--verbose',
        action='store_true'
    )
    group.add_argument(
        '--hide-warnings',
        dest='hide_warnings',
        action='store_true',
        help='Hide ITK wrapper warnings'
    )

    group.add_argument(
        '--internal-type',
        dest='intern IMal_type',
        choices=['IUC', 'IF', 'ID'],
        default='IF',
        help='Specify the internal pixel type to be used in image'
             'registration'
    )
    group.add_argument(
        '--prefix',
        dest='path_prefix',
        default='/home/sami/Pictures/SuperTomo',
        help='Path to image files'
    )
    group.add_argument(
        '--register',
        action='store_true',
        dest='registration',
        help='Registration on/off',
        default=False
    )
    group.add_argument(
        '--fuse',
        action='store_true',
        dest='fusion',
        help='Fusion on/off',
        default=False
    )
    group.add_argument(
        '--fixed', '-f',
        dest='fixed_image_path',
        metavar='PATH',
        help='Specify PATH to Fixed Image'
    )
    group.add_argument(
        '--moving', '-m',
        dest='moving_image_path',
        metavar='PATH',
        help='Specify PATH to Moving Image'
    )

    group.add_argument(
        '--psf', '-p',
        dest='psf_path',
        metavar='PATH',
        help='Specify PATH to PSF images.'
    )
    group.add_argument(
        '--psf-type',
        dest='psf_type',
        choices=['single', 'measured'],
        default='single',
        help='Define a PSF to be used in the fusion'
    )
    group.add_argument(
        '--transform',
        action='store_true',
        help='Transform with pre-defined parameters (in a file)'
    )
    group.add_argument(
        '--transform-path', '-t',
        dest='transform_path',
        metavar='PATH',
        help='Specify PATH to transform file'
    )
    group.add_argument(
        '--max-nof-iterations',
        type='int',
        default=100,
        help='Specify maximum number of iterations.'
    )
    group.add_argument(
        '--convergence-epsilon',
        type='float',
        default=0.05,
        help='Specify small positive number that determines '
             'the window for convergence criteria.'
    )
    group.add_argument(
        '--degrade-input',
        action='store_true',
        default=False,
        help='Degrade input: apply noise to convolved input.'
    )
    group.add_argument(
        '--degrade-input-snr',
        type='float',
        default=0.0,
        help='Specify the signal-to-noise ratio when using --degrade-input.'
             'If set to 0, snr will be estimated as sqrt(max(input image)).'
    )
    group.add_argument(
        '--first-estimate',
        choices=['input image',
                 'convolved input image',
                 'sum of all projections',
                 'stupid tomo',
                 '2x convolved input image',
                 'last result',
                 'image_mean',
                 'average'],
        default='image_mean',
        help='Specify first estimate for iteration.'
    )
    group.add_argument(
        '--save-intermediate-results',
        action='store_true',
        help='Save intermediate results.'
    )
    group.add_argument(
        '--no-save-intermediate-results',
        dest='save_intermediate_results',
        action='store_false',
        help='See ``--save-intermediate-results`` option.'
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
        '--set-3d-rendering',
        dest='rendering_method',
        choices=['volume', 'surface'],
        default='volume',
        help='Set the rendering method to be used for 3d data plots'
    )

    group.add_argument(
        '--subset',
        dest='subset_image',
        type='float',
        default=1.0,
        help="If you would like to work with a subset of data, instead of"
             "the whole image, specify a multiplier ]0, 1[ here"
    )
    group.add_argument(
        '--rescale',
        dest='rescale_to_full_range',
        action='store_true',
        help='Rescale intensities of the input images to the full dynamic'\
             'range allowed by the pixel type'
    )

    group.add_argument(
        '--output-cast',
        dest='output_cast',
        action='store_true',
        help='By default the fusion output is returned as a 32-bit image'
             'This switch can be used to enbale 8-bit unsigned output'
    )
    group.add_argument(
        '--fusion-method',
        dest='fusion_method',
        choices=['multiplicative', 'multiplicative-opt', 'summative', 'summative-opt'],
        default='summative'
    )
