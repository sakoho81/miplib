import argparse


def get_deconvolution_options_group(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group("Deconvolution", "Options for controlling the deconvolution algorithm")

    group.add_argument(
        '--max-nof-iterations',
        type=int,
        default=20,
        help='Specify maximum number of iterations.'
    )
    group.add_argument(
        '--update-blind-psf',
        type=int,
        default=0,
        help='Updates the blind psf during deconvolution, at specified iteration intervals.'
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
        '--wiener-nsr',
        type=float,
        default=100.0
    )

    group.add_argument(
        '--first-estimate',
        choices=['image',
                 'blurred',
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
        default=0.0001,
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
    group.add_argument(
        '--rl-auto-background',
        action="store_true"
    )
    group.add_argument(
        '--rl-frc-stop',
        type=float,
        default=0.0,
        help= "Set a stopping condition for the deconvolution based on FRC"
    )
    return parser





