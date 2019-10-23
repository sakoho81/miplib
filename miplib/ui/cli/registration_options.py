import argparse

def get_registration_options_group(parser):

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
        default='correlation',
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
        default=0.1,
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
        default=0.001,
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
        default=300,
        help='Specify an estimate for initial rotation angle'
    )
    group.add_argument(
        '--reg-relax-factor',
        dest='relaxation_factor',
        type=float,
        default=0.7,
        help='Defines how quickly optmizer shortens the step size'
    )
    group.add_argument(
        '--reg-print-prog',
        dest='print_registration_progress',
        action='store_true'
    )
    group.add_argument(
        '--reg-enable-observers',
        action='store_true'
    )

    group.add_argument(
        '--reg-translate-only',
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
        default=.7
    )

    return parser
