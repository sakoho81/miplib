import argparse


def get_frc_script_options(arguments):
    parser = argparse.ArgumentParser(description='Fourier ring correlation analysis',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('directory')
    parser.add_argument('--debug',
                        action='store_true')
    parser.add_argument('--outdir', dest='pathout',
                        help='Select output folder where to save the log file'
                             + ' and the plots')
    parser = get_frc_options_group(parser)
    return parser.parse_args(arguments)


def get_frc_options_group(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    group = parser.add_argument_group("Fourier ring correlation analysis", "Options for FRC analysis")

    group.add_argument('--bin-delta', dest='d_bin', type=float, default=5,
                       help='Set thickness of the ring for FRC calculation')

    group.add_argument('--square', dest='resol_square', action='store_true',
                       help='Enable analysis only in the resolution square')

    group.add_argument('--hanning', dest='hanning', action='store_true',
                       help='Enable multiplication of the images with a hanning window')

    group.add_argument("--normalize-power", dest="normalize_power", action="store_true")

    group.add_argument('--frc-curve-fit-degree',
                       dest='frc_curve_fit_degree',
                       type=int,
                       default=8)

    group.add_argument('--resolution-threshold-curve-fit-degree',
                       dest='resolution_threshold_curve_fit_degree',
                       type=int,
                       default=3)

    group.add_argument('--resolution-threshold-criterion',
                       dest='resolution_threshold_criterion',
                       choices=['one-bit', 'half-bit', 'fixed', 'three-sigma', 'snr'],
                       default='half-bit')

    group.add_argument('--resolution-threshold-value',
                       type=float,
                       default=0.5,
                       help="The resolution threshold value to be used when fixed threshold" \
                            "is applied")

    group.add_argument('--resolution-point-sigma',
                       type=float,
                       default=0.01,
                       help="The maximum difference between the value of the FRC and threshold"
                            "curves at the intersection. ")

    group.add_argument('--resolution-snr-value',
                       type=float,
                       default=0.5,
                       help="The target SNR value for the resolution measurement.")

    group.add_argument('--angle-delta',
                       dest='d_angle',
                       type=int,
                       default=20,
                       help="The size of angle increment in directional FSC analysis."
    )

    group.add_argument('--extract-angle-delta',
                       dest='d_extract_angle',
                       type=float,
                       default=5.0,
                       help="The size of the angle when using hollow sphere iterator."
                       )

    group.add_argument('--enable-hollow-iterator',
                       dest='hollow_iterator',
                       action='store_true',
                       help="Enable hollow iterator"
                       )

    group.add_argument('--curve-fit-min',
                       dest='min_filter',
                       action='store_true',
                       help="Enable min filtering for Correlation curve fitting. This will help"
                            "with saturation artefacts, but doesn't behave nicely with very few"
                            "data points."
                       )

    group.add_argument('--use-splines',
                       action='store_true')

    return parser
