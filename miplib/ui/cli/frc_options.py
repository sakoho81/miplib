import argparse

def get_frc_options_group(parser):
    assert isinstance(parser, argparse.ArgumentParser)

    group = parser.add_argument_group("Fourier ring correlation analysis", "Options for FRC analysis")

    group.add_argument('--bin-delta', dest='d_bin', type=int, default=1,
                       help='Set thickness of the ring for FRC calculation')

    group.add_argument('--square', dest='resol_square', action='store_true',
                       help='Enable analysis only in the resolution square')

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
                       default='fixed')

    group.add_argument('--resolution-threshold-value',
                       type=float,
                       default=1.0/7,
                       help="The resolution threshold value to be used when fixed threshold" \
                            "is applied")

    group.add_argument('--resolution-point-sigma',
                       type=float,
                       default=0.01,
                       help="The maximum difference between the value of the FRC and threshold"
                            "curves at the intersection. ")

    group.add_argument('--resolution-snr-value',
                       type=float,
                       default=0.25,
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

    group.add_argument('--disable-hamming',
                       action='store_true')

    group.add_argument('--frc-curve-fit-type',
                       choices=['smooth-spline', 'spline', 'polynomial'],
                       default='spline')

    return parser
