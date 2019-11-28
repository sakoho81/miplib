import argparse


def get_ism_reconstruction_options_group(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group(
        "ISM reconstruction",
        "Options for controlling the ISM reconstruction")

    group.add_argument(
        '--ism-spad-pitch',
        type=float,
        default=75,
        help="The pixel pitch (horizontal/vertical) in microns on the SPAD array"
    )
    group.add_argument(
        '--ism-spad-fov-au',
        type=float,
        default=1.5,
        help='Define the size of the SPAD field of view in Airy units'
    )
    group.add_argument(
        '--ism-wavelength',
        type=float,
        default=.550,
        help='Define the wavelength to be used for theoretical shifts calculation.'
    )
    group.add_argument(
        '--ism-na',
        type=float,
        default=1.4,
        help='The objective numerical aperture.'
    )
    group.add_argument(
        '--ism-alpha',
        type=float,
        default=0.5,
        help="The ISM reassignment factor."
    )

    return parser

