import argparse
import psf
import miplib.ui.cli.argparse_helpers as helpers

def parse_psf_type(args):
    if args == "confocal" or args == "sted":
        return psf.GAUSSIAN | psf.CONFOCAL
    elif args == "widefield":
        return psf.GAUSSIAN | psf.WIDEFIELD
    else:
        raise argparse.ArgumentTypeError("Unknown PSF type")


def get_psf_estimation_options_group(parser):
    assert isinstance(parser, argparse.ArgumentParser)
    group = parser.add_argument_group(
        "PSF estimation", 
        "Options for controlling the PSF estimation algorithm")

    group.add_argument(
        '--psf-type',
        type=parse_psf_type,
        default=psf.GAUSSIAN | psf.CONFOCAL
    )

    group.add_argument(
        '--psf-shape',
        type=helpers.parse_int_tuple,
        default=(256,256)

    )
    group.add_argument(
        '--psf-size',
        type=helpers.parse_float_tuple,
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