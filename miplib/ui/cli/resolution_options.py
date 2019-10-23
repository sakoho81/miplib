import argparse


def get_fsc_script_options(arguments):
    parser = argparse.ArgumentParser("Command line arguments for the 3D FSC script")

    parser.add_argument("halfmap1", 
                        type=str,
                        help="First half map of 3D reconstruction. MRC format. Can be masked or unmasked.",
                        metavar="HALFMAP1.MRC")

    parser.add_argument("halfmap2",
                        type=str,
                        help="Second half map of 3D reconstruction. MRC format. Can be masked or unmasked.",
                        metavar="HALFMAP2.MRC")

    parser.add_argument("fullmap",
                        type=str,
                        help="Full map of 3D reconstruction. MRC format. Can be masked or unmasked, "
                             "can be sharpened or unsharpened. ",
                        metavar="FULLMAP.MRC")

    parser.add_argument('--dir',
                        dest='working_directory',
                        default='/home/sami/Data',
                        help='Path to image files')

    parser.add_argument("--apix",
                        type=float,
                        default=1.0,
                        help="Angstrom per pixel of 3D map.",
                        metavar="FLOAT")

    parser.add_argument("--mask",
                        type=str,
                        help="If given, it would be used to mask the half maps during 3DFSC generation and analysis.",
                        metavar="MASK.MRC")

    parser.add_argument("--dthetaInDegrees",
                        type=float,
                        default=20.0,
                        help="Angle of cone to be used for 3D FSC sampling in degrees. Default is 20 degrees.",
                        metavar="FLOAT")

    parser.add_argument("--histogram",
                        type=str,
                        default="histogram",
                        help="Name of output histogram graph. No file extension required - it will automatically be "
                             "given a .pdf extension. No paths please.",
                        metavar="FILENAME")

    parser.add_argument("--FSCCutoff",
                        type=float,
                        default=0.143,
                        help="FSC cutoff criterion. 0.143 is default.",
                        metavar="FLOAT")

    parser.add_argument("--ThresholdForSphericity",
                        type=float,
                        default=0.5,
                        help="Threshold value for 3DFSC volume for calculating sphericity. 0.5 is default.",
                        metavar="FLOAT")

    parser.add_argument("--HighPassFilter",
                        type=float,
                        default=200.0,
                        help="High pass filter for thresholding in Angstrom. Prevents small dips in directional "
                             "FSCs at low spatial frequency due to noise from messing up the thresholding step. "
                             "Decrease if you see a huge wedge missing from your thresholded 3DFSC volume. "
                             "200 Angstroms is default.",
                        metavar="FLOAT")

    parser.add_argument("--Skip3DFSCGeneration",
                        action="store_true",
                        help="Allows for skipping of 3DFSC generation to directly run the analysis on a previously "
                             "generated set of results.",
                        metavar="True or False")

    parser.add_argument("--numThresholdsForSphericityCalcs",
                        type=int,
                        default=0,
                        help="calculate sphericities at different threshold cutoffs to determine sphericity deviation "
                             "across spatial frequencies. This can be useful to evaluate possible effects of "
                             "overfitting or improperly assigned orientations.",
                        metavar="INT")

    return parser.parse_args(arguments)
