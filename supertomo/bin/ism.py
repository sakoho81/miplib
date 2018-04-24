
import os
import sys

import SimpleITK as sitk

from supertomo.data.io import carma
from supertomo.processing.registration import registration
from supertomo.ui import supertomo_options, utils
from supertomo.processing import itk as itkutils


def main():
    options = supertomo_options.get_register_script_options(sys.argv[1:])


if __name__ == "__main__":
    main()
