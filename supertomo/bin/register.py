#!/usr/bin/env python
# -*- python -*-

"""
register.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the SuperTomo fusion calculation
"""

import os
import sys

from supertomo.reconstruction import registration
from supertomo.io import image_data
from supertomo.ui import arguments

def main():
    options = arguments.get_register_script_options(sys.argv[2:])

    filename = sys.argv[1]
    if not filename.endswith(".hdf5"):
        raise ValueError("Please specify a HDF5 data file")
    path = os.path.join(options.working_directory, filename)
    if not os.path.exists(path):
        raise ValueError("The specified file %s does not exist" % path)

    data = image_data.ImageData(path)

