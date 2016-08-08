
"""
fuse.py

Copyright (c) 2016 Sami Koho  All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This is the main program file for the SuperTomo fusion calculation
"""

import sys
import os

import supertomo.reconstruction.fusion as fusion
import supertomo.io.image_data as image_data
import supertomo.ui.arguments as arguments


def main():
    options = arguments.get_fusion_options(sys.argv[1:])

    full_path = os.path.join(options.working_directory,
                             options.data_file)

    if not os.path.isfile(full_path):
        raise AttributeError("No such file: %s" % full_path)
    elif not full_path.endswith(".hdf5"):
        raise AttributeError("Not a HDF5 file")

    data = image_data.ImageData(full_path)

    task = fusion.MultiViewFusionRL(data, options)
    task.execute()

    task.show_result()

    while True:
        keep = raw_input("Do you want to save the result (yes/no)? ")
        if keep in ('y', 'Y', 'yes', 'YES'):
            task.save_to_hdf()
        elif keep in ('n', 'N', 'no', 'No'):
            print "Exiting without saving results."
            break
        else:
            print "Unkown command. Please state yes or no"

    data.close()


if __name__ == "__main__":
    main()
