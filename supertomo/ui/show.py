import subprocess
import supertomo.io.tiffile as tiffile
import os

vaa3d_bin = "/home/sami/bin/Vaa3D_Ubuntu_64bit_v3.200/vaa3d"


def evaluate_3d_image(image):
    """
    A utility function that can be used to display the registration
    and/or fusion results in Vaa3D volume viewer. The function returns
    a Boolean value based on whether the user wants to save the image
    into the data storage or not.

    Parameters
    ----------
    image       A 3D image volume as a numpy.ndarray. The order of the
                dimensions should be ZXYC. C can be omitted if one.

    Returns     Return True or False, based on whether the user wants
                to keep the image or not.
    -------

    """
    assert os.path.exists(vaa3d_bin)

    filename = "temp.tif"
    tiffile.imsave(filename, image)

    subprocess.call([vaa3d_bin, "-i", filename])

    while True:
        answer = raw_input("Do you want to save the image (yes/no)?")

        if answer in ('yes', 'Yes', 'YES', 'y', 'N'):
            keep = True
            break
        elif answer in ('no', 'No', 'NO', 'n', 'N'):
            keep = False
            break
        else:
            print "Unrecognized answer. Please state yes or no."

    os.remove(filename)
    return keep
