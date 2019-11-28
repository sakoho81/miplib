import itertools

import numpy as np
import matplotlib.pyplot as plt

from miplib.data.containers.image_data import ImageData

plt.style.use("seaborn-paper")

from miplib.data.containers.array_detector_data import ArrayDetectorData


def make_template_image(data, imagesz=250):
    """
    Makes the "fingerprint" map of the array detector images
    :param data: ArrayDetecorData object with all the iamges
    :param imagesz: integer Size of the images in pixels
    :return: returns a tuple of images (one for each channel)
    """
    assert isinstance(data, ArrayDetectorData)

    blocksz = int(imagesz/np.sqrt(data.ndetectors))

    # First calculate the total photon count for images from each detector
    # and photosensor
    pixels = np.zeros(data.ndetectors*data.ngates)

    data.iteration_axis = 'detectors'
    for idx, image in enumerate(data):
        pixels[idx] = image.sum()

    # Then generate a template image for each photosensor
    container = []
    for gate in range(data.ngates):
        image = np.zeros((imagesz, imagesz))
        idx = 0
        for x, y in itertools.product(range(0, imagesz, blocksz), range(0, imagesz, blocksz)):
            pixel_index = gate*data.ndetectors + idx
            image[x:x + blocksz, y:y + blocksz] = pixels[pixel_index]
            idx += 1

        container.append(image)

    return container


def make_psf_plot(data, size=(5,5)):
    """
    Makes a 5x5 matrix plot of Point-Spread-Functions (PSFs) that are used in the
    blind multi-image APR-ISM image fusion
    :param data: ArrayDetectorData or ImageData object that contains the 25 psfs
    :param size: Size of the plot
    :return:     Returns the figure
    """
    if isinstance(data, ArrayDetectorData):
        hdf = False
    elif isinstance(data, ImageData):
        hdf = True
    else:
        raise ValueError("data needs to be ArrayDetecorData or ImageData object")

    fig, axs = plt.subplots(5, 5, figsize=size)

    for idx, ax in enumerate(axs.flatten()):
        if hdf:
            data.set_active_image(idx, 0, 100, "psf")
            ax.imshow(data[:], cmap="hot")
        else:
            ax.imshow(data[0,idx])

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.tight_layout(pad=-0.3, w_pad=-0.3, h_pad=-0.3)

    return fig


def calculate_theoretical_shifts_xy(pitch, magnification, alpha=0.5, width=5):
    """ Calculate theoretical ISM shift matrix, based on detector pixel pitch
    and 
    
    Arguments:
        pitch {float} -- Distance between two detector elements. 
        
        magnification {float} -- Total magnification from object plane to the 
        detector plane.
    
    Keyword Arguments:
        width {int} -- Width of the detector (number of pixels) (default: {5})
        alpha {float} -- the reassignment factor (default: {0.5})
    
    Returns:
        [list(float, float)] -- Returns a list of the y and x coordinates of
        the image offsets
    """
    
    pitch_pt = pitch*alpha/magnification

    radius = width//2
    axis = np.linspace(-pitch_pt*radius, pitch_pt*radius, width)
    y_pt, x_pt = np.meshgrid(axis,axis)

    x_pts = list(x_pt.ravel())
    y_pts = list(y_pt.ravel())[::-1]

    return y_pts, x_pts

