import os
#import subprocess

import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# import miplib.data.io.tiffile as tiffile
#
# vaa3d_bin = "/home/sami/bin/Vaa3D_Ubuntu_64bit_v3.200/vaa3d"
#
# def evaluate_3d_image(data):
#     """
#     A utility function that can be used to display the registration
#     and/or fusion results in Vaa3D volume viewer. The function returns
#     a Boolean value based on whether the user wants to save the data
#     into the data storage or not.
#
#     Parameters
#     ----------
#     data       A 3D data volume as a numpy.ndarray. The order of the
#                 dimensions should be ZXYC. C can be omitted if one.
#
#     """
#     assert os.path.exists(vaa3d_bin)
#
#     filename = "temp.tif"
#     tiffile.imsave(filename, data)
#
#     subprocess.call([vaa3d_bin, "-i", filename])
#
#     os.remove(filename)


# callback invoked by the interact ipython method for scrolling through the data stacks of
# the two images (moving and fixed)
def display_3d_slices(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # create a figure with two subplots and the specified size
    plt.subplots(1, 2, figsize=(10, 8))

    # draw the fixed data in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap='gray')
    plt.title('fixed data')
    plt.axis('off')

    # draw the moving data in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap='gray')
    plt.title('moving data')
    plt.axis('off')

# callback invoked by the ipython interact method for scrolling and modifying the alpha blending
# of an data stack of two images that occupy the same physical space.


def display_3d_slice_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
    plt.imshow(sitk.GetArrayFromImage(img), cmap='gray')
    plt.axis('off')


def create_axial_views_plot(image, x_idx, y_idx, z_idx):

    assert issubclass(image.__class__, np.ndarray)
    assert image.ndim == 3

    xy = image[z_idx, :, :]
    xz = image[:, y_idx, :]
    yz = image[:, :, x_idx]

    yz = np.transpose(yz)

    width_ratio = xy.shape[1]/yz.shape[1]
    height_ratio = xy.shape[0]/xz.shape[0]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[width_ratio, 1],
                           height_ratios=[height_ratio, 1])

    ax0 = plt.subplot(gs[0, 0])
    ax0.imshow(xy, cmap="hot")
    ax0.set_title("XY")
    ax0.axis('off')

    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(yz, cmap="hot")
    ax1.set_title("YZ")
    ax1.axis('off')

    ax2 = plt.subplot(gs[1,0])
    ax2.imshow(xz, cmap="hot")
    ax2.set_title("XZ")
    ax2.axis('off')

    #fig.delaxes(axes[1, 1])
    return fig


def display_2d_images(image1,
                      image2,
                      image1_title='image1',
                      image2_title='image2',
                      vertical=False):
    """
    A function that can be used to display two SimpleITK images side by side.
    It is also possible to select paired landmarks from the two images, by
    enabling the landmarks argument.

    Parameters

    image1      A numpy.ndarray or its subclass
    image2      A numpy.ndarray or its subclass

    """
    assert issubclass(type(image1), np.ndarray)
    assert issubclass(type(image2), np.ndarray)

    assert image1.ndim == 2 and image2.ndim == 2

    if vertical:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(13, 10),
            gridspec_kw = {'height_ratios':[3, 1], 'width_ratios': [1, 1]}
        )
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))

    # draw the fixed data in the first subplot
    ax1.imshow(image1, cmap="hot")
    ax1.set_title(image1_title)
    ax1.axis('off')

    # draw the moving data in the second subplot
    ax2.imshow(image2, cmap="hot")
    ax2.set_title(image2_title)
    ax2.axis('off')

    plt.show()


def display_2d_image(image):
    """
    A function that can be used to display two SimpleITK images side by side.
    It is also possible to select paired landmarks from the two images, by
    enabling the landmarks argument.

    Parameters

    data       a Numpy array or a SimpleITk data object

    """

    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)

    assert image.ndim == 2

    plt.imshow(image, cmap="rainbow")
    plt.axis('off')
    plt.show()


def display_2d_slices_with_alpha(alpha, fixed, moving):
    img = (1.0 - alpha) * fixed + alpha * moving
    plt.imshow(sitk.GetArrayFromImage(img), cmap='gray')
    plt.axis('off')


def display_2d_image_overlay(image1, image2, image3=None):
    '''
    Overlays 2-3 images into a single RGB plot. This was intended for use in
    evaluating registration results.
    Parameters
    ----------
    image1      A 2D numpy.array or sitk.Image that
    image2      A 2D numpy.array or sitk.Image that
    image3      A 2D numpy.array or sitk.Image that

    Returns     Nothing
    -------

    '''
    if isinstance(image1, sitk.Image):
        image1 = sitk.GetArrayFromImage(image1)
    if isinstance(image2, sitk.Image):
        image2 = sitk.GetArrayFromImage(image2)

    if image1.shape != image2.shape:
        raise ValueError("The dimensions of the images to be overlaid should match")

    if image3 is None:
        image3 = np.zeros(image1.shape, dtype=np.uint8)

    rgb_image = np.concatenate([aux[..., np.newaxis] for aux in (image1, image2, image3)], axis=-1)

    plt.imshow(rgb_image)
    plt.axis('off')
    plt.show()


def show_pics_from_disk(filenames, title="Image collage"):
    """
    A utility for creating a collage of images, to be shown
    in a single plot. The images are loaded from disk according
    to the provided filenames:
    :param filenames:   A list containing the data filenames
    :param title:       Name of the plot
    :return:            Nothing
    """
    if len(filenames) > 1:
        if 4 < len(filenames) <= 9:
            fig, subplots = plt.subplots(3, 3)
        elif 9 < len(filenames) <= 16:
            fig, subplots = plt.subplots(4, 4)
        elif 16 < len(filenames) <= 25:
            fig, subplots = plt.subplots(5, 5)
        elif 25 < len(filenames) <= 36:
            fig, subplots = plt.subplots(6, 6)
        else:
            fig, subplots = plt.subplots(2, 2)

        # fig.title(title)
        i = 0
        j = 0
        k = 0
        while k < len(filenames):
            j = 0
            while j < subplots.shape[1] and k < len(filenames):
                print(filenames[i + j])
                subplots[i, j].imshow(plt.imread(filenames[k]), cmap='hot')
                subplots[i, j].set_title(os.path.basename(filenames[k]))
                subplots[i, j].axis("off")
                k += 1
                j += 1
            i += 1
        plt.subplots_adjust(wspace=-0.5, hspace=0.2)
        plt.suptitle(title, size=16)
        plt.show()

    else:
        plt.imshow(plt.imread(filenames))
        plt.axis("off")
        plt.show()

