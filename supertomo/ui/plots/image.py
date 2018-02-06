import subprocess
import supertomo.data.io.tiffile as tiffile
import os

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

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

    """
    assert os.path.exists(vaa3d_bin)

    filename = "temp.tif"
    tiffile.imsave(filename, image)

    subprocess.call([vaa3d_bin, "-i", filename])

    os.remove(filename)


# callback invoked by the interact ipython method for scrolling through the image stacks of
# the two images (moving and fixed)
def display_3d_slices(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # create a figure with two subplots and the specified size
    plt.subplots(1, 2, figsize=(10, 8))

    # draw the fixed image in the first subplot
    plt.subplot(1, 2, 1)
    plt.imshow(fixed_npa[fixed_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('fixed image')
    plt.axis('off')

    # draw the moving image in the second subplot
    plt.subplot(1, 2, 2)
    plt.imshow(moving_npa[moving_image_z, :, :], cmap=plt.cm.Greys_r)
    plt.title('moving image')
    plt.axis('off')

# callback invoked by the ipython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.


def display_3d_slice_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha) * fixed[:, :, image_z] + alpha * moving[:, :, image_z]
    plt.imshow(sitk.GetArrayFromImage(img), cmap=plt.cm.Greys_r)
    plt.axis('off')


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
            2, 1, figsize=(10, 8),
            gridspec_kw = {'height_ratios':[3, 1], 'width_ratios':[1, 1]}
        )
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    # draw the fixed image in the first subplot
    ax1.imshow(image1, cmap=plt.cm.Greys_r)
    ax1.set_title(image1_title)
    ax1.axis('off')

    # draw the moving image in the second subplot
    ax2.imshow(image2, cmap=plt.cm.Greys_r)
    ax2.set_title(image2_title)
    ax2.axis('off')

    plt.show()


def display_2d_image(image):
    """
    A function that can be used to display two SimpleITK images side by side.
    It is also possible to select paired landmarks from the two images, by
    enabling the landmarks argument.

    Parameters

    image       a Numpy array or a SimpleITk image object

    """

    if isinstance(image, sitk.Image):
        image = sitk.GetArrayFromImage(image)

    assert image.ndim == 2

    plt.imshow(image)
    plt.axis('off')
    plt.show()


def display_2d_slices_with_alpha(alpha, fixed, moving):
    img = (1.0 - alpha) * fixed + alpha * moving
    plt.imshow(sitk.GetArrayFromImage(img), cmap=plt.cm.Greys_r)
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


