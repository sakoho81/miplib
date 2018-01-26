import numpy as np
import supertomo.data.containers.myimage as myimage


def zoom_to_isotropic_spacing(image):  # type: (myimage.MyImage) -> None
    spacing = image.get_spacing()
    shape = image.get_dimensions()
    min_spacing = min(spacing)
    zoom = tuple(pixel_spacing/min_spacing for pixel_spacing in spacing)
    if all(zoom) == 1:
        return
    else:
        new_shape = tuple(int(pixels*dim_zoom)for (pixels, dim_zoom) in zip(shape, zoom))
        resize(image, new_shape)


def resize(image, size, order=3):  # type: (myimage.MyImage, tuple) -> None
    """
    Resize the image, using interpolation.

    :param order: The interpolation type defined as order of the b-spline
    :param image: The MyImage object.
    :param size: A tuple of new image dimensions.

    """
    assert isinstance(size, tuple)
    zoom = [float(a)/b for a, b in zip(size, image.get_dimensions())]
    print "The zoom is %s" % zoom

    image.images = np.zoom(image, tuple(zoom), order=order)
    image.set_spacing(i/j for i, j in zip(image.get_spacing(), zoom))


def apply_hanning(image):  # type: (myimage.MyImage) -> None
    """
    Apply Hanning window to the image.

    :return:
    """
    npix = image.get_dimensions()
    window = np.hanning(npix)
    window = np.outer(window, window)

    image *= window


