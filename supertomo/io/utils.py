import SimpleITK as sitk
from .tiffile import TiffFile

scale_c = 1e6


def get_imagej_tiff(filename, memmap=False):
    """
    ImageJ has a bit peculiar way of saving image metadata, especially the tags
    for voxel spacing, which is of main interest in SuperTomo. This function reads
    a 3D TIFF into a Numpy array and also calculates the voxel spacing parameters
    from the TIFF tags. I will not guarantee that this will work with any other TIFF
    files.

    :param filename:    Path to a TIFF.
    :param memmap:      Enables Memory mapping in case the TIFF file is too large to
                        be read in memory completely.
    :return:            Image data as a Numpy array, voxel spacing tuple
    """
    assert filename.endswith((".tif", ".tiff"))
    tags = {}
    # Read images and tags
    with TiffFile(filename) as image:
        # Get images
        images = image.asarray(memmap=memmap)
        # Get tags
        page = image[0]
        for tag in page.tags.values():
            tags[tag.name] = tag.value

    # Figure out z-spacing, which in ImageJ is hidden in the "image_description"
    # header (why, one might ask).
    image_descriptor = tags["image_description"].split("\n")
    z_spacing = None
    for line in image_descriptor:
        if "spacing" in line:
            z_spacing = float(line.split("=")[-1])
            break
    assert z_spacing is not None

    # Create a tuple for zxy-spacing. The order of the dimensions follows that of the
    # image data
    spacing = (z_spacing/scale_c, 1.0/tags["x_resolution"][0], 1.0/tags["y_resolution"][0])
    return images, spacing


def get_itk_image(filename):
    """
    A function for reading image file types typical to ITK (mha & mhd). This is mostly
    of a historical significance, because in the original SuperTomo 1 such files were
    used, mostly for convenience.

    :param filename:        Path to an ITK image
    :return:                Image data as a Numpy array, voxel spacing tuple
    """
    assert filename.endswith((".mha", ".mhd"))
    return get_array_from_itk_image(sitk.ReadImage(filename))


def get_itk_image_from_array(array, spacing):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing[::-1])

    return image


def get_array_from_itk_image(image):
    array = sitk.GetArrayFromImage(image)
    # In ITK the order of the dimensions differs from Numpy. The array conversion
    # re-orders the dimensions, but of course the same has to be done to the spacing
    # information.
    spacing_orig = image.GetSpacing()[::-1]
    spacing = tuple(dim/scale_c for dim in spacing_orig)

    return array, spacing