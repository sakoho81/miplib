import numpy as np
from ..containers.array_detector_data import ArrayDetectorData
from ..containers.image_data import ImageData
from ..containers.image import Image


def convert_to_image(data):
    """
    Convert ArrayDetectorData into an Image (a Numpy array)
    :param data: data to convert
    :return: an Image object (gate, channel, z, y, x)
    """


    gates = data.ngates
    channels = data.ndetectors

    dtype = data[0, 0].dtype
    ndims = data[0, 0].ndim
    shape = (1, ) * (3 - ndims) + data[0, 0].shape
    spacing = (1,) * (3 - ndims) + tuple(data[0, 0].spacing)

    im_shape = (gates, channels,) + shape
    im_data = np.zeros(im_shape, dtype=dtype)

    for gate_idx in range(gates):
        for channel_idx in range(channels):
            channel_im = data[gate_idx, channel_idx]
            if ndims < 3:
                new_shape = (1,) * (3 - ndims) + channel_im.shape
                channel_im = np.reshape(channel_im, new_shape)
            im_data[gate_idx, channel_idx] = channel_im

    return Image(im_data, spacing)


def convert_to_imagedata(data, path, data_type="original"):
    """
    A very simple converter to save ArrayDetectorData into the HDF5 data structure
    used in MIPLIB for large image datasets.

    :param data: an ArrayDetectorDAta object
    :param path: full path to the new hdf5 file. Should end with .hdf5
    :return: returns a handle to the new hdf5 file. The file is left open, so remember
    to call close() method in order to ensure that all the data is written on the disk.
    """
    image_data = ImageData(path)

    for gate_idx in range(data.ngates):
        for det_idx in range(data.ndetectors):
            temp = data[gate_idx, det_idx]
            if data_type == "original":
                image_data.add_original_image(temp, 100, det_idx, gate_idx, 0, temp.spacing)
            elif data_type == "registered":
                image_data.add_registered_image(temp, 100, det_idx, gate_idx, 0, temp.spacing)
            elif data_type == "psf":
                image_data.add_psf(temp, 100, det_idx, gate_idx, 0, temp.spacing)

    return image_data

def convert_to_numpy(data):
    """
    Convert ArrayDetectorData into a Numpy array.

    :param data: the object to be converted
    :type data: ArrayDetectorData

    :return: the data array, organized as (gate, detector, z, y, x). If a two-dimensional
    dataset is provided, the return shape is the same (len(z)=1).

    """
    assert isinstance(data, ArrayDetectorData)

    # Get image shape
    n_dim = data[0,0].ndim
    if n_dim == 2:
        image_shape = (1,) + data[0,0].shape
    elif n_dim == 3:
        image_shape = data[0,0].shape
    else:
        raise ValueError(f"Unsupported array shape ({data[0,0].shape})")
    
    # Initialize new Numpy array
    array_shape = (data.ngates, data.ndetectors) + image_shape
    array = np.zeros(array_shape, dtype=data[0,0].dtype)

    # Copy values
    for gate_idx in range(data.ngates):
        for det_idx in range(data.ndetectors):
            if n_dim == 2:
                array[gate_idx, det_idx, 0] = data[gate_idx, det_idx]
            else:
                array[gate_idx, det_idx] = data[gate_idx, det_idx]

    image_spacing = data[0,0].spacing
    
    return array, image_spacing
            


