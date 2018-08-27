from ..containers.array_detector_data import ArrayDetectorData
from ..containers.image_data import ImageData


def convert_to_imagedata(data, path, data_type="original"):
    """
    A very simple converter to save ArrayDetectorData into the HDF5 data structure
    used in MIPLIB for large image datasets.

    :param data: an ArrayDetectorDAta object
    :param path: full path to the new hdf5 file. Should end with .hdf5
    :return: returns a handle to the new hdf5 file. The file is left open, so remember
    to call close() method in order to ensure that all the data is written on the disk.
    """
    assert isinstance(data, ArrayDetectorData)

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