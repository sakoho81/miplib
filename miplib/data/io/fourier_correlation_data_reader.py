import os

import h5py

from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection, FourierCorrelationData
from miplib.data.containers.image import Image


class FourierCorrelationDataReader(object):
    """
    A class for writing Fourier Correlation Data into a file.
    """

    # region Constructor and Destructor
    def __init__(self, file_path):

        # Create output dir, if it doesn't exist output dir if
        if not os.path.isfile(file_path) or not file_path.endswith(".hdf5"):
            raise ValueError("Not a valid filename: %s" % file_path)

        self.data = h5py.File(file_path, mode="r")

    def __del__(self):
        self.close()

    # endregion

    def read_metadata(self):
        """
        Read a metadata dictionary from the HDF5 files header
        """
        return dict(self.data.attrs)

    def read_images(self, index=None):
        """
        Read images from the data structure.
        :returns images: a tuple of Image objects, or a single image
        """

        if "images" not in self.data:
            raise ValueError("No images to read")

        if index is not None:
            image_name = "image_%i" % index
            data_set = self.data["images"][image_name]
            spacing = data_set.attrs["pixel_size"].split()[0:len(data_set.shape)]
            return Image(data_set[:], spacing)

        images = []
        for data_set in self.data["images"]:
            spacing = data_set.attrs["pixel_size"].split()[0:len(data_set.shape)]
            images.append(Image(data_set[:], spacing))

        return images

    def read_data_set(self):
        """
        Read Fourier Correlation Data file (FRC, FSC etc) to the FourierCorrelationDataCollection
        data structure.
        :returns FourierCorrelationDataCollection
        """
        group_prefix = "data_set_"
        data_sets = FourierCorrelationDataCollection()
        for group_name in list(self.data.keys()):
            if group_prefix in group_name:
                angle = group_name.split("_")[-1]
                data_set = FourierCorrelationData()
                resolution_group = self.data[group_name]["resolution"]
                correlation_group = self.data[group_name]["correlation"]

                data_set.resolution["threshold"] = resolution_group["threshold"][:]
                data_set.resolution["resolution-point"] = \
                    resolution_group.attrs["resolution-point"].split()[:-1]
                data_set.resolution["criterion"] = resolution_group.attrs["criterion"]
                data_set.resolution["resolution-threshold-coefficients"] = \
                    resolution_group["resolution-threshold-coefficients"][:]

                data_set.correlation["correlation"] = correlation_group["correlation"][:]
                data_set.correlation["frequency"] = correlation_group["frequency"][:]
                data_set.correlation["points-x-bin"] = correlation_group["points-x-bin"][:]
                data_set.correlation["curve-fit"] = correlation_group["curve-fit"][:]
                data_set.correlation["curve-fit-coefficients"] = \
                    correlation_group["curve-fit-coefficients"][:]

                data_sets[int(angle)] = data_set

        return data_sets

    def close(self):
        """
        A function to explicitly close the data file (will be called by the destructor, if not.
        :return:
        """
        self.data.close()

