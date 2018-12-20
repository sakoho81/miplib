import os

import h5py

import miplib.ui.utils as uiutils
from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection
from miplib.data.containers.image import Image


class FourierCorrelationDataWriter(object):
    """
    A class for wrtiting Fourier Correlation Data into a file.
    """
    # region Constructor and Destructor
    def __init__(self, output_dir, filename, append=False):

        # Create output dir, if it doesn't exist output dir if
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, filename)

        assert output_path.endswith(".hdf5")

        if os.path.isfile(output_path):
            self.data = h5py.File(output_path, mode="r+" if append else "w")
        else:
            self.data = h5py.File(output_path, mode="w")

    def __del__(self):
        self.close()
    # endregion

    def write_metadata(self, metadata):
        """
        Write a metadata dictionary to the HDF5 files header as a general description of
        the dataset.
        :param metadata: a dictionary with all the necessary data to be written in the file
        """
        assert isinstance(metadata, dict)

        for key, value in metadata:
            self.data.attrs[key] = value

    def write_images(self, images):
        """
        Write images to the data structure
        :param images: a tuple of Image objects, or a single image
        """
        if not isinstance(images, tuple):
            images = tuple(images)
        for image in images:
            assert isinstance(image, Image)

        self.data.create_group("images")

        image_name_prefix = "image_"
        for idx, image in enumerate(images):
            image_name = image_name_prefix+str(idx)
            self.data["images"].create_dataset(image_name, data=image)
            if image.ndim == 2:
                self.data["images"][image_name].attrs["pixel_size"] = "%d %d (yx)" % image.spacing
            else:
                self.data["images"][image_name].attrs["pixel_size"] = "%d %d %d (zyx)" % image.spacing

    def write_data_set(self, data):
        """
        Write Fourier Correlation Data (FRC, FSC etc) to the data structure.
        :param data:
        :type data: FourierCorrelationDataCollection
        """
        assert isinstance(data, FourierCorrelationDataCollection)

        group_prefix = "data_set_"

        for angle, data_set in data:
            group_name = group_prefix + angle
            if group_name in self.data:
                if not uiutils.get_user_input(
                        "The dataset %s already exists in the file structure. Do you want"
                        "to overwrite it?" % angle):
                    continue

                # Create a group fot every dataset and sub-groups for the two dictionaries
                # in the FourierCorrelationData structure.
                data_set_group = self.data.create_group(group_name)
                resolution_group = data_set_group.create_group("resolution")
                correlation_group = data_set_group.create_group("correlation")

                resolution_group.create_dataset("threshold", data=data_set.resolution["threshold"])
                resolution_group.attrs["resolution"] = data_set.resolution["resolution"]
                resolution_group.attrs["resolution-point"] = "%d %d (yx)" % data_set.resolution["resolution-point"]
                resolution_group.attrs["criterion"] = data_set.resolution["criterion"]
                resolution_group.create_dataset("resolution-threshold-coefficients",
                                                data=data_set.resolution["resolution-threshold-coefficients"])

                correlation_group.create_dataset("correlation", data=data_set.correlation["correlation"])
                correlation_group.create_dataset("frequency", data=data_set.correlation["frequency"])
                correlation_group.create_dataset("points-x-bin", data=data_set.correlation["points-x-bin"])
                correlation_group.create_dataset("curve-fit", data=data_set.correlation["curve-fit"])
                correlation_group.create_dataset("curve-fit-coefficients",
                                                 data=data_set.correlation["curve-fit-coefficients"])

    def close(self):
        """
        A function to explicitly close the data file (will be called by the destructor, if not.
        :return:
        """
        self.data.close()

