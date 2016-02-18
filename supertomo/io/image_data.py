import os
import h5py
import numpy

image_types = ("original", "registered", "fused")


class ImageData():
    """
    The data storage in SuperTomo is based on a HDF5 file format. This
    allows the efficient handing of large datasets
    """

    def __init__(self, path):

        if os.path.isfile(path) and path.endswith(".hdf5"):
            self.data = h5py.File(path, mode="r+")
        else:
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            assert path.endswith(".hdf5")

            self.data = h5py.File(path, mode="w")
            self.data.create_group("info")

            self.series_count = 0
            self.channel_count = 1

        self.active_image = None

    def add_original_image(self, data, rotation_angle, spacing, chunk_size=None):
        """
        Add a source image to the HDF5 file. This function is typically used
        when the HDF5 file is created from source image files.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.

        :param rotation_angle:  Estimated rotation angle of the view, in
                                respect to the regular STED angle
        :param spacing:         Voxel size
        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        # Group images by rotation angle
        group_name = "original/" + str(self.series_count)
        image_group = self.data.create_group(group_name)

        image_group["spacing"] = spacing
        image_group["rotation_angle"] = rotation_angle

        if len(data.shape) == 4:
            self.channel_count = data.shape[0]
            for channel in range(0, data.shape[0]):
                if chunk_size is None:
                    image_group.create_dataset(str(channel), data=data[channel])
                else:
                    image_group.create_dataset(str(channel), data=data[channel], chunks=chunk_size)
        else:
            if chunk_size is None:
                image_group.create_dataset("0", data=data)
            else:
                image_group.create_dataset("0", data=data, chunks=chunk_size)

        self.series_count += 1
        self.data["series_count"] = self.series_count

    def add_registered_image(self, data, transform, spacing, chunk_size=None):
        """
        Add a resampled, registered image.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.

        :param transform:       The spatial transform that was used to generate
                                the resampled image.

        :param spacing:         Voxel size
        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        # Group images by rotation angle
        group_name = "registered/" + str(self.active_image)
        image_group = self.data.create_group(group_name)

        image_group["spacing"] = spacing
        image_group["transform"] = transform

        if len(data.shape) == 4:
            for channel in range(0, data.shape[0]):
                if chunk_size is None:
                    image_group.create_dataset(str(channel), data=data[channel])
                else:
                    image_group.create_dataset(str(channel), data=data[channel], chunks=chunk_size)
        else:
            if chunk_size is None:
                image_group.create_dataset("0", data=data)
            else:
                image_group.create_dataset("0", data=data, chunks=chunk_size)

    def add_fused_image(self, data, spacing):
        """
        Add a fused image.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.
        :param spacing:         Voxel size
        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        # Group images by rotation angle
        image_group = self.data.create_group("fused")

        image_group["spacing"] = spacing
        image_group["transform"] = transform

        if len(data.shape) == 4:
            for channel in range(0, data.shape[0]):
                image_group.create_dataset(str(channel), data=data[channel])
        else:
            image_group.create_dataset("0", data=data)

    def get_rotation_angle(self):
        return self.data[self.active_image]["rotation_angle"]

    def get_voxel_size(self):
        return self.data[self.active_image]["spacing"]

    def set_active_image(self, index, type, channel):
        """
        Select which view is currently active.

        :param index:   View index, goes from 0 to number of views - 1
        """
        if index >= self.series_count:
            print "Invalid index. There are only %i images in the file" % self.series_count
            return
        elif type not in image_types:
            print "Unkown image type."
            return
        else:
            self.active_image = type + "/" + str(index) + "/" + str(channel)

    def close(self):
        """
        Close the file object.
        """
        self.data.close()

    def __getitem__(self, item):
        return self.data[self.active_image][item]

    def __setitem__(self, key, value):
        self.data[self.active_image][key] = value