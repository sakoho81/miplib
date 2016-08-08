import os
import h5py
import numpy
import scipy.ndimage as ndimage
import SimpleITK as sitk

from ..utils import itkutils
from ..definitions import *


class ImageData():
    """
    The data storage in SuperTomo is based on a HDF5 file format. This
    allows the efficient handing of large datasets
    """

    def __init__(self, path):

        if not os.path.exists(os.path.dirname(path)):
            os.mkdir(os.path.dirname(path))
        assert path.endswith(".hdf5")

        if os.path.exists(path):
            self.data = h5py.File(path, mode="r+")
            self.series_count = self.data.attrs["series_count"]
            self.channel_count = self.data.attrs["channel_count"]
        else:
            self.data = h5py.File(path, mode="w")
            self.series_count = 0
            self.channel_count = 1

            self.data.attrs["series_count"] = self.series_count
            self.data.attrs["channel_count"] = self.channel_count

        self.active_image = None

    def add_original_image(self, data, angle, spacing, index, scale, channel, chunk_size=None):
        """
        Add a source image to the HDF5 file. This function is typically used
        when the HDF5 file is created from source image files.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.

        :param angle:  Estimated rotation angle of the view, in
                                respect to the regular STED angle
        :param spacing:         Voxel size
        :param index            The image ordering index
        :param scale            Percentage from full size. It is possible to save
                                multiple versions of an image in different sizes.
        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        if channel > self.channel_count + 1:
            raise ValueError("Add the color channels in the correct order")

        # Create a new image group, based on the ordering index. If the
        # group exists, an attempt is made to add a new dataset.
        group_name = "original/" + index
        if group_name not in self.data:
            image_group = self.data.create_group(group_name)
            self.series_count += 1
            self.data.attrs["series_count"] = self.series_count
        else:
            image_group = self.data[group_name]

        name = "channel_" + str(channel) + "_scale_" + str(scale)

        # Don't overwrite an existing image.
        if name in image_group:
            return

        # Zoom axial dimension for isotropic pixel size.
        if spacing[0] != spacing[1]:
            print "Image %s needs to be resampled for isotropic spacing." \
                  "This will take a minute" % name
            z_zoom = spacing[0] / spacing[1]
            data = ndimage.zoom(data, (z_zoom, 1, 1), order=3)
            spacing = tuple(spacing[x] if x != 0 else z_zoom for x in len(spacing))

        # Activate chunked storage of requested
        if chunk_size is None:
            image_group.create_dataset(name, data=data[channel])
        else:
            image_group.create_dataset(name, data=data[channel], chunks=chunk_size)

        image_group[name].attrs["angle"] = angle
        image_group[name].attrs["spacing"] = spacing
        image_group[name].attrs["size"] = data.shape

    def add_registered_image(self, data, transform, transform_type, channel,
                             index, scale, spacing, chunk_size=None):
        """
        Add a resampled, registered image.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.

        :param transform:       The spatial transform that was used to generate
                                the resampled image.

        :param transform_type   An integer of the transform type. Is based on the
                                sitk TransformEnum type (see .utils.py)

        :param spacing:         Voxel size
        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        group_name = "registered/" + index
        if group_name not in self.data:
            image_group = self.data.create_group(group_name)
        else:
            image_group = self.data[group_name]

        if channel > 0:
            if self.channel_count == 1:
                raise ValueError("Invalid channel count")

        name = "channel_" + str(channel) + "_scale_" + str(scale)
        if name in image_group:
            return

        if chunk_size is None:
            image_group.create_dataset(name, data=data)
        else:
            image_group.create_dataset(name, data=data, chunks=chunk_size)

        # Each image has its own attributes
        image_group[name].attrs["spacing"] = spacing
        image_group[name].attrs["size"] = data.shape

        if transform is not None:
            assert issubclass(transform, sitk.Transform)
            image_group[name].attrs["tfm_type"] = transform_type
            image_group[name].attrs["tfm_params"] = transform.GetParameters()
            image_group[name].attrs["tfm_fixed_params"] = transform.GetFixedParameters()

    def add_psf(self, data, angle, channel, index, scale, spacing, chunk_size=None):
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        group_name = "psf/" + index
        if group_name not in self.data:
            image_group = self.data.create_group(group_name)
        else:
            image_group = self.data[group_name]

        if channel > 0:
            if self.channel_count == 1:
                raise ValueError("Invalid channel count")

        name = "channel_" + str(channel) + "_scale_" + str(scale)
        if name in image_group:
            return

        if chunk_size is None:
            image_group.create_dataset(name, data=data)
        else:
            image_group.create_dataset(name, data=data, chunks=chunk_size)

        image_group[name].attrs["angle"] = angle
        image_group[name].attrs["spacing"] = spacing
        image_group[name].attrs["size"] = data.shape

    def add_transform(self, scale, index, channel, params, fixed_params, transform_type):
        name = "registered/" + index + "/channel_" + channel + "_scale_" + scale
        if name not in self.data:
            raise ValueError("Dataset %s does not exist" % name)
        self.data[name].attrs["tfm_type"] = transform_type
        self.data[name].attrs["tfm_params"] = params
        self.data[name].attrs["tfm_fixed_params"] = fixed_params

    def create_rescaled_images(self, scale, chunk_size):
        """
        Creates rescaled versions of original images. Typically downscaling would
        be used to speed up certain image processing tasks. The scaled images
        are saved directly into the HDF5 file.

        Parameters
        ----------
        scale       Scale as a fraction of the original image size, i.e. scale 0.5
                    will create images 50% from the full size image.
        chunk_size  The same as with the other images. Can be used to define a
                    particular chunk size for data storage.

        Returns
        -------

        """
        # Iterate over all the images
        for index in range(self.series_count):
            group_name = "original/" + str(index)
            image_group = self.data[group_name]
            # Iterate over channels
            for channel in range(self.channel_count):
                name_new = "channel_" + str(channel) + "_scale_" + str(scale)
                name_ref = "channel_" + str(channel) + "_scale_100"
                # Check if exists and create if not.
                if name_new in image_group:
                    continue
                else:
                    spacing = scale*image_group[name_ref].attrs["spacing"]
                    zoom = (scale, scale, scale)
                    data = ndimage.zoom(image_group[name_ref], zoom, order=3)
                    if chunk_size is None:
                        image_group.create_dataset(name_new, data=data)
                    else:
                        image_group.create_dataset(name_new, data=data, chunks=chunk_size)

                    image_group[name_new].attrs["angle"] = image_group[name_ref].attrs["angle"]
                    image_group[name_new].attrs["spacing"] = spacing
                    image_group[name_new].attrs["size"] = data.shape

    def add_fused_image(self, data, channel, scale, spacing):
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

        if "fused" not in self.data:
            image_group = self.data.create_group("fused")
        else:
            image_group = self.data["fused"]

        if channel > 0 and self.channel_count == 1:
                raise ValueError("Invalid channel count")

        name = "channel" + str(channel) + "scale" + str(scale)
        if name in image_group:
            return

        image_group.create_dataset(name, data=data)
        image_group[name].attrs["spacing"] = "spacing"

    def get_rotation_angle(self):
        return self.data[self.active_image].attrs["angle"]

    def get_voxel_size(self):
        return self.data[self.active_image].attrs["spacing"]

    def get_image_size(self):
        return self.data[self.active_image].attrs["size"]

    def get_dtype(self):
        return self.data[self.active_image].dtype

    def get_number_of_images(self, type):
        assert type in image_types_c
        return len(self.data[type])

    def get_transform(self):
        assert "registered" in self.active_image
        tfm_type = self.data[self.active_image].attrs["tfm_type"]
        tfm_params = self.data[self.active_image].attrs["tfm_params"]
        tfm_fixed_params = self.data[self.active_image].attrs["tfm_fixed_params"]

        return itkutils.make_itk_transform(tfm_type,
                                           3,
                                           tfm_params,
                                           tfm_fixed_params)

    def set_active_image(self, index, channel, scale, image_type):
        """
        Select which view is currently active.

        :param index:      View index, goes from 0 to number of views - 1
        :param image_type  Image type as a string, listed in image_types
        """
        if index >= self.series_count:
            print "Invalid index. There are only %i images in the file" % self.series_count
            return
        elif image_type not in image_types_c:
            print "Unkown image type."
            return
        else:
            self.active_image = image_type + "/" + str(index) + "/channel_" + str(channel) + "_scale_" + str(scale)
            if self.active_image not in self.data:
                raise ValueError("No such image: %s" % self.active_image)


    def set_fused_block(self, block, start_index):
        assert isinstance(block, numpy.ndarray) and isinstance(start_index, numpy.ndarray)
        stop_index = start_index + block.shape
        self.data["fused"][start_index:stop_index] = block

    def get_registered_block(self, block_size, start_index):
        assert isinstance(block_size, numpy.ndarray)
        assert isinstance(start_index, numpy.ndarray)

        assert "registered" in self.active_image, "You must specify a registered image"

        image_size = self.get_image_size()
        end_index= start_index + block_size

        if (image_size > end_index).all():
            block = self.data[self.active_image][
                    start_index[0]:end_index[0],
                    start_index[1]:end_index[1],
                    start_index[2]:end_index[2]
                    ]
            return block, block_size
        else:
            block = numpy.zeros(block_size)
            block_crop = end_index - image_size
            block_crop[block_crop < 0] = 0
            block_size -= block_crop
            end_index = start_index + block_size

            block[0:block_size[0], 0:block_size[1], 0:block_size[2]] = self.data[
                                                                      start_index[0]:end_index[0],
                                                                      start_index[1]:end_index[1],
                                                                      start_index[2]:end_index[2]
                                                                      ]
            return block, block_size

    def close(self):
        """
        Close the file object.
        """
        self.data.attrs["series_count"] = self.series_count
        self.data.attrs["channel_count"] = self.channel_count

        self.data.close()

    def check_if_exists(self, type, index, channel, scale):
        name = type + "/" + str(index) + "/channel_" + str(channel) + "_scale_" + str(scale)
        return name in self.data

    def __getitem__(self, item):
        return self.data[self.active_image][item]

    def __setitem__(self, key, value):
        self.data[self.active_image][key] = value