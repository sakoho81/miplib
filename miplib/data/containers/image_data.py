import os

import h5py
import numpy
import scipy.ndimage as ndimage

import miplib.processing.itk as itkutils
import miplib.ui.utils as uiutils
from miplib.data.containers.image import Image
from miplib.data.definitions import *
import miplib.processing.ndarray as arrayutils


class ImageData(object):
    """
    The data storage in miplib is based on a HDF5 file format. This
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

    def add_original_image(self, data, scale, index, channel, angle, spacing, chunk_size=None):
        """
        Add a source image to the HDF5 file.

        Parameters
        ----------
        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.
        :param scale            Percentage from full size. It is possible to save
                                multiple versions of an image in different sizes.
        :param index            The image ordering index
        :param channel          The color channel to be associated with the image.
        :param angle:           Estimated rotation angle of the view, in
                                respect to the regular STED angle
        :param spacing:         Voxel size

        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        # if int(channel) > self.channel_count + 1:
        #     raise ValueError("Add the color channels in the correct order")

        # Create a new image group, based on the ordering index. If the
        # group exists, an attempt is made to add a new dataset.
        group_name = "original/" + str(index)
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
        if data.ndim == 3 and spacing[0] != spacing[1]:
            print("Image index %s needs to be resampled for isotropic spacing." \
                  "This will take a minute" % index)
            z_zoom = spacing[0] / spacing[1]
            data = ndimage.zoom(data, (z_zoom, 1, 1), order=3)
            spacing = tuple(spacing[x] if x != 0 else spacing[x]/z_zoom for x in range(len(spacing)))

        # Activate chunked storage of requested
        if chunk_size is None:
            image_group.create_dataset(name, data=data)
        else:
            image_group.create_dataset(name, data=data, chunks=chunk_size)

        image_group[name].attrs["angle"] = angle
        image_group[name].attrs["spacing"] = spacing
        image_group[name].attrs["size"] = data.shape

        # # The first image is the same in the registered group as well,
        # # so a soft link will be created here.
        # if int(index) == 0:
        #     reg_group_name = "registered/" + index
        #     reg_group = self.data.create_group(reg_group_name)
        #     reg_group[name] = image_group[name]

    def add_registered_image(self, data, scale, index, channel, angle, spacing, chunk_size=None):
        """
        Add a registered/resampled image to the HDF5 file.

        Parameters
        ----------
        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.
        :param scale            Percentage from full size. It is possible to save
                                multiple versions of an image in different sizes.
        :param index            The image ordering index
        :param channel          The color channel to be associated with the image.
        :param angle:           Estimated rotation angle of the view, in
                                respect to the regular STED angle
        :param spacing:         Voxel size

        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        group_name = "registered/" + str(index)
        if group_name not in self.data:
            image_group = self.data.create_group(group_name)
        else:
            image_group = self.data[group_name]

        # if channel > 0:
        #     if self.channel_count == 1:
        #         raise ValueError("Invalid channel count")

        name = "channel_" + str(channel) + "_scale_" + str(scale)
        if name in image_group:
            if uiutils.get_user_input("The dataset %s already exists in image "
                                      "group %s. Do you want to overwrite "
                                      "it? " % (name, group_name)):
                del image_group[name]
            else:
                return

        if chunk_size is None:
            image_group.create_dataset(name, data=data)
        else:
            image_group.create_dataset(name, data=data, chunks=chunk_size)

        # Each image has its own attributes
        image_group[name].attrs["angle"] = angle
        image_group[name].attrs["spacing"] = spacing
        image_group[name].attrs["size"] = data.shape

    def add_psf(self, data, scale, index, channel, angle, spacing, chunk_size=None,
                calculated=False):
        """
        Add a PSF image to the HDF5 file.

        Parameters
        ----------
        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.
        :param scale            Percentage from full size. It is possible to save
                                multiple versions of an image in different sizes.
        :param index            The image ordering index
        :param channel          The color channel to be associated with the image.
        :param angle:           Estimated rotation angle of the view, in
                                respect to the regular STED angle
        :param spacing:         Voxel size

        :param chunk_size:      A specific chunk size can be defined here in
                                order to optimize the data access, when
                                working with partial images.
        :return:
        """

        assert isinstance(data, numpy.ndarray), "Invalid data format."

        group_name = "psf/" + str(index)
        if group_name not in self.data:
            image_group = self.data.create_group(group_name)
        else:
            image_group = self.data[group_name]

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
        image_group[name].attrs["calculated"] = calculated

    def add_transform(self, scale, index, channel, params, fixed_params, transform_type):
        """
        Adds a spatial transformation as an attribute to the corresponding registered
        view. This means that the registered/resampled image has to be added first, otherwise
        an error is raised.

        Parameters
        ----------
        :param scale            The scale, index and channel parameters identify the
        :param index            registered view, the transform is associated with.
        :param channel

        :param params           The transform parameters. Usually what comes out of
                                transform.GetParameters()
        :param fixed_params     The transform fixed parameters (origin). Usually
                                what comes out of transform.GetFixedParameters()
        :param transform_type
        """
        name = "registered/" + str(index) + "/channel_" + str(channel) + "_scale_" + str(scale)

        if name not in self.data:
            raise ValueError("Dataset %s does not exist" % name)

        self.data[name].attrs["tfm_type"] = transform_type
        self.data[name].attrs["tfm_params"] = params
        self.data[name].attrs["tfm_fixed_params"] = fixed_params

    def add_fused_image(self, data, channel, scale, spacing):
        """
        Add a fused image.

        :param data:            Contains an image from a single rotation angle.
                                The data should be in numpy.ndarray format.
                                Multi-channel data is expected to be a 4D array
                                in which the color channel is the first
                                dimension.
        :param channel          The color channel to be associated with the image.
        :param scale            Percentage from full size. It is possible to save
                                multiple versions of an image in different sizes.
        :param spacing:         Voxel size

        """
        assert isinstance(data, numpy.ndarray), "Invalid data format."

        if "fused" not in self.data:
            image_group = self.data.create_group("fused")
        else:
            image_group = self.data["fused"]

        if channel > 0 and self.channel_count == 1:
            raise ValueError("Invalid channel count")

        name = "channel_" + str(channel) + "_scale_" + str(scale)
        if name in image_group:
            return

        image_group.create_dataset(name, data=data)
        image_group[name].attrs["spacing"] = spacing

    def create_rescaled_images(self, type, scale, chunk_size=None):
        """
        Creates rescaled versions of images of a given type. The scaled images
        are saved directly into the HDF5 file.

        Parameters
        ----------
        type        The image type
        scale       Scale is the percentage of the original image size.
        chunk_size  The same as with the other images. Can be used to define a
                    particular chunk size for data storage.
        """
        if scale in self.get_scales(type):
            if uiutils.get_user_input(
                            "The scale %i already exists for the image type "
                            "%s. Do you want to recalculate?" % (scale, type)
            ):
                pass
            else:
                return
        # Iterate over all the images
        for index in range(self.get_number_of_images(type)):
            group_name = type + "/" + str(index)
            image_group = self.data[group_name]
            # Iterate over channels
            for channel in range(self.channel_count):
                name_new = "channel_" + str(channel) + "_scale_" + str(scale)
                name_ref = "channel_" + str(channel) + "_scale_100"
                # Check if exists and delete if yes.
                if name_new in image_group:
                    del image_group[name_new]
                    continue

                # Zoom
                spacing = tuple(100*x/scale for x in image_group[name_ref].attrs["spacing"])
                z_factor = float(scale)/100
                zoom = (z_factor, ) * self.get_number_of_dimensions()
                data = ndimage.zoom(image_group[name_ref], zoom, order=3)

                if chunk_size is None:
                    image_group.create_dataset(name_new, data=data)
                else:
                    image_group.create_dataset(name_new, data=data, chunks=chunk_size)

                image_group[name_new].attrs["angle"] = image_group[name_ref].attrs["angle"]
                image_group[name_new].attrs["spacing"] = spacing
                image_group[name_new].attrs["size"] = data.shape

    def calculate_missing_psfs(self):
        """
        In case separate PSFs were not recorded for every view, the missing PSFs can be
        calculated here before image fusion. This requires that the spatial transform
        is available for every registered view.
        """
        max_scale = max(self.get_scales("registered"))
        if max_scale < 100:
            if uiutils.get_user_input("There is no registration result "
                                      "available for the original images. The "
                                      "largest available scale is %i. Do you "
                                      "want to proceed with that? " %
                                      max_scale):
                pass
            else:
                raise ValueError("No suitable registration result available.")

        for channel in range(self.channel_count):
            self.set_active_image(0, channel, 100, "psf")
            image_spacing = self.get_voxel_size()
            psf_orig = self.data[self.active_image][:]

            for index in range(1, self.get_number_of_images("registered")):
                if not self.check_if_exists("psf", index, channel, 100):
                    self.set_active_image(index, channel, max_scale, "registered")
                    transform = self.get_transform()
                    psf_new = itkutils.rotate_psf(psf_orig,
                                                  transform,
                                                  image_spacing,
                                                  return_numpy=True)
                    self.add_psf(psf_new, 100, index, channel,
                                 self.get_rotation_angle(), image_spacing,
                                 calculated=True)

    def copy_registration_result(self, from_scale, to_scale):
        """
        With this function it is possible
        to migrate the registration results from one scale to another.

        With very large images it is sometimes easier and faster to perform
        image registration with downsampled versions of the original images.
        The accuracy of the registration result is often very good, even with
        60 percent downsampled images.

        Parameters
        ----------
        :item from_scale    The scale for which there is an existing
                            registration result.
        :item to_scale      The scale for which the new registration results
                            should be calculated.

        Returns
        -------

        """

        # Check that the registration result for the specified scale
        # exists.
        assert from_scale in self.get_scales("registered")
        print("Copying registration results from %i to %i percent scale" % (
              from_scale, to_scale))
        if to_scale not in self.get_scales("original"):
            self.create_rescaled_images("original", to_scale)

        for channel in range(self.channel_count):
            print("Resampling view 0")
            self.set_active_image(0, channel, to_scale, "original")
            self.add_registered_image(self.data[self.active_image][:], to_scale,
                                      0, channel, 0, self.get_voxel_size())
            self.set_active_image(0, channel, to_scale, "registered")
            reference = self.get_itk_image()

            for view in range(1, self.get_number_of_images("original")):
                print("Resampling view %i" % view)
                self.set_active_image(view, channel, from_scale, "registered")
                transform = self.get_transform()
                transform_params = self.get_transform_parameters()
                self.set_active_image(view, channel, to_scale, "original")
                image = self.get_itk_image()
                angle = self.get_rotation_angle(radians=False)
                spacing = self.get_voxel_size()
                result = itkutils.convert_from_itk_image(
                    itkutils.resample_image(image, transform, reference=reference)
                )[0]
                self.add_registered_image(result, to_scale, view, channel, angle,
                                          spacing)
                self.add_transform(to_scale, view, channel, transform_params[0], transform_params[1], transform_params[2])

                #  def add_transform(self, scale, index, channel, params, fixed_params, transform_type):

    def get_rotation_angle(self, radians=True):
        """
        Get rotation angle of the currently active image.

        Parameters
        ----------
        radians     Use radians instead of degrees

        Returns
        -------
        Returns the rotation angle, as degrees or radians
        """
        if radians:
            angle = numpy.pi * int(self.data[self.active_image].attrs["angle"]) / 180
            return angle
        else:
            return int(self.data[self.active_image].attrs["angle"])

    def get_voxel_size(self):
        """
        Get the voxel size of the currently active image.

        Returns
        -------
        Voxel size as a three element tuple (assuming 3D image).
        """
        return list(self.data[self.active_image].attrs["spacing"])

    def get_max(self):
        return self.data[self.active_image][:].max()

    def get_image_size(self):
        """
        Get dimensions of the currently active image

        Returns
        -------
        Image dimensions as a tuple.
        """
        return self.data[self.active_image].attrs["size"]

    def get_dtype(self):
        """
        Get the datatype of the currently acitve image

        Returns
        -------
        The datatype as a numpy.dtype compatible parameter
        """
        return self.data[self.active_image].dtype

    def get_number_of_images(self, image_type):
        """
        Get the number of images of a given type stored in the data structure

        Parameters
        ----------
        :param image_type     The image type

        Returns
        -------
        The number of images of a given type.

        """
        assert image_type in image_types_c
        return len(self.data[image_type])

    def get_number_of_dimensions(self):
        return self.data[self.active_image].ndim

    def get_scales(self, image_type):
        """
        Get a list of the image sizes available for a given image type.

        Parameters
        ----------
        :param image_type       The image type

        Returns
        -------
        Returns a list of the saved scales. Raises an error if the scales
        have been saved inconsistently, i.e. all images of the same type
        do not have the same scales available.
        """

        assert image_type in image_types_c
        scales = []

        def find_scale(name):
            scales.append(int(name.split("_")[-1]))

        for index in range(self.get_number_of_images(image_type)):
            scales = []
            group_name = image_type + "/" + str(index)
            image_group = self.data[group_name]
            image_group.visit(find_scale)

            if index == 0:
                scales_ref = scales
            else:
                if set(scales_ref) != set(scales):
                    raise ValueError("Database error. Resampled images have not been"
                                     "saved consistently for image type %s" %
                                     image_type)

        return scales

    def get_transform(self):
        """
        Get the spatial transformation of the current registered view. Requires a
        registered view to be set as active.

        Returns
        -------
        Return an ITK spatial transform.
        """
        assert "registered" in self.active_image
        tfm_type = self.data[self.active_image].attrs["tfm_type"]
        tfm_params = self.data[self.active_image].attrs["tfm_params"]
        tfm_fixed_params = self.data[self.active_image].attrs["tfm_fixed_params"]
        ndim = self.get_number_of_dimensions()

        return itkutils.make_itk_transform(tfm_type,
                                           ndim,
                                           tfm_params,
                                           tfm_fixed_params)

    def get_transform_parameters(self):
        assert "registered" in self.active_image
        tfm_type = self.data[self.active_image].attrs["tfm_type"]
        tfm_params = self.data[self.active_image].attrs["tfm_params"]
        tfm_fixed_params = self.data[self.active_image].attrs["tfm_fixed_params"]

        return tfm_params, tfm_fixed_params, tfm_type

    def set_active_image(self, index, channel, scale, image_type):
        """
        Select which view is currently active.

        :param index:       View index, goes from 0 to number of views - 1
        :param channel      The currently active color channel. Goes from 0 to
                            number of channels - 1
        :param scale        Image size, as a percentage of the full size.
        :param image_type   Image type as a string, listed in image_types_c
        """
        if int(index) >= self.series_count:
            print("Invalid index. There are only %i images in the file" % self.series_count)
            return
        elif image_type not in image_types_c:
            print("Unkown image type.")
            return
        else:
            if image_type == "fused":
                self.active_image = image_type + "/channel_" + str(channel) + "_scale_" + str(scale)
            else:
                self.active_image = image_type + "/" + str(index) + "/channel_" + str(channel) + "_scale_" + str(scale)
            if self.active_image not in self.data:
                raise ValueError("No such image: %s" % self.active_image)

    # def set_fused_block(self, block, start_index):
    #     assert isinstance(block, numpy.ndarray) and isinstance(start_index, numpy.ndarray)
    #     stop_index = start_index + block.shape
    #     self.data["fused"][start_index:stop_index] = block

    def get_registered_block(self, block_size, block_pad, block_start_index):
        """
        When fusing large images, it is often necessary to divide the images
        into several blocks in order to keep the memory requirements at bay.
        For such use cases functionality was added here to read a partial image
        of a given block size and start index directly from disk. Padding is
        supported as well.

        Parameters
        ----------
        :param block_size   The size of the desired block
        :param block_pad    The amount of padding to be applied to the sides of the
                            block. This kind of partial overlap of adjacent blocks is
                            needed to avoid fusion artifacts at the block boundaries.

        :param block_start_index
                            The index pointing to the beginning of the block.

        Returns
        -------
        The padded image block as a 3D numpy array.

        """
        assert isinstance(block_size, numpy.ndarray)

        assert "registered" in self.active_image, "You must specify a registered image"

        image_size = self.data[self.active_image].shape

        # Apply padding
        end_index = block_start_index + block_size + block_pad
        start_index = block_start_index - block_pad
        # print "Getting a block from ", self.active_image
        # print "The start index is %i %i %i" % tuple(start_index)
        # print "The block size is %i %i %i" % tuple(block_size)

        block_idx = arrayutils.start_to_stop_idx(start_index, end_index)

        if (image_size >= end_index).all() and (start_index >= 0).all():
            block = self.data[self.active_image][block_idx]
            return block

        else:
            pad_block_size = block_size + 2 * block_pad
            block = numpy.zeros(pad_block_size)

            # If the start_index is close to the image boundaries, it is very
            # probable that padding will introduce negative start_index values.
            # In such case the first pixel index must be corrected.
            if (start_index < 0).any():
                block_start = numpy.negative(start_index.clip(max=0))
                image_start = start_index + block_start
            else:
                block_start = (0, 0, 0)
                image_start = start_index
            # If the padded block is larger than the image size the
            # block_size must be adjusted.
            if not (image_size >= end_index).all():
                block_crop = end_index - image_size
                block_crop[block_crop < 0] = 0
                block_end = pad_block_size - block_crop
            else:
                block_end = pad_block_size

            end_index = start_index + block_end

            block_read_idx = arrayutils.start_to_stop_idx(image_start, end_index)
            block_write_idx = arrayutils.start_to_stop_idx(block_start, block_end)

            block[block_write_idx] = self.data[self.active_image][block_read_idx]
            return block

    def get_itk_image(self):
        """
        Get the currently active image as a ITK image instead of a Numpy array.
        """
        return itkutils.convert_from_numpy(self.data[self.active_image][:],

                                           self.data[self.active_image].attrs["spacing"])

    def get_image(self):
        """
        Get the currently active image as an Image object instead of a Numpy array
        """
        return Image(self.data[self.active_image][:],
                     self.data[self.active_image].attrs["spacing"])

    def get_active_image_index(self):
        """
        Get the image of the currently active image
        """
        return self.active_image.split('/')[1]

    def close(self):
        """
        Close the file object.
        """
        self.data.attrs["series_count"] = self.series_count
        self.data.attrs["channel_count"] = self.channel_count

        self.data.close()

    def check_if_exists(self, image_type, index, channel, scale):
        """
        Check if the specified image already exists in the data structure.

        Parameters
        ----------
        :param image_type         The parameters needed to identify an image in the
        :param index        data structure.
        :param channel
        :param scale

        Returns
        -------
        True if Yes, False if No.
        """
        name = image_type + "/" + str(index) + "/channel_" + str(channel) + "_scale_" + str(scale)
        return name in self.data

    def __getitem__(self, item):
        return self.data[self.active_image][item]

    def __setitem__(self, key, value):
        self.data[self.active_image][key] = value