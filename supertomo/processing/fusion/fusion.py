"""
fusion.py

Copyright (C) 2014, 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the SuperTomo2 multi-view image fusion algorithms.
They have been inteded for use with computers that do not support
hardware GPU acceleration. THe accelerated versions of the same functions
can be found in fusion_cuda.py. The fftconvolve function that is
used in this file, can take advantage of MKL optimizations available
in the Anaconda Accelerate package.

"""
import datetime
import itertools
import os
import shutil
import sys
import tempfile
import time

import numpy
import supertomo.processing.ops_ext as ops_ext
from scipy.ndimage.interpolation import zoom
from scipy.signal import fftconvolve, medfilt

import supertomo.processing.ndarray as ops_array
import supertomo.processing.to_string as ops_output
from supertomo.data.containers import temp_data, image_data, image
from supertomo.data.io import tiffile


class MultiViewFusionRL(object):
    """
    The Richardson-Lucy fusion is a result of simultaneous deblurring of
    several 3D volumes.
    """

    def __init__(self, data, options):
        """
        :param data:    a ImageData object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        assert isinstance(data, image_data.ImageData)

        self.data = data
        self.options = options

        # Select views to fuse
        if self.options.fuse_views == -1:
            self.views = xrange(self.data.get_number_of_images("registered"))
        else:
            self.views = self.options.fuse_views

        self.n_views = len(self.views)

        # Get image size
        self.data.set_active_image(0, self.options.channel, self.options.scale,
                                   "registered")
        self.image_size = numpy.array(self.data.get_image_size())

        print "The original image size is %i %i %i" % tuple(self.image_size)

        self.voxel_size = self.data.get_voxel_size()
        self.iteration_count = 0

        # Setup blocks
        self.num_blocks = options.num_blocks
        self.block_size, self.image_size = self.__calculate_block_and_image_size()
        self.memmap_directory = tempfile.mkdtemp()

        # Memmap the estimates to reduce memory requirements. This will slow
        # down the fusion process considerably..
        if self.options.memmap_estimates:
            estimate_new_f = os.path.join(self.memmap_directory, "estimate_new.dat")
            self.estimate_new = numpy.memmap(estimate_new_f, dtype='float32',
                                             mode='w+',
                                             shape=tuple(self.image_size))

            estimate_f = os.path.join(self.memmap_directory, "estimate.dat")
            self.estimate = numpy.memmap(estimate_f, dtype=numpy.float32,
                                         mode='w+',
                                         shape=tuple(self.image_size))
        else:
            self.estimate = numpy.zeros(tuple(self.image_size),
                                        dtype=numpy.float32)
            self.estimate_new = numpy.zeros(tuple(self.image_size),
                                            dtype=numpy.float32)

        if not self.options.disable_tau1:
            prev_estimate_f = os.path.join(self.memmap_directory, "prev_estimate.dat")
            self.prev_estimate = numpy.memmap(prev_estimate_f, dtype=numpy.float32,
                                              mode='w+',
                                              shape=tuple(self.image_size))
        # Setup PSFs
        self.psfs = []
        self.adj_psfs = []
        self.__get_psfs()
        if "opt" in self.options.fusion_method:
            self.virtual_psfs = []
            self.__compute_virtual_psfs()
        else:
            pass

        print "The fusion will be run with %i blocks" % self.num_blocks
        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)
        print "The internal block size is %i %i %i" % padded_block_size
        # Create temporary directory and data file.
        self.data_to_save = ('count', 't', 'mn', 'mx', 'tau1', 'tau2', 'leak', 'e',
                             's', 'u', 'n', 'u_esu')

        if self.options.temp_dir is None:
            self.temp_data = temp_data.TempData()
        else:
            self.temp_data = temp_data.TempData(directory=self.options.temp_dir)

        date_now = datetime.datetime.now().strftime("%H_%M_%S_")
        tempfile_name = '{}_fusion_data.csv'.format(date_now)
        self.temp_data.create_data_file(tempfile_name, self.data_to_save)
        self.temp_data.write_comment('Fusion Command: %s' % (' '.join(map(str, sys.argv))))

    def compute_estimate(self):
        """
        Calculates a single RL fusion estimate. There is no reason to call this
        function -- it is used internally by the class during fusion process.
        """

        print 'Beginning the computation of the %i. estimate' % self.iteration_count

        if "multiplicative" in self.options.fusion_method:
            self.estimate_new[:] = numpy.float32(1.0)
        else:
            self.estimate_new[:] = numpy.float32(0)

        # Iterate over views
        for idx, view in enumerate(self.views):

            # Get PSFs for view
            psf = self.psfs[idx]
            adj_psf = self.adj_psfs[idx]

            self.data.set_active_image(view, self.options.channel,
                                       self.options.scale, "registered")

            weighting = float(self.data.get_max()) / 255

            # Iterate over blocks
            for x, y, z in itertools.product(xrange(0, self.image_size[0], self.block_size[0]),
                                             xrange(0, self.image_size[1], self.block_size[1]),
                                             xrange(0, self.image_size[2], self.block_size[2])):

                index = numpy.array((x, y, z), dtype=int)
                if self.options.block_pad > 0:
                    estimate_block = self.get_padded_block(index.copy())
                else:
                    estimate_block = self.estimate[index[0]:index[0]+self.block_size[0],
                                                   index[1]:index[1]+self.block_size[1],
                                                   index[2]:index[2]+self.block_size[2]]

                # Execute: cache = convolve(PSF, estimate), non-normalized
                estimate_block_new = fftconvolve(estimate_block, psf, mode='same')
                estimate_block_new *= weighting

                # Execute: cache = data/cache
                block = self.data.get_registered_block(self.block_size,
                                                       self.options.block_pad,
                                                       index.copy())

                with numpy.errstate(divide="ignore"):
                    estimate_block_new = block / estimate_block_new
                    estimate_block_new[estimate_block_new == numpy.inf] = 0.0
                    estimate_block_new = numpy.nan_to_num(estimate_block_new)

                # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
                # Convolution with virtual PSFs is performed here as well, if
                # necessary
                estimate_block_new = fftconvolve(estimate_block_new, adj_psf, mode='same')

                # Update the contribution from a single view to the new estimate
                if self.options.block_pad == 0:
                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] *= estimate_block_new
                    else:
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] += estimate_block_new
                else:
                    pad = self.options.block_pad

                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] *= estimate_block_new[pad:pad + self.block_size[0],
                                   pad:pad + self.block_size[1],
                                   pad:pad + self.block_size[2]]

                    else:
                        # print "The block size is ", self.block_size
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] += estimate_block_new[pad:pad + self.block_size[0],
                                   pad:pad + self.block_size[1],
                                   pad:pad + self.block_size[2]]

        # Divide with the number of projections
        if "summative" in self.options.fusion_method:
            self.estimate_new *= (1.0 / self.n_views)
        else:
            self.estimate_new[:] = ops_array.nroot(self.estimate_new,
                                                   self.n_views)

        return ops_ext.update_estimate_poisson(self.estimate,
                                               self.estimate_new,
                                               self.options.convergence_epsilon)

    def __compute_virtual_psfs(self):
        """
        Implements a Virtual PSF calculation routine, as described in "Efficient
        Bayesian-based multiview deconvolution" by Preibich et al in Nature
        Methods 11/6 (2014)
        """

        print "Caclulating Virtual PSFs"

        for i in xrange(self.n_views):
            virtual_psf = numpy.ones(self.psfs[0].shape, dtype=self.psfs[0].dtype)
            for j in xrange(self.n_views):
                if j == i:
                    pass
                else:
                    cache = fftconvolve(
                        fftconvolve(
                            self.adj_psfs[i],
                            self.psfs[j],
                            mode='same'
                        ),
                        self.adj_psfs[j],
                        mode='same'

                    )

                    virtual_psf *= cache.real

            virtual_psf *= self.adj_psfs[i]
            virtual_psf /= virtual_psf.sum()
            self.adj_psfs[i] = virtual_psf
            #self.virtual_psfs.append(virtual_psf)

    def execute(self):
        """
        This is the main fusion function
        """

        print "Preparing image fusion."

        save_intermediate_results = self.options.save_intermediate_results

        first_estimate = self.options.first_estimate

        self.data.set_active_image(0,
                                   self.options.channel,
                                   self.options.scale,
                                   "registered")
        orig_size = self.data.get_image_size()

        if first_estimate == 'first_image':
            self.estimate[:] = self.data[:].astype(numpy.float32)
        elif first_estimate == 'first_image_mean':
            self.estimate[:] = numpy.float32(numpy.mean(self.data[:]))
        elif first_estimate == 'sum_of_all':
            for i in range(self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate[:orig_size[0], :orig_size[1], :orig_size[2]] += self.data[:].astype(numpy.float32)
            self.estimate *= (1.0/self.n_views)
        elif first_estimate == 'simple_fusion':
            self.estimate[:orig_size[0], :orig_size[1], :orig_size[2]] = self.data[:]
            for i in xrange(1, self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate[:orig_size[0], :orig_size[1], :orig_size[2]] = (
                    self.estimate[:orig_size[0], :orig_size[1], :orig_size[2]] - (
                        self.estimate[:orig_size[0], :orig_size[1], :orig_size[2]] - self.data[:]
                    ).clip(min=0)
                ).clip(min=0).astype(numpy.float32)
        elif first_estimate == 'average_af_all':
            self.estimate[:] = numpy.float32(0)
            for i in range(self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate[:] += (self.data[:].astype(numpy.float32) /
                                     self.n_views)
        elif first_estimate == 'constant':
            self.estimate[:] = numpy.float32(self.options.estimate_constant)
        else:
            raise NotImplementedError(repr(first_estimate))

        stop_message = ''
        self.iteration_count = 0
        max_count = self.options.max_nof_iterations
        initial_photon_count = self.data[:].sum()

        bar = ops_output.ProgressBar(0,
                                   max_count,
                                   totalWidth=40,
                                   show_percentage=False)

        # The Fusion calculation starts here
        # ====================================================================
        try:
            while True:

                info_map = {}
                ittime = time.time()

                if not self.options.disable_tau1:
                    self.prev_estimate[:] = self.estimate.copy()

                e, s, u, n = self.compute_estimate()

                self.iteration_count += 1
                photon_leak = 1.0 - (e + s + u) / initial_photon_count
                leak = 100 * photon_leak
                u_esu = u / (e + s + u)

                if not self.options.disable_tau1:
                    tau1 = abs(self.estimate - self.prev_estimate).sum() / abs(
                        self.prev_estimate).sum()
                    info_map['TAU1=%s'] = tau1


                # Update UI
                info_map['E/S/U/N=%s/%s/%s/%s'] = int(e), int(s), int(u), int(n)
                info_map['LEAK=%s%%'] = 100 * photon_leak
                info_map['U/ESU=%s'] = u_esu
                info_map['TIME=%ss'] = t = time.time() - ittime
                bar.updateComment(' ' + ', '.join([k % (ops_output.tostr(info_map[k])) for k in sorted(info_map)]))
                bar(self.iteration_count)
                print

                # Save parameters to file
                self.temp_data.write_row(', '.join(self.data_to_save))

                # Save intermediate image
                if save_intermediate_results:
                    self.temp_data.save_image(
                        self.estimate,
                        'result_%s.tif' % self.iteration_count
                    )

                # Check if it's time to stop:
                if int(u) == 0 and int(n) == 0:
                    stop_message = 'The number of non converging photons reached to zero.'
                    break
                elif self.iteration_count >= max_count:
                    stop_message = 'The number of iterations reached to maximal count: %s' % max_count
                    break
                elif not self.options.disable_tau1 and tau1 <= self.options.rltv_stop_tau:
                    stop_message = 'Desired tau-threshold achieved'
                    break
                else:
                    continue

        except KeyboardInterrupt:
            stop_message = 'Iteration was interrupted by user.'

        # if self.num_blocks > 1:
        #     self.estimate = self.estimate[0:real_size[0], 0:real_size[1], 0:real_size[2]]

        print
        bar.updateComment(' ' + stop_message)
        bar(self.iteration_count)
        print

    def __get_psfs(self):
        """
        Reads the PSFs from the HDF5 data structure and zooms to the same pixel
        size with the registered images, of selected scale and channel.
        """
        self.data.set_active_image(0, self.options.channel,
                                   self.options.scale, "registered")
        image_spacing = self.data.get_voxel_size()

        for i in self.views:
            self.data.set_active_image(i, 0, 100, "psf")
            psf_orig = self.data[:]
            psf_spacing = self.data.get_voxel_size()

            # Zoom to the same voxel size
            zoom_factors = tuple(x/y for x, y in zip(psf_spacing, image_spacing))
            psf_new = zoom(psf_orig, zoom_factors).astype(numpy.float32)

            psf_new /= psf_new.sum()

            # Save the zoomed and rotated PSF, as well as its mirrored version
            self.psfs.append(psf_new)
            self.adj_psfs.append(psf_new[::-1, ::-1, ::-1])

    def save_to_hdf(self):
        """
        Save result to the SuperTomo2 data structure.

        """
        self.data.set_active_image(0,
                                   self.options.channel,
                                   self.options.scale,
                                   "registered")
        spacing = self.data.get_voxel_size()

        self.data.add_fused_image(self.estimate,
                                  self.options.channel,
                                  self.options.scale,
                                  spacing)

    def save_to_tiff(self, filename):
        """
        Save fusion result to TIFF

        Parameters
        ----------
        :param filename     full path to the image

        """

        tiffile.imsave(filename, self.estimate)

    def show_result(self):
        """
        Show fusion result. This is a temporary solution for now
        calling Fiji through ITK. An internal viewer would be
        preferable.
        """
        import SimpleITK as sitk
        sitk.Show(sitk.GetImageFromArray(self.estimate))
        #show.evaluate_3d_image(self.get_8bit_result())

    def __calculate_block_and_image_size(self):
        """
        Calculate the block size and the internal image size for a given
        number of blocks. 1,2,4 or 8 blocks are currently supported.

        """
        block_size = self.image_size
        image_size = self.image_size

        if self.num_blocks == 1:
            return block_size, image_size
        elif self.num_blocks == 2:
            multiplier = numpy.array([2, 1, 1])
        elif self.num_blocks == 4:
            multiplier = numpy.array([4, 1, 1])
        elif self.num_blocks == 8:
            multiplier = numpy.array([4, 2, 1])
        elif self.num_blocks == 12:
            multiplier = numpy.array([4, 2, 2])
        elif self.num_blocks == 24:
            multiplier = numpy.array([4, 3, 2])
        elif self.num_blocks == 48:
            multiplier = numpy.array([4, 4, 3])
        elif self.num_blocks == 64:
            multiplier = numpy.array([4, 4, 4])
        elif self.num_blocks == 96:
            multiplier = numpy.array([6, 4, 4])
        elif self.num_blocks == 144:
            multiplier = numpy.array([4, 6, 6])
        else:
            raise NotImplementedError

        block_size = numpy.ceil(self.image_size.astype(numpy.float16) / multiplier).astype(numpy.int64)
        image_size += (multiplier * block_size - image_size)

        return block_size, image_size

    def get_padded_block(self, block_start_index):
        """
        Get a padded block from the self.estimate

        Parameters
        ----------
        :param block_start_index  The real block start index, not considering the padding

        Returns
        -------
        Returns the padded estimate block as a numpy array.

        """

        block_pad = self.options.block_pad
        image_size = self.image_size

        # Apply padding
        end_index = block_start_index + self.block_size + block_pad
        start_index = block_start_index - block_pad

        # If the padded block fits within the image boundaries, nothing special
        # is needed to extract it. Normal numpy slicing notation is used.
        if (image_size >= end_index).all() and (start_index >= 0).all():
            block = self.estimate[
                    start_index[0]:end_index[0],
                    start_index[1]:end_index[1],
                    start_index[2]:end_index[2]
                    ]
            return block

        else:
            block_size = self.block_size + 2 * block_pad
            # Block outside the image boundaries will be filled with zeros.
            block = numpy.zeros(block_size)
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
                block_end = block_size - block_crop
            else:
                block_end = block_size

            end_index = start_index + block_end

            block[block_start[0]:block_end[0],
                  block_start[1]:block_end[1],
                  block_start[2]:block_end[2]] = self.estimate[image_start[0]:end_index[0],
                                                               image_start[1]:end_index[1],
                                                               image_start[2]:end_index[2]]
            return block

    def get_8bit_result(self, denoise=True):
        """
        Returns the current estimate (the fusion result) as an 8-bit uint, rescaled
        to the full 0-255 range.
        """
        if denoise:
            result = medfilt(self.estimate)
        else:
            result = self.estimate

        result *= (255.0 / result.max())
        result[result < 0] = 0
        return image.Image(result.astype(numpy.uint8), self.voxel_size)

    def close(self):
        if self.options.memmap_estimates:
            del self.estimate
            del self.estimate_new
        if not self.options.disable_tau1:
            del self.prev_estimate

        shutil.rmtree(self.memmap_directory)
