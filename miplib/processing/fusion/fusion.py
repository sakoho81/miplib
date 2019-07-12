"""
fusion.py

Copyright (C) 2014, 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the miplib multi-view image fusion algorithms.
They have been inteded for use with computers that do not support
hardware GPU acceleration. THe accelerated versions of the same functions
can be found in fusion_cuda.py. The fftconvolve function that is
used in this file, can take advantage of MKL optimizations available
in the Anaconda Accelerate package.

"""
import itertools
import os
import shutil
import tempfile
import time

import numpy
import pandas

import miplib.processing.ops_ext as ops_ext
from scipy.ndimage.interpolation import zoom
from scipy.signal import fftconvolve, medfilt

import miplib.processing.ndarray as ops_array
import miplib.processing.to_string as ops_output
from miplib.data.containers import image_data, image
from miplib.data.containers.image import Image
from . import utils as fusion_utils


class MultiViewFusionRL(object):
    """
    The Richardson-Lucy fusion is a result of simultaneous deblurring of
    several 3D volumes.
    """

    def __init__(self, data, writer, options):
        """
        :param data:    a ImageData object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        assert isinstance(data, image_data.ImageData)

        self.data = data
        self.options = options
        self.writer = writer

        # Select views to fuse
        if self.options.fuse_views == -1:
            self.views = range(self.data.get_number_of_images("registered"))
        else:
            self.views = self.options.fuse_views

        self.n_views = len(self.views)

        # Get weights
        self.weights = numpy.zeros(self.n_views, dtype=numpy.float32)

        for idx, view in enumerate(self.views):
            self.data.set_active_image(view, self.options.channel,
                                       self.options.scale, "registered")

            self.weights[idx] = self.data.get_max()

        self.weights /= self.weights.sum()


        # Get image size
        self.data.set_active_image(0, self.options.channel, self.options.scale,
                                   "registered")
        self.image_size = self.data.get_image_size()
        self.imdims = len(self.image_size)

        print("The original image size is {}".format(tuple(self.image_size)))

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
            self.estimate_new = Image(numpy.memmap(estimate_new_f, dtype='float32',
                                                   mode='w+',
                                                   shape=tuple(self.image_size)), self.voxel_size)

            estimate_f = os.path.join(self.memmap_directory, "estimate.dat")
            self.estimate = Image(numpy.memmap(estimate_f, dtype=numpy.float32,
                                               mode='w+',
                                               shape=tuple(self.image_size)), self.voxel_size)
        else:
            self.estimate = Image(numpy.zeros(tuple(self.image_size),
                                              dtype=numpy.float32), self.voxel_size)
            self.estimate_new = Image(numpy.zeros(tuple(self.image_size),
                                                  dtype=numpy.float32), self.voxel_size)

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

        print("The fusion will be run with %i blocks" % self.num_blocks)
        padded_block_size = tuple(i + 2 * self.options.block_pad for i in self.block_size)
        print("The internal block size is %s" % (padded_block_size,))

        self.column_headers = ('t', 'tau1', 'leak', 'e',
                               's', 'u', 'n', 'uesu')
        self._progress_parameters = numpy.empty((self.options.max_nof_iterations, len(self.column_headers)),
                                                dtype=numpy.float32)

    @property
    def progress_parameters(self):
        return pandas.DataFrame(data=self._progress_parameters, columns=self.column_headers)

    def compute_estimate(self):
        """
        Calculates a single RL fusion estimate. There is no reason to call this
        function -- it is used internally by the class during fusion process.
        """

        print('Beginning the computation of the %i. estimate' % self.iteration_count)

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

            #weighting = float(self.data.get_max()) / 255
            weighting = self.weights[idx]

            iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
            pad = self.options.block_pad
            cache_idx = tuple(slice(pad, pad + block) for block in self.block_size)

            # Iterate over blocks
            for pos in itertools.product(*iterables):

                estimate_idx = tuple(slice(j, j + k) for j, k in zip(pos, self.block_size))
                index = numpy.array(pos, dtype=int)
                if self.options.block_pad > 0:
                    estimate_block = self.get_padded_block(index.copy())
                else:
                    estimate_block = self.estimate[estimate_idx]

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
                        self.estimate_new[estimate_idx] *= estimate_block_new
                    else:
                        self.estimate_new[estimate_idx] += estimate_block_new
                else:
                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[estimate_idx] *= estimate_block_new[cache_idx]

                    else:
                        # print "The block size is ", self.block_size
                        self.estimate_new[estimate_idx] += estimate_block_new[cache_idx]

        # I changed the weighting scheme a little bit
        # I'm not sure if this thing is necessary; maybe in the multiplicative?
        # Divide with the number of projections
        # if "summative" in self.options.fusion_method:
        #     self.estimate_new *= (1.0 / self.n_views)
        # else:
        #     self.estimate_new[:] = ops_array.nroot(self.estimate_new,
        #                                            self.n_views)

        return ops_ext.update_estimate_poisson(self.estimate,
                                               self.estimate_new,
                                               self.options.convergence_epsilon)

    def execute(self):
        """
        This is the main fusion function
        """

        print("Preparing image fusion.")

        save_intermediate_results = self.options.save_intermediate_results

        first_estimate = self.options.first_estimate

        self.data.set_active_image(0,
                                   self.options.channel,
                                   self.options.scale,
                                   "registered")

        if first_estimate == 'first_image':
            self.estimate[:] = self.data[:].astype(numpy.float32)
        elif first_estimate == 'first_image_mean':
            self.estimate[:] = numpy.float32(numpy.mean(self.data[:]))
        elif first_estimate == 'sum_of_originals':
            self.estimate[:] = fusion_utils.sum_of_all(self.data,
                                                       self.options.channel,
                                                       self.options.scale)
        elif first_estimate == 'sum_of_registered':
            self.estimate[:] = fusion_utils.sum_of_all(self.data,
                                                       self.options.channel,
                                                       self.options.scale,
                                                       "registered")
        elif first_estimate == 'simple_fusion':
            self.estimate[:] = fusion_utils.simple_fusion(self.data,
                                                          self.options.channel,
                                                          self.options.scale)
        elif first_estimate == 'average_af_all':
            self.estimate[:] = fusion_utils.average_of_all(self.data,
                                                           self.options.channel,
                                                           self.options.scale,
                                                           "registered")
        elif first_estimate == 'constant':
            self.estimate[:] = numpy.float32(self.options.estimate_constant)
        else:
            raise NotImplementedError(repr(first_estimate))

        self.iteration_count = 0
        max_count = self.options.max_nof_iterations
        initial_photon_count = self.data[:].sum()

        bar = ops_output.ProgressBar(0,
                                     max_count,
                                     totalWidth=40,
                                     show_percentage=False)

        self._progress_parameters = numpy.zeros((self.options.max_nof_iterations,
                                                 len(self.column_headers)),
                                                dtype=numpy.float32)

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
                u_esu = u / (e + s + u)

                if not self.options.disable_tau1:
                    tau1 = abs(self.estimate - self.prev_estimate).sum() / abs(
                        self.prev_estimate).sum()
                    info_map['TAU1=%s'] = tau1

                t = time.time() - ittime
                leak = 100 * photon_leak

                # Update UI
                info_map['E/S/U/N=%s/%s/%s/%s'] = int(e), int(s), int(u), int(n)
                info_map['LEAK=%s%%'] = leak
                info_map['U/ESU=%s'] = u_esu
                info_map['TIME=%ss'] = t
                bar.updateComment(' ' + ', '.join([k % (ops_output.tostr(info_map[k])) for k in sorted(info_map)]))
                bar(self.iteration_count)
                print()

                # Save parameters to file
                self._progress_parameters[self.iteration_count - 1] = (t, tau1, leak, e, s, u, n, u_esu)

                # Save intermediate image
                if save_intermediate_results:
                    self.writer.write(Image(self.estimate, self.voxel_size))

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

        print()
        bar.updateComment(' ' + stop_message)
        bar(self.iteration_count)
        print()

    # region Prepare PSFs
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
            zoom_factors = tuple(x / y for x, y in zip(psf_spacing, image_spacing))
            psf_new = zoom(psf_orig, zoom_factors).astype(numpy.float32)

            psf_new /= psf_new.sum()

            # Save the zoomed and rotated PSF, as well as its mirrored version
            self.psfs.append(psf_new)

            if self.imdims == 3:
                self.adj_psfs.append(psf_new[::-1, ::-1, ::-1])
            else:
                self.adj_psfs.append(psf_new[::-1, ::-1])

    def __compute_virtual_psfs(self):
        """
        Implements a Virtual PSF calculation routine, as described in "Efficient
        Bayesian-based multiview deconvolution" by Preibich et al in Nature
        Methods 11/6 (2014)
        """

        print("Caclulating Virtual PSFs")

        for i in range(self.n_views):
            virtual_psf = numpy.ones(self.psfs[0].shape, dtype=self.psfs[0].dtype)
            for j in range(self.n_views):
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
            # self.virtual_psfs.append(virtual_psf)

    # endregion


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
            multiplier3 = numpy.array([2, 1, 1])
            multiplier2 = numpy.array([2, 1])
        elif self.num_blocks == 4:
            multiplier3 = numpy.array([4, 1, 1])
            multiplier2 = numpy.array([2, 2])
        elif self.num_blocks == 8:
            multiplier3 = numpy.array([4, 2, 1])
            multiplier2 = numpy.array([4, 2])
        elif self.num_blocks == 12:
            multiplier3 = numpy.array([4, 2, 2])
            multiplier2 = numpy.array([4, 3])
        elif self.num_blocks == 24:
            multiplier3 = numpy.array([4, 3, 2])
            multiplier2 = numpy.array([6, 4])
        elif self.num_blocks == 48:
            multiplier3 = numpy.array([4, 4, 3])
            multiplier2 = numpy.array([8, 6])
        elif self.num_blocks == 64:
            multiplier3 = numpy.array([4, 4, 4])
            multiplier2 = numpy.array([8, 8])
        elif self.num_blocks == 96:
            multiplier3 = numpy.array([6, 4, 4])
            multiplier2 = numpy.array([12, 8])
        elif self.num_blocks == 144:
            multiplier3 = numpy.array([4, 6, 6])
            multiplier2 = numpy.array([12, 12])
        else:
            raise NotImplementedError

        if self.imdims == 2:
            block_size = numpy.ceil(self.image_size.astype(numpy.float16) / multiplier2).astype(numpy.int64)
            image_size += (multiplier2 * block_size - image_size)
        else:
            block_size = numpy.ceil(self.image_size.astype(numpy.float16) / multiplier3).astype(numpy.int64)
            image_size += (multiplier3 * block_size - image_size)

        return block_size, image_size

    def get_padded_block(self, image, block_start_index):
        """
        Get a padded block from the self.estimate

        Parameters
        ----------
        :param image: a numpy.ndarray or or its subclass
        :param block_start_index  The real block start index, not considering the padding

        Returns
        -------
        Returns the padded estimate block as a numpy array.

        """

        block_pad = self.options.block_pad
        image_size = self.image_size
        ndims = self.imdims

        # Apply padding
        end_index = block_start_index + self.block_size + block_pad
        start_index = block_start_index - block_pad

        idx = tuple(slice(start, stop) for start, stop in zip(start_index, end_index))

        # If the padded block fits within the image boundaries, nothing special
        # is needed to extract it. Normal numpy slicing notation is used.
        if (image_size >= end_index).all() and (start_index >= 0).all():
            return image[idx]

        else:
            block_size = tuple(i + 2 * block_pad for i in self.block_size)
            # Block outside the image boundaries will be filled with zeros.
            block = numpy.zeros(block_size)
            # If the start_index is close to the image boundaries, it is very
            # probable that padding will introduce negative start_index values.
            # In such case the first pixel index must be corrected.
            if (start_index < 0).any():
                block_start = numpy.negative(start_index.clip(max=0))
                image_start = start_index + block_start
            else:
                block_start = (0,) * ndims
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

            block_idx = tuple(slice(start, stop) for start, stop in zip(block_start, block_end))
            image_idx = tuple(slice(start, stop) for start, stop in zip(image_start, end_index))

            block[block_idx] = image[image_idx]

            return block

    # region Get Output
    def get_result(self, cast_to_8bit=False):
        """
        Show fusion result. This is a temporary solution for now
        calling Fiji through ITK. An internal viewer would be
        preferable.
        """
        if cast_to_8bit:
            result = self.estimate.copy()
            result *= (255.0 / result.max())
            result[result < 0] = 0
            return Image(result, self.voxel_size)

        return Image(self.estimate, self.voxel_size)

    def save_to_hdf(self):
        """
        Save result to the miplib data structure.

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

    # endregion

    def close(self):
        if self.options.memmap_estimates:
            del self.estimate
            del self.estimate_new
        if not self.options.disable_tau1:
            del self.prev_estimate

        shutil.rmtree(self.memmap_directory)
