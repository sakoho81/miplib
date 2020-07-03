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
import sys
import tempfile
import time
import pandas

import numpy
import miplib.processing.ops_ext as ops_ext
from scipy.ndimage.interpolation import zoom
from scipy.ndimage.filters import uniform_filter
from scipy.signal import fftconvolve, medfilt

import miplib.processing.to_string as ops_output
import miplib.processing.ndarray
import miplib.psf.psfgen as psfgen
from miplib.data.containers import temp_data
from miplib.data.containers.image import Image
from miplib.data.messages.image_writer_wrappers import ImageWriterBase
from miplib.processing.segmentation import masking
from numpy.fft import fftn, fftshift

import miplib.analysis.resolution.fourier_ring_correlation as frc

class DeconvolutionRL(object):
    """
    The Richardson-Lucy fusion is a result of simultaneous deblurring of
    several 3D volumes.
    """

    def __init__(self, image, psf, writer, options):
        """
        :param image:    a MyImage object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        assert isinstance(image, Image)
        assert isinstance(psf, Image)
        if options.save_intermediate_results:
            assert issubclass(writer.__class__, ImageWriterBase)

        self.image = image
        self.psf = psf
        self.options = options
        self.writer = writer

        self.image_size = numpy.array(self.image.shape)
        self.image_spacing = self.image.spacing
        self.psf_spacing = self.psf.spacing
        self.imdims = image.ndim

        self.__get_psfs()

        if options.verbose:
            print("The original image size is %s" % (self.image_size,))

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
                                                   shape=tuple(self.image_size)), self.image_spacing)

            estimate_f = os.path.join(self.memmap_directory, "estimate.dat")
            self.estimate = Image(numpy.memmap(estimate_f, dtype=numpy.float32,
                                               mode='w+',
                                               shape=tuple(self.image_size)), self.image_spacing)
        else:
            self.estimate = Image(numpy.zeros(tuple(self.image_size),
                                              dtype=numpy.float32), self.image_spacing)
            self.estimate_new = Image(numpy.zeros(tuple(self.image_size),
                                                  dtype=numpy.float32), self.image_spacing)

        if not self.options.disable_tau1:
            prev_estimate_f = os.path.join(self.memmap_directory, "prev_estimate.dat")
            self.prev_estimate = Image(numpy.memmap(prev_estimate_f, dtype=numpy.float32,
                                                    mode='w+',
                                                    shape=tuple(self.image_size)), self.image_spacing)

        padded_block_size = tuple(i + 2 * self.options.block_pad for i in self.block_size)
        if options.verbose:
            print("The deconvolution will be run with %i blocks" % self.num_blocks)
            print("The internal block size is %s" % (padded_block_size,))

        # Create temporary directory and data file.
        self.column_headers = ('t', 'tau1', 'leak', 'e',
                             's', 'u', 'n', 'uesu')
        self._progress_parameters = numpy.empty((self.options.max_nof_iterations, len(self.column_headers)),
                                                dtype=numpy.float32)

        # Get initial resolution (in case you are using the FRC based stopping.)
        if self.options.rl_frc_stop > 0:
            self.resolution = frc.calculate_single_image_frc(self.image, 
                self.options).resolution["resolution"]

        # Enable automatic background correction with --rl-auto-background
        if self.options.rl_auto_background:
            background_mask = masking.make_local_intensity_based_mask(
                image, threshold=30, kernel_size=60, invert=True)
            masked_image = Image(image * background_mask, image.spacing)
            self.options.rl_background = numpy.mean(masked_image[masked_image > 0])

    @property
    def progress_parameters(self):
        return pandas.DataFrame(data=self._progress_parameters, columns=self.column_headers)

    def compute_estimate(self):
        """
        Calculates a single RL deconvolution estimate. There is no reason to call this
        function -- it is used internally by the class during fusion process.
        """

        if self.options.verbose:
            print('Beginning the computation of the %i. estimate' % self.iteration_count)

        self.estimate_new[:] = numpy.float32(0)

        # Iterate over blocks
        block_nr = 1
        iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
        pad = self.options.block_pad
        cache_idx = tuple(slice(pad, pad + block) for block in self.block_size)

        for idx in itertools.product(*iterables):

            estimate_idx = tuple(slice(j, j+k) for j, k in zip(idx, self.block_size))

            index = numpy.array(idx, dtype=int)

            if self.options.block_pad > 0:
                estimate_block = self.get_padded_block(
                    self.estimate, index.copy())
                image_block = self.get_padded_block(self.image, index.copy())
            else:
                estimate_block = self.estimate[estimate_idx]
                image_block = self.image[estimate_idx]

            # print "The current block is %i" % block_nr
            block_nr += 1

            # Execute: cache = convolve(PSF, estimate), non-normalized
            cache = fftconvolve(estimate_block, self.psf, mode='same')

            if self.options.rl_background != 0:
                cache += self.options.rl_background

                # ops_ext.inverse_division_inplace(cache, image_block)
            with numpy.errstate(divide="ignore"):
                cache = image_block.astype(numpy.float32) / cache
                cache[cache == numpy.inf] = 0.0
                cache = numpy.nan_to_num(cache)

            # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
            # Convolution with virtual PSFs is performed here as well, if
            # necessary
            cache = fftconvolve(cache, self.adj_psf, mode='same')

            self.estimate_new[estimate_idx] = cache[cache_idx]

        if self.options.tv_lambda > 0 and self.iteration_count > 0:
            if self.estimate.ndim == 2:
                spacing = list(self.image_spacing)
                spacing.insert(0,1)
                dv_est = ops_ext.div_unit_grad(numpy.expand_dims(self.estimate, 0),
                                               spacing)[0]
            else:
                dv_est = ops_ext.div_unit_grad(self.estimate, self.image_spacing)
            with numpy.errstate(divide="ignore"):
                self.estimate_new /= (1.0 - self.options.tv_lambda * dv_est)
                self.estimate_new[self.estimate_new == numpy.inf] = 0.0
                self.estimate_new[:] = numpy.nan_to_num(self.estimate_new)

        return ops_ext.update_estimate_poisson(self.estimate,
                                               self.estimate_new,
                                               self.options.convergence_epsilon)

    def execute(self):
        """
        This is the main fusion function
        """

        save_intermediate_results = self.options.save_intermediate_results

        first_estimate = self.options.first_estimate

        if first_estimate == 'image':
            self.estimate[:] = self.image[:].astype(numpy.float32)
        elif first_estimate == 'blurred':
            self.estimate[:] = uniform_filter(self.image, 3).astype(numpy.float32)
        elif first_estimate == 'image_mean':
            self.estimate[:] = numpy.float32(numpy.mean(self.image[:]))
        elif first_estimate == 'constant':
            self.estimate[:] = numpy.float32(self.options.estimate_constant)
        else:
            raise NotImplementedError(repr(first_estimate))

        self.iteration_count = 0
        max_count = self.options.max_nof_iterations
        initial_photon_count = self.image[:].sum()

        bar = ops_output.ProgressBar(0,
                                     max_count,
                                     totalWidth=40,
                                     show_percentage=False)

        self._progress_parameters = numpy.zeros((self.options.max_nof_iterations, len(self.column_headers)),
                                                dtype=numpy.float32)


        # duofrc_prev = 0
        # The Fusion calculation starts here
        # ====================================================================
        try:
            while True:

                if (
                        self.options.update_blind_psf > 0 and 
                        self.iteration_count > 0 and 
                        (self.iteration_count+1) % self.options.update_blind_psf == 0
                   ):
                    self.psf = psfgen.generate_frc_based_psf(Image(self.estimate, self.image_spacing), self.options)
                    self.__get_psfs()
                    self.image = self.estimate.copy()

                info_map = {}
                ittime = time.time()

                self.prev_estimate[:] = self.estimate.copy()

                e, s, u, n = self.compute_estimate()

                self.iteration_count += 1
                photon_leak = 1.0 - (e + s + u) / initial_photon_count
                u_esu = u / (e + s + u)

                tau1 = abs(self.estimate - self.prev_estimate).sum() / abs(
                    self.prev_estimate).sum()
                info_map['TAU1=%s'] = tau1

                t = time.time() - ittime
                leak = 100 * photon_leak

                if self.options.verbose:
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
                    # self.temp_data.save_image(
                    #     self.estimate,
                    #     'result_%s.tif' % self.iteration_count
                    # )
                    self.writer.write(Image(self.estimate, self.image_spacing))

                # Check if it's time to stop:
                if int(u) == 0 and int(n) == 0:
                    stop_message = 'The number of non converging photons reached to zero.'
                    break
                elif self.iteration_count >= max_count:
                    stop_message = 'The number of iterations reached to maximal count: %s' % max_count
                    break
                elif not self.options.disable_tau1 and tau1 <= self.options.stop_tau:
                    stop_message = 'Desired tau-threshold achieved'
                    break
                elif self.options.rl_frc_stop > 0:
                    resolution_new = frc.calculate_single_image_frc(
                            Image(self.estimate, self.image_spacing), self.options).resolution["resolution"]
                    frc_diff = numpy.abs(self.resolution - resolution_new)
                    if frc_diff <= self.options.rl_frc_stop:
                        print('Desired FRC diff reached after {} iterations'.format(
                            self.iteration_count))
                        break
                    else:
                        self.resolution = resolution_new
                    
                # elif self.iteration_count >= 4 and abs(frc_diff) <= .0001:
                #     stop_message = 'FRC stop condition reached'
                #     break
                else:
                    continue

        except KeyboardInterrupt:
            stop_message = 'Iteration was interrupted by user.'

        # if self.num_blocks > 1:
        #     self.estimate = self.estimate[0:real_size[0], 0:real_size[1], 0:real_size[2]]
        if self.options.verbose:
            print()
            bar.updateComment(' ' + stop_message)
            bar(self.iteration_count)
            print()

    def __get_psfs(self):
        """
        Reads the PSFs from the HDF5 data structure and zooms to the same pixel
        size with the registered images, of selected scale and channel.
        """
        psf_orig = self.psf[:]

        # Zoom to the same voxel size
        zoom_factors = tuple(x / y for x, y in zip(self.psf_spacing, self.image_spacing))
        psf_new = zoom(psf_orig, zoom_factors).astype(numpy.float32)

        psf_new /= psf_new.sum()

        # Save the zoomed and rotated PSF, as well as its mirrored version
        self.psf = psf_new
        if self.imdims == 3:
            self.adj_psf = psf_new[::-1, ::-1, ::-1]
        else:
            self.adj_psf = psf_new[::-1, ::-1]

    def get_result(self):
        """
        Show fusion result. This is a temporary solution for now
        calling Fiji through ITK. An internal viewer would be
        preferable.
        """

        return Image(self.estimate, self.image_spacing)

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

    def get_8bit_result(self, denoise=False):
        """
        Returns the current estimate (the fusion result) as an 8-bit uint, rescaled
        to the full 0-255 range.
        """
        if denoise:
            image = medfilt(self.estimate)
        else:
            image = self.estimate

        image *= (255.0 / image.max())
        image[image < 0] = 0
        return Image(image.astype(numpy.uint8), self.image_spacing)

    # def get_saved_data(self):
    #     return pandas.DataFrame(columns=self.column_headers, data=self._progress_parameters)

    def close(self):
        if self.options.memmap_estimates:
            del self.estimate
            del self.estimate_new
        if not self.options.disable_tau1:
            del self.prev_estimate

        shutil.rmtree(self.memmap_directory)

        #self.temp_data.close_data_file()
