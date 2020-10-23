# coding=utf-8
"""
fusion.py

Copyright (C) 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the GPU accelerated miplib multi-view image fusion
algorithms. The MultiViewFusionRLCuda class implements all the same methods
as the MultiViewFusionRL class, for non-accelerated iterative image fusion.

"""

import itertools
import os

import numpy
import miplib.processing.ops_ext as ops_ext
import cupy as cp
from cupyx.scipy import fftpack

import miplib.processing.fusion.fusion as fusion
import miplib.processing.ndarray as ops_array


class MultiViewFusionRLCuda(fusion.MultiViewFusionRL):
    """
    This class implements GPU accelerated versions of the iterative image
    fusion algorithms discussed in:

    Koho, S., Deguchi, T., & Hanninen, P. E. (2015). A software tool for
    tomographic axial superresolution in STED microscopy. Journal of Microscopy,
    260(2), 208–218. http://doi.org/10.1111/jmi.12287

    MultiViewFusionRLCuda is inherits most of its functionality from
    MultiViewFusionRL (see fusion.py).
    """
    def __init__(self, data, writer , options):
        """
        :param data:    a ImageData object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        :param writer:  a writer object that can save intermediate results
        """
        fusion.MultiViewFusionRL.__init__(self, data, writer, options)

        padded_block_size = self.block_size + 2*self.options.block_pad

        self._fft_plan = fftpack.get_fft_plan(cp.zeros(padded_block_size, dtype=cp.complex64))
        self.__get_fourier_psfs()

    def compute_estimate(self):
        """
            Calculates a single RL fusion estimate. There is no reason to call this
            function -- it is used internally by the class during fusion process.
        """
        print(f'Beginning the computation of the {self.iteration_count + 1}. estimate')

        if "multiplicative" in self.options.fusion_method:
            self.estimate_new[:] = numpy.ones(self.image_size, dtype=numpy.float32)
        else:
            self.estimate_new[:] = numpy.zeros(self.image_size, dtype=numpy.float32)


        # Iterate over views
        for idx, view in enumerate(self.views):

            psf_fft = self.psfs_fft[idx]
            adj_psf_fft = self.adj_psfs_fft[idx]

            self.data.set_active_image(view, self.options.channel,
                                       self.options.scale, "registered")

            weighting = self.weights[idx]
            background = self.background[idx]

            iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
            pad = self.options.block_pad
            block_idx = tuple(slice(pad, pad + block) for block in self.block_size)

            for pos in itertools.product(*iterables):

                estimate_idx = tuple(slice(j, j + k) for j, k in zip(pos, self.block_size))
                index = numpy.array(pos, dtype=int)

                if self.options.block_pad > 0:
                    h_estimate_block = self.get_padded_block(
                        self.estimate, index.copy()).astype(numpy.complex64)
                else:
                    h_estimate_block = self.estimate[estimate_idx].astype(numpy.complex64)

                # Convolve estimate block with the PSF
                h_estimate_block_new = self._fft_convolve(h_estimate_block, psf_fft)

                # Apply weighting
                h_estimate_block_new *= weighting

                # Apply background
                h_estimate_block_new += background

                # Divide image block with the convolution result
                h_image_block = self.data.get_registered_block(self.block_size,
                                                               self.options.block_pad,
                                                               index.copy()).astype(numpy.float32)

                #h_estimate_block_new = ops_array.safe_divide(h_image_block, h_estimate_block_new)
                ops_ext.inverse_division_inplace(h_estimate_block_new,
                                                 h_image_block)

                # Correlate with adj PSF
                h_estimate_block_new = self._fft_convolve(h_estimate_block_new, adj_psf_fft).real

                # Update the contribution from a single view to the new estimate
                self._write_estimate_block(h_estimate_block_new, estimate_idx, block_idx)

        # Divide with the number of projections
        if "summative" in self.options.fusion_method:
            # self.estimate_new[:] = self.float_vmult(self.estimate_new,
            #                                         self.scaler)
            self.estimate_new *= (1.0 / self.n_views)
        else:
            self.estimate_new[self.estimate_new < 0] = 0
            self.estimate_new[:] = ops_array.nroot(self.estimate_new,
                                                   self.n_views)

        # TV Regularization (doesn't seem to do anything miraculous).
        if self.options.tv_lambda > 0 and self.iteration_count > 0:
            dv_est = ops_ext.div_unit_grad(self.estimate, self.voxel_size)
            self.estimate_new = ops_array.safe_divide(self.estimate, (1.0 - self.options.rltv_lambda * dv_est))

        # Update estimate inplace. Get convergence statistics.
        return ops_ext.update_estimate_poisson(self.estimate,
                                               self.estimate_new,
                                               self.options.convergence_epsilon)

    def _fft_convolve(self, h_data, h_kernel):
        """
        Calculate a convolution on GPU, using FFTs.

        :param h_data: a Numpy array with the data to convolve
        :param h_kernel: a Numpy array with the convolution kernel. The kernel
        should already be in Fourier domain (To avoid repeating the transform at
        every iteration.)
        """
        #todo: See whether to add back streams. I removed them on Cupy refactor.

        d_data = cp.asarray(h_data)
        d_data = fftpack.fftn(d_data, overwrite_x=True, plan=self._fft_plan)

        d_kernel = cp.asarray(h_kernel)
        d_data *= d_kernel

        d_data = fftpack.ifftn(d_data, overwrite_x=True, plan=self._fft_plan)
        return cp.asnumpy(d_data)

    def __get_fourier_psfs(self):
        """
        Pre-calculates the PSFs during image fusion process.
        """
        print("Pre-calculating PSFs")

        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)

        memmap_shape = (self.n_views,) + padded_block_size

        if self.options.disable_fft_psf_memmap:
            self.psfs_fft = numpy.zeros(memmap_shape, dtype=numpy.complex64)
            self.adj_psfs_fft = numpy.zeros(memmap_shape, dtype=numpy.complex64)
        else:
            psfs_fft_f = os.path.join(self.memmap_directory, "psf_fft_f.dat")
            self.psfs_fft = numpy.memmap(psfs_fft_f, dtype='complex64', mode='w+', shape=memmap_shape)
            adj_psfs_fft_f = os.path.join(self.memmap_directory, "adj_psf_fft_f.dat")
            self.adj_psfs_fft = numpy.memmap(adj_psfs_fft_f, dtype='complex64', mode='w+', shape=memmap_shape)

        for idx in range(self.n_views):
            self.psfs_fft[idx] = ops_array.expand_to_shape(
                self.psfs[idx], padded_block_size).astype(numpy.complex64)
            self.adj_psfs_fft[idx] = ops_array.expand_to_shape(
                self.adj_psfs[idx], padded_block_size).astype(numpy.complex64)
            self.psfs_fft[idx] = numpy.fft.fftshift(self.psfs_fft[idx])
            self.adj_psfs_fft[idx] = numpy.fft.fftshift(self.adj_psfs_fft[idx])

            self.psfs_fft[idx] = cp.asnumpy(fftpack.fftn(cp.asarray(self.psfs_fft[idx]), plan=self._fft_plan))
            self.adj_psfs_fft[idx] = cp.asnumpy(fftpack.fftn(cp.asarray(self.adj_psfs_fft[idx]), plan=self._fft_plan))

    def close(self):
        if not self.options.disable_fft_psf_memmap:
            del self.psfs_fft
            del self.adj_psfs_fft
        fusion.MultiViewFusionRL.close(self)






