# coding=utf-8
"""
deconvolve.py

Copyright (C) 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the GPU accelerated miplib deconvolution
algorithms. The MultiViewFusionRLCuda class implements all the same methods
as the MultiViewFusionRL class, for non-accelerated iterative image fusion.

"""

import itertools

import numpy as np
import miplib.processing.ops_ext as ops_ext
import cupy as cp
from cupyx.scipy import fftpack
from . import deconvolve
import miplib.processing.ndarray as ops_array


class DeconvolutionRLCuda(deconvolve.DeconvolutionRL):
    """
    This class implements GPU accelerated versions of the iterative image
    fusion algorithms discussed in:

    Koho, S., Deguchi, T., & Hanninen, P. E. (2015). A software tool for
    tomographic axial superresolution in STED microscopy. Journal of Microscopy,
    260(2), 208–218. http://doi.org/10.1111/jmi.12287

    MultiViewFusionRLCuda is inherits most of its functionality from
    MultiViewFusionRL (see deconvolve.py).
    """
    def __init__(self, image, psf, writer, options):
        """
        :param image:   the image as a Image object
        :param psf:     the psf as an Image object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        deconvolve.DeconvolutionRL.__init__(self, image, psf, writer, options)
        self._fft_plan = fftpack.get_fft_plan(cp.zeros(self.block_size, dtype=cp.complex64))
        self.__get_fourier_psfs()

    def compute_estimate(self):
        """
            Calculates a single RL fusion estimate. There is no reason to call this
            function -- it is used internally by the class during fusion process.
        """
        self.estimate_new[:] = np.zeros(self.image_size, dtype=np.float32)

        # Iterate over blocks
        iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
        pad = self.options.block_pad
        block_idx = tuple(slice(pad, pad + block) for block in self.block_size)

        for pos in itertools.product(*iterables):

            estimate_idx = tuple(slice(j, j + k) for j, k in zip(pos, self.block_size))
            index = np.array(pos, dtype=int)

            if self.options.block_pad > 0:
                h_estimate_block = self.get_padded_block(self.estimate, index.copy()).astype(np.complex64)
            else:
                h_estimate_block = self.estimate[estimate_idx].astype(np.complex64)

            # # Execute: cache = convolve(PSF, estimate), non-normalized
            h_estimate_block_new = self._fft_convolve(h_estimate_block, self.psf_fft)

            # Execute: cache = data/cache. Add background bias if requested.
            h_image_block = self.get_padded_block(self.image, index.copy()).astype(np.float32)
            if self.options.rl_background != 0:
                h_image_block += self.options.rl_background
            ops_ext.inverse_division_inplace(h_estimate_block_new, h_image_block)

            # Execute correlation with PSF
            h_estimate_block_new = self._fft_convolve(h_estimate_block_new, self.adj_psf_fft).real

            # Get new weights
            self.estimate_new[estimate_idx] = h_estimate_block_new[block_idx]

        # TV Regularization (doesn't seem to do anything miraculous).
        if self.options.tv_lambda > 0 and self.iteration_count > 0:
            dv_est = ops_ext.div_unit_grad(self.estimate, self.image_spacing)
            self.estimate_new = ops_array.safe_divide(self.estimate_new,
                                                      (1.0 - self.options.rltv_lambda * dv_est))

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
        psf = self.psf[:]
        if self.imdims == 3:
            adj_psf = psf[::-1, ::-1, ::-1]
        else:
            adj_psf = psf[::-1, ::-1]

        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)

        psf_fft = ops_array.expand_to_shape(psf, padded_block_size).astype(np.complex64)
        adj_psf_fft = ops_array.expand_to_shape(adj_psf, padded_block_size).astype(np.complex64)
        psf_fft = np.fft.fftshift(psf_fft)
        adj_psf_fft = np.fft.fftshift(adj_psf_fft)

        self.psf_fft = np.fft.fftn(psf_fft)
        self.adj_psf_fft = np.fft.fftn(adj_psf_fft)






