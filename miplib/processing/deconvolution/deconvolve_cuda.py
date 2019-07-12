# coding=utf-8
"""
deconvolve.py

Copyright (C) 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the GPU accelerated miplib multi-view image fusion
algorithms. The MultiViewFusionRLCuda class implements all the same methods
as the MultiViewFusionRL class, for non-accelerated iterative image fusion.

"""

import itertools

import numpy
import miplib.processing.ops_ext as ops_ext
from numba import cuda, vectorize
from pyculib.fft import FFTPlan, fft_inplace, ifft_inplace

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

        padded_block_size = self.block_size + 2*self.options.block_pad
        if self.imdims == 2:
            threadpergpublock = 32, 32
        else:
            threadpergpublock = 32, 32, 8

        blockpergrid = self.__best_grid_size(
            tuple(reversed(padded_block_size)), threadpergpublock)

        FFTPlan(padded_block_size, itype=numpy.complex64, otype=numpy.complex64)


        print(('Optimal kernel config: %s x %s' % (blockpergrid, threadpergpublock)))

        self.__get_fourier_psfs()

    def compute_estimate(self):
        """
            Calculates a single RL fusion estimate. There is no reason to call this
            function -- it is used internally by the class during fusion process.
        """
        print('Beginning the computation of the %i. estimate' % \
              (self.iteration_count + 1))

        self.estimate_new[:] = numpy.zeros(self.image_size, dtype=numpy.float32)

        # Iterate over blocks
        stream1 = cuda.stream()
        stream2 = cuda.stream()

        iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
        pad = self.options.block_pad
        block_idx = tuple(slice(pad, pad + block) for block in self.block_size)

        if self.imdims == 2:
            for pos in itertools.product(*iterables):

                estimate_idx = tuple(slice(j, j + k) for j, k in zip(idx, self.block_size))
                index = numpy.array(pos, dtype=int)

                if self.options.block_pad > 0:
                    h_estimate_block = self.get_padded_block(self.estimate, index.copy()).astype(numpy.complex64)
                else:
                    h_estimate_block = self.estimate[estimate_idx].astype(numpy.complex64)

                d_estimate_block = cuda.to_device(h_estimate_block, stream=stream1)
                d_psf = cuda.to_device(self.psf_fft, stream=stream2)

                # Execute: cache = convolve(PSF, estimate), non-normalized
                fft_inplace(d_estimate_block, stream=stream1)
                stream2.synchronize()

                self.vmult(d_estimate_block, d_psf, out=d_estimate_block)
                ifft_inplace(d_estimate_block)

                h_estimate_block_new = d_estimate_block.copy_to_host()

                # Execute: cache = data/cache
                h_image_block = self.get_padded_block(self.image, index.copy()).astype(numpy.float32)
                ops_ext.inverse_division_inplace(h_estimate_block_new, h_image_block)

                d_estimate_block = cuda.to_device(h_estimate_block_new,
                                                  stream=stream1)
                d_adj_psf = cuda.to_device(self.adj_psf_fft, stream=stream2)

                fft_inplace(d_estimate_block, stream=stream1)
                stream2.synchronize()
                self.vmult(d_estimate_block, d_adj_psf, out=d_estimate_block)
                ifft_inplace(d_estimate_block)
                h_estimate_block_new = d_estimate_block.copy_to_host().real

                self.estimate_new[estimate_idx] = h_estimate_block_new[block_idx]

        # TV Regularization (doesn't seem to do anything miraculous).
        if self.options.rltv_lambda > 0 and self.iteration_count > 0:
            dv_est = ops_ext.div_unit_grad(self.estimate, self.image_spacing)
            with numpy.errstate(divide="ignore"):
                self.estimate_new /= (1.0 - self.options.rltv_lambda * dv_est)
                self.estimate_new[self.estimate_new == numpy.inf] = 0.0
                self.estimate_new[:] = numpy.nan_to_num(self.estimate_new)

        # Update estimate inplace. Get convergence statistics.
        return ops_ext.update_estimate_poisson(self.estimate,
                                               self.estimate_new,
                                               self.options.convergence_epsilon)

    @staticmethod
    def __best_grid_size(size, tpb):
        bpg = numpy.ceil(numpy.array(size, dtype=numpy.float) / tpb).astype(numpy.int).tolist()
        return tuple(bpg)

    @staticmethod
    @vectorize(['complex64(complex64, complex64)'], target='cuda')
    def vmult(a, b):
        """
        Implements complex array multiplication on GPU

        Parameters
        ----------
        :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64
        :param  b

        Returns
        -------

        a*b

        """
        return a * b

    @staticmethod
    @vectorize(['float32(float32, float32)'], target='cuda')
    def float_vmult(a, b):
        """
        Implements array multiplication on GPU

        Parameters
        ----------
        :param  a  Two Numpy arrays of the same shape, dtype=numpy.float32
        :param  b

        Returns
        -------

        a*b

        """

        return a * b

    def __get_fourier_psfs(self):
        """
        Pre-calculates the PSFs during image fusion process.
        """
        print("Pre-calculating PSFs")

        psf = self.psf[:]
        if self.imdims == 3:
            adj_psf = psf[::-1, ::-1, ::-1]
        else:
            adj_psf = psf[::-1, ::-1]

        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)

        self.psfs_fft = ops_array.expand_to_shape(psf, padded_block_size).astype(numpy.complex64)
        self.adj_psf_fft = ops_array.expand_to_shape(adj_psf, padded_block_size).astype(numpy.complex64)
        self.psf_fft = numpy.fft.fftshift(self.psf_fft)
        self.adj_psf_fft = numpy.fft.fftshift(self.adj_psf_fft)

        fft_inplace(self.psf_fft)
        fft_inplace(self.adj_psf_fft)






