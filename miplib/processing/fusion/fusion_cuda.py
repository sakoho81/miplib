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
from numba import cuda, vectorize
from pyculib.fft import FFTPlan, fft_inplace, ifft_inplace

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
    def __init__(self, data, options):
        """
        :param data:    a ImageData object

        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        fusion.MultiViewFusionRL.__init__(self, data, options)

        padded_block_size = self.block_size + 2*self.options.block_pad
        threadpergpublock = 32, 32, 8
        blockpergrid = self.__best_grid_size(
            tuple(reversed(padded_block_size)), threadpergpublock)

        FFTPlan(padded_block_size, itype=numpy.complex64, otype=numpy.complex64)



        #yself.scaler = numpy.full(self.image_size, 1.0/self.n_views,
        # dtype=numpy.float32)

        print(('Optimal kernel config: %s x %s' % (blockpergrid, threadpergpublock)))

        self.__get_fourier_psfs()

    def compute_estimate(self):
        """
            Calculates a single RL fusion estimate. There is no reason to call this
            function -- it is used internally by the class during fusion process.
        """
        print('Beginning the computation of the %i. estimate' % \
              (self.iteration_count + 1))

        if "multiplicative" in self.options.fusion_method:
            self.estimate_new[:] = numpy.ones(self.image_size, dtype=numpy.float32)
        else:
            self.estimate_new[:] = numpy.zeros(self.image_size, dtype=numpy.float32)

        stream1 = cuda.stream()
        stream2 = cuda.stream()

        # Iterate over views
        for idx, view in enumerate(self.views):

            psf_fft = self.psfs_fft[idx]
            adj_psf_fft = self.adj_psfs_fft[idx]

            self.data.set_active_image(view, self.options.channel,
                                       self.options.scale, "registered")

            weighting = self.weights[idx]
            iterables = (range(0, m, n) for m, n in zip(self.image_size, self.block_size))
            pad = self.options.block_pad
            block_idx = tuple(slice(pad, pad + block) for block in self.block_size)

            for pos in itertools.product(*iterables):

                estimate_idx = tuple(slice(j, j + k) for j, k in zip(pos, self.block_size))
                index = numpy.array(pos, dtype=int)

                if self.options.block_pad > 0:
                    h_estimate_block = self.get_padded_block(
                        index.copy()).astype(numpy.complex64)
                else:
                    h_estimate_block = self.estimate[estimate_idx].astype(numpy.complex64)

                d_estimate_block = cuda.to_device(h_estimate_block, stream=stream1)
                d_psf = cuda.to_device(psf_fft, stream=stream2)

                # Execute: cache = convolve(PSF, estimate), non-normalized
                fft_inplace(d_estimate_block, stream=stream1)
                stream2.synchronize()

                self.vmult(d_estimate_block, d_psf, out=d_estimate_block)
                ifft_inplace(d_estimate_block)

                h_estimate_block_new = d_estimate_block.copy_to_host()

                # Execute: cache = data/cache
                h_image_block = self.data.get_registered_block(self.block_size,
                                                               self.options.block_pad,
                                                               index.copy()).astype(numpy.float32)
                h_estimate_block_new *= weighting
                ops_ext.inverse_division_inplace(h_estimate_block_new,
                                                 h_image_block)

                d_estimate_block = cuda.to_device(h_estimate_block_new,
                                                  stream=stream1)
                d_adj_psf = cuda.to_device(adj_psf_fft, stream=stream2)

                fft_inplace(d_estimate_block, stream=stream1)
                stream2.synchronize()
                self.vmult(d_estimate_block, d_adj_psf, out=d_estimate_block)
                ifft_inplace(d_estimate_block)
                h_estimate_block_new = d_estimate_block.copy_to_host().real

                # Update the contribution from a single view to the new estimate
                if self.options.block_pad == 0:
                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[estimate_idx] *= h_estimate_block_new
                    else:

                        self.estimate_new[estimate_idx] += h_estimate_block_new
                else:

                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[estimate_idx] *= h_estimate_block_new[block_idx]

                    else:
                        # print "The block size is ", self.block_size
                        self.estimate_new[estimate_idx] += h_estimate_block_new[block_idx]

        # # Divide with the number of projections
        # if "summative" in self.options.fusion_method:
        #     # self.estimate_new[:] = self.float_vmult(self.estimate_new,
        #     #                                         self.scaler)
        #     self.estimate_new *= (1.0 / self.n_views)
        # else:
        #     self.estimate_new[self.estimate_new < 0] = 0
        #     self.estimate_new[:] = ops_array.nroot(self.estimate_new,
        #                                            self.n_views)

        # TV Regularization (doesn't seem to do anything miraculous).
        if self.options.rltv_lambda > 0 and self.iteration_count > 0:
            dv_est = ops_ext.div_unit_grad(self.estimate, self.voxel_size)
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

        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)

        memmap_shape = numpy.insert(numpy.array(padded_block_size), 0,
                                    len(self.views))

        if self.options.disable_fft_psf_memmap:
            self.psfs_fft = numpy.zeros(tuple(memmap_shape),
                                        dtype=numpy.complex64)
            self.adj_psfs_fft = numpy.zeros(tuple(memmap_shape),
                                            dtype=numpy.complex64)
        else:
            psfs_fft_f = os.path.join(self.memmap_directory, 'psf_fft_f.dat')
            self.psfs_fft = numpy.memmap(psfs_fft_f, dtype='complex64',
                                         mode='w+', shape=tuple(memmap_shape))
            adj_psfs_fft_f = os.path.join(self.memmap_directory,
                                          'adj_psf_fft_f.dat')
            self.adj_psfs_fft = numpy.memmap(adj_psfs_fft_f, dtype='complex64',
                                             mode='w+', shape=tuple(memmap_shape))

        for idx in range(self.n_views):
            self.psfs_fft[idx] = ops_array.expand_to_shape(
                self.psfs[idx], padded_block_size).astype(numpy.complex64)
            self.adj_psfs_fft[idx] = ops_array.expand_to_shape(
                self.adj_psfs[idx], padded_block_size).astype(numpy.complex64)
            self.psfs_fft[idx] = numpy.fft.fftshift(self.psfs_fft[idx])
            self.adj_psfs_fft[idx] = numpy.fft.fftshift(self.adj_psfs_fft[idx])

            fft_inplace(self.psfs_fft[idx])
            fft_inplace(self.adj_psfs_fft[idx])

    def close(self):
        if not self.options.disable_fft_psf_memmap:
            del self.psfs_fft
            del self.adj_psfs_fft
        fusion.MultiViewFusionRL.close(self)






