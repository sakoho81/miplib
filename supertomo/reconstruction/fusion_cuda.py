# coding=utf-8
"""
fusion.py

Copyright (C) 2016 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains the GPU accelerated SuperTomo2 multi-view image fusion
algorithms. The MultiViewFusionRLCuda class implements all the same methods
as the MultiViewFusionRL class, for non-accelerated iterative image fusion.

"""

import numpy
import itertools
import os

from accelerate.cuda.fft import FFTPlan, fft_inplace, ifft_inplace
from numba import cuda, vectorize
from supertomo.utils import generic_utils as genutils
import fusion
import ops_ext


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

        # Select views to fuse
        if self.options.fuse_views == -1:
            self.views = xrange(self.n_views)
        else:
            self.views = self.options.fuse_views

        self.n_views = len(self.views)

        #yself.scaler = numpy.full(self.image_size, 1.0/self.n_views,
        # dtype=numpy.float32)

        print('Optimal kernel config: %s x %s' % (blockpergrid, threadpergpublock))

        self.__get_fourier_psfs()

    def compute_estimate(self):
        """
            Calculates a single RL fusion estimate. There is no reason to call this
            function -- it is used internally by the class during fusion process.
        """
        print 'Beginning the computation of the %i. estimate' % self.iteration_count

        if "multiplicative" in self.options.fusion_method:
            self.estimate_new[:] = numpy.ones(self.image_size, dtype=numpy.float32)
        else:
            self.estimate_new[:] = numpy.zeros(self.image_size, dtype=numpy.float32)

        # Iterate over blocks
        stream1 = cuda.stream()
        stream2 = cuda.stream()

        for idx, view in enumerate(self.views):
            psf_fft = self.psfs_fft[idx]
            adj_psf_fft = self.adj_psfs_fft[idx]

            self.data.set_active_image(view, self.options.channel,
                                       self.options.scale, "registered")

            for x, y, z in itertools.product(xrange(0, self.image_size[0], self.block_size[0]),
                                             xrange(0, self.image_size[1], self.block_size[1]),
                                             xrange(0, self.image_size[2], self.block_size[2])):

                index = numpy.array((x, y, z), dtype=int)

                if self.options.block_pad > 0:
                    h_estimate_block = self.get_padded_block(
                        index.copy()).astype(numpy.complex64)
                else:
                    h_estimate_block = self.estimate[
                                       index[0]:index[0] + self.block_size[0],
                                       index[1]:index[1] + self.block_size[1],
                                       index[2]:index[2] + self.block_size[
                                           2]].astype(numpy.complex64)

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
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] *= h_estimate_block_new
                    else:

                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] += h_estimate_block_new
                else:
                    pad = self.options.block_pad

                    if "multiplicative" in self.options.fusion_method:
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] *= h_estimate_block_new[
                                 pad:pad + self.block_size[0],
                                 pad:pad + self.block_size[1],
                                 pad:pad + self.block_size[2]
                             ]
                    else:
                        # print "The block size is ", self.block_size
                        self.estimate_new[
                            index[0]:index[0] + self.block_size[0],
                            index[1]:index[1] + self.block_size[1],
                            index[2]:index[2] + self.block_size[2]
                        ] += h_estimate_block_new[
                                 pad:pad + self.block_size[0],
                                 pad:pad + self.block_size[1],
                                 pad:pad + self.block_size[2]
                             ]
        # Divide with the number of projections
        if "summative" in self.options.fusion_method:
            # self.estimate_new[:] = self.float_vmult(self.estimate_new,
            #                                         self.scaler)
            self.estimate_new *= (1.0 / self.n_views)
        else:
            self.estimate_new[:] = genutils.nroot(self.estimate_new,
                                                  self.n_views)

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
        print "Pre-calculating PSFs"

        padded_block_size = tuple(self.block_size + 2*self.options.block_pad)
        memmap_shape = numpy.insert(numpy.array(padded_block_size), 0,
                                    self.n_views)

        psfs_fft_f = os.path.join(self.memmap_directory, 'psf_fft_f.dat')
        self.psfs_fft = numpy.memmap(psfs_fft_f, dtype='complex64',
                                     mode='w+', shape=tuple(memmap_shape))
        adj_psfs_fft_f = os.path.join(self.memmap_directory,
                                      'adj_psf_fft_f.dat')
        self.adj_psfs_fft = numpy.memmap(adj_psfs_fft_f, dtype='complex64',
                                         mode='w+', shape=tuple(memmap_shape))

        for view in self.views:
            self.psfs_fft[view] = genutils.expand_to_shape(
                self.psfs[view], padded_block_size).astype(numpy.complex64)
            self.adj_psfs_fft[view] = genutils.expand_to_shape(
                self.adj_psfs[view], padded_block_size).astype(numpy.complex64)
            self.psfs_fft[view] = numpy.fft.fftshift(self.psfs_fft[view])
            self.adj_psfs_fft[view] = numpy.fft.fftshift(self.adj_psfs_fft[view])

            fft_inplace(self.psfs_fft[view])
            fft_inplace(self.adj_psfs_fft[view])

    def close(self):
        del self.psfs_fft
        del self.adj_psfs_fft
        fusion.MultiViewFusionRL.close(self)






