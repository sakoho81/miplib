import numpy
import itertools

from accelerate.cuda.fft import FFTPlan, fft_inplace, ifft_inplace
from numba import cuda, vectorize
from supertomo.utils import generic_utils as genutils


class MultiViewFusionRLCuda(fusion.MultiViewFusionRL):
    def __init__(self, data, options):
        fusion.MultiViewFusionRL.__init__(self, data, options)

        threadpergpublock = 32, 32, 8
        self.blockpergrid = self.__best_grid_size(
            tuple(reversed(self.block_size)), threadpergpublock)

        FFTPlan(shape=self.block_size, itype=numpy.complex64,
                otype=numpy.complex64)

        print('kernel config: %s x %s' % (self.blockpergrid, threadpergpublock))

    def compute_estimate(self):
        if "multiplicative" in self.options.fusion_method:
            estimate_new = numpy.ones(self.image_size, dtype=numpy.float32)
        else:
            estimate_new = numpy.zeros(self.image_size, dtype=numpy.float32)

        # d_estimate_new = cuda.to_device(estimate_new)

        # Iterate over blocks
        block_nr = 1
        for x, y, z in itertools.product(xrange(0, self.image_size[0], self.block_size[0]),
                                         xrange(0, self.image_size[1], self.block_size[1]),
                                         xrange(0, self.image_size[2], self.block_size[2])):

            index = numpy.array((x, y, z), dtype=int)
            if self.options.block_pad > 0:
                h_estimate_block = self.__get_padded_block(index.copy())
            else:
                h_estimate_block = self.estimate[index[0]:index[0] + self.block_size[0],
                                   index[1]:index[1] + self.block_size[1],
                                   index[2]:index[2] + self.block_size[2]]

            print "The current block is %i" % block_nr
            block_nr += 1

            d_estimate_block = cuda.to_device(h_estimate_block.astype(numpy.complex64))

            # Iterate over views
            for view in xrange(self.n_views):

                h_psf = genutils.expand_to_shape(self.psfs[view], d_estimate_block.shape)
                d_psf = cuda.to_device(h_psf.astype(numpy.complex64))

                h_adj_psf = genutils.expand_to_shape(self.adj_psfs[view], d_estimate_block.shape)
                d_adj_psf = cuda.to_device(h_adj_psf.astype(numpy.complex64))

                # Execute: cache = convolve(PSF, estimate), non-normalized
                fft_inplace(d_estimate_block)
                fft_inplace(d_psf)
                self.vmult(d_estimate_block, d_psf, out=d_estimate_block)

                ifft_inplace(d_estimate_block)

                h_estimate_block = d_estimate_block.copy_to_host().astype(numpy.float32)

                # Execute: cache = data/cache
                self.data.set_active_image(view, self.options.channel, self.options.scale, "registered")
                block = self.data.get_registered_block(self.block_size, self.options.block_pad, index.copy())

                # ops_ext.inverse_division_inplace(cache, block)
                with numpy.errstate(divide="ignore"):
                    h_estimate_block = block / h_estimate_block
                    h_estimate_block[h_estimate_block == numpy.inf] = 0.0
                    h_estimate_block = numpy.nan_to_num(h_estimate_block)

                # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
                # Convolution with virtual PSFs is performed here as well, if
                # necessary

                d_estimate_block = cuda.to_device(h_estimate_block.astype(numpy.complex64))

                fft_inplace(d_estimate_block)
                fft_inplace(d_adj_psf)
                self.vmult(d_estimate_block, d_psf, out=d_estimate_block)

                ifft_inplace(d_estimate_block)

                h_estimate_block = d_estimate_block.copy_to_host().astype(numpy.float32)

                # Update the contribution from a single view to the new estimate
                if self.options.block_pad == 0:
                    if "multiplicative" in self.options.fusion_method:
                        estimate_new[index[0]:index[0] + self.block_size[0],
                        index[1]:index[1] + self.block_size[1],
                        index[2]:index[2] + self.block_size[2]] *= h_estimate_block
                    else:

                        estimate_new[index[0]:index[0] + self.block_size[0],
                                     index[1]:index[1] + self.block_size[1],
                        index[2]:index[2] + self.block_size[2]] += h_estimate_block
                else:
                    pad = self.options.block_pad

                    if "multiplicative" in self.options.fusion_method:
                        estimate_new[index[0]:index[0] + self.block_size[0],
                                     index[1]:index[1] + self.block_size[1],
                                     index[2]:index[2] + self.block_size[2]] *= h_estimate_block[pad:-pad]
                    else:
                        # print "The block size is ", self.block_size
                        estimate_new[index[0]:index[0] + self.block_size[0],
                        index[1]:index[1] + self.block_size[1],
                        index[2]:index[2] + self.block_size[2]] += h_estimate_block[pad:pad + self.block_size[0],
                                                                   pad:pad + self.block_size[1],
                                                                   pad:pad + self.block_size[2]]

    @staticmethod
    def __best_grid_size(size, tpb):
        bpg = numpy.ceil(numpy.array(size, dtype=numpy.float) / tpb).astype(numpy.int).tolist()
        return tuple(bpg)

    @staticmethod
    @vectorize(['complex64(complex64, complex64)'], target='cuda')
    def vmult(a, b):
        return a * b



