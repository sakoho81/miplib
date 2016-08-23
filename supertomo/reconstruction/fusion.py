"""
fusion.py

Copyright (C) 2014 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

The functions in this file can be used to combine or fuse multiple 3D image
volumes. The code is based on Richardson-Lucy 3D deconvolution.

"""

import itertools
import sys
import time

import numpy
from numpy.testing import utils as numpy_utils
from scipy.signal import fftconvolve
from scipy.ndimage.interpolation import zoom

from supertomo.io import image_data, temp_data, tiffile
from supertomo.reconstruction import ops_ext
from supertomo.utils import itkutils, generic_utils as genutils
from supertomo.ui import show


class MultiViewFusionRL:
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
        self.n_views = self.data.get_number_of_images("registered")
        self.data.set_active_image(0, 0, options.scale, "registered")
        self.image_size = self.data.get_image_size()
        self.estimate = numpy.zeros(self.image_size, dtype=numpy.float64)
        self.iteration_count = 0

        # Setup blocks
        data.set_active_image(0, self.options.channel, self.options.scale, "registered")
        self.num_blocks = options.num_blocks
        self.block_size = numpy.ceil(self.image_size / self.num_blocks)

        # Setup PSFs
        self.psfs = []
        self.adj_psfs = []
        self.__get_psfs()
        if "opt" in self.options.fusion_method:
            self.virtual_psfs = []
            self.__compute_virtual_psfs()
        else:
            pass

        # Create temporary directory and data file.
        self.data_to_save = ('count', 't', 'mn', 'mx', 'tau1', 'tau2', 'leak', 'e',
                             's', 'u', 'n', 'u_esu', 'mem')
        self.temp_data = temp_data.TempData()
        self.temp_data.create_data_file("fusion_data.csv", self.data_to_save)
        self.temp_data.write_comment('Fusion Command: %s' % (' '.join(map(str, sys.argv))))

    def compute_estimate(self):
        """
        Calculates a single RL fusion estimate. There is no reason to call this
        function -- it is used internally by the class during fusion process.
        """

        if self.options.verbose:
            print 'Beginning the computation of the %ith estimate' % self.iteration_count

        if "multiplicative" in self.options.fusion_method:
            estimate_new = numpy.ones(self.image_size, dtype=numpy.float64)
        else:
            estimate_new = numpy.zeros(self.image_size, dtype=numpy.float64)

        # Iterate over blocks
        for x, y, z in itertools.product(xrange(0, self.image_size[0], self.block_size[0]),
                                         xrange(0, self.image_size[1], self.block_size[1]),
                                         xrange(0, self.image_size[2], self.block_size[2])):

            index = numpy.array((x, y, z), dtype=int)
            estimate_block = self.estimate[index: index + self.block_size]

            # Iterate over views
            for view in xrange(self.n_views):

                # Execute: cache = convolve(PSF, estimate), non-normalized
                cache = fftconvolve(estimate_block, self.psfs[view], mode='same')

                # Execute: cache = data/cache
                self.data.set_active_image(view, self.options.channel, self.options.scale, "registered")
                block, block_size = self.data.get_registered_block(self.block_size, index)
                ops_ext.inverse_division_inplace(cache, block)

                # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
                # Convolution with virtual PSFs is performed here as well, if
                # necessary
                if "opt" in self.options.fusion_method:
                    cache = fftconvolve(cache, self.virtual_psfs[view])
                else:
                    cache = fftconvolve(cache, self.adj_psfs[view])

                # Update the contribution from a single view to the new estimate
                if "multiplicative" in self.options.fusion_method:
                    estimate_new[index[0]:index[0]+block_size[0],
                                 index[1]:index[1]+block_size[1],
                                 index[2]:index[2]+block_size[2]] *= cache.real
                else:
                    estimate_new[index[0]:index[0] + block_size[0],
                                 index[1]:index[1] + block_size[1],
                                 index[2]:index[2] + block_size[2]] += cache.real

        # Divide with the number of projections
        if "summative" in self.options.fusion_method:
            estimate_new *= (1.0 / self.n_views)
        else:
            estimate_new = genutils.nroot(estimate_new, self.n_views)

        return ops_ext.update_estimate_poisson(self.estimate,
                                               estimate_new,
                                               self.options.convergence_epsilon)

    def __compute_virtual_psfs(self):
        """
        Implements a Virtual PSF calculation routine, as described in "Efficient
        Bayesian-based multiview deconvolution" by Preibich et al in Nature
        Methods 11/6 (2014)
        """

        if self.options.verbose:
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
            self.virtual_psfs.append(virtual_psf)

    def execute(self):
        """
        This is the main fusion function
        """

        save_intermediate_results = self.options.save_intermediate_results

        first_estimate = self.options.first_estimate

        self.data.set_active_image(0,
                                   self.options.channel,
                                   self.options.scale,
                                   "registered")

        if first_estimate == 'first_image':
            self.estimate = self.data[:].astype(numpy.float64)
        elif first_estimate == 'first_image_mean':
            self.estimate = numpy.full(
                self.data.get_image_size(),
                float(numpy.mean(self.data[:])),
                dtype=numpy.float64
            )
        elif first_estimate == 'sum_of_all':
            for i in range(self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate += self.data[:].astype(numpy.float64)
            self.estimate *= (1.0/self.n_views)
        elif first_estimate == 'simple_fusion':
            self.estimate = self.data[:]
            for i in xrange(1, self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate = (self.estimate - (self.estimate - self.data[:]).clip(min=0)).clip(min=0).astype(numpy.float64)
        elif first_estimate == 'average_af_all':
            self.estimate = numpy.zeros(self.image_size, dtype=numpy.float64)
            for i in range(self.n_views):
                self.data.set_active_image(i,
                                           self.options.channel,
                                           self.options.scale,
                                           "registered")
                self.estimate += (self.data[:].astype(numpy.float64) / self.n_views)
        else:
            raise NotImplementedError(repr(first_estimate))

        stop_message = ''
        self.iteration_count = 0
        max_count = self.options.max_nof_iterations
        initial_photon_count = self.data[:].sum()

        bar = genutils.ProgressBar(0,
                                   max_count,
                                   totalWidth=40,
                                   show_percentage=False)

        # The Fusion calculation starts here
        # ====================================================================
        try:
            while True:

                info_map = {}
                ittime = time.time()

                prev_estimate = self.estimate.copy()

                e, s, u, n = self.compute_estimate()

                self.iteration_count += 1
                photon_leak = 1.0 - (e + s + u) / initial_photon_count
                leak = 100 * photon_leak
                u_esu = u / (e + s + u)

                mn, mx = self.estimate.min(), self.estimate.max()
                tau1 = abs(self.estimate - prev_estimate).sum() / abs(prev_estimate).sum()
                mem = int(numpy_utils.memusage() / 2 ** 20)

                # Update UI
                info_map['E/S/U/N=%s/%s/%s/%s'] = int(e), int(s), int(u), int(n)
                info_map['LEAK=%s%%'] = 100 * photon_leak
                info_map['U/ESU=%s'] = u_esu
                info_map['TAU1=%s'] = tau1
                info_map['MEM=%sMB'] = mem
                info_map['TIME=%ss'] = t = time.time() - ittime
                bar.updateComment(' ' + ', '.join([k % (genutils.tostr(info_map[k])) for k in sorted(info_map)]))
                bar(self.iteration_count)

                # Save parameters to file
                self.temp_data.write_row(', '.join(self.data_to_save))

                # Save intermediate image
                if save_intermediate_results:
                    self.temp_data.save_image(
                        genutils.cast_to_dtype(self.estimate, numpy.uint8, remove_outliers=False),
                        'result_%s.tif' % self.iteration_count
                    )

                # Check if it's time to stop:
                if int(u) == 0 and int(n) == 0:
                    stop_message = 'The number of non converging photons reached to zero.'
                    break
                elif self.iteration_count >= max_count:
                    stop_message = 'The number of iterations reached to maximal count: %s' % max_count
                    break
                elif 'tau1' in self.data_to_save and tau1 <= self.options.rltv_stop_tau:
                    stop_message = 'Desired tau-threshold achieved'
                    break
                else:
                    continue

        except KeyboardInterrupt:
            stop_message = 'Iteration was interrupted by user.'

        print
        bar.updateComment(' ' + stop_message)
        bar(self.iteration_count)
        print

    def __get_psfs(self):
        """


        """

        self.n_views = self.data.get_number_of_images("registered")
        n_psfs = self.data.get_number_of_images("psf")

        assert n_psfs == 1 or n_psfs == self.n_views, "Wrong amount of PSFs in the data file"

        self.psfs = []

        self.data.set_active_image(0, 0, 100, "psf")
        psf_spacing = self.data.get_voxel_size()
        psf_orig = self.data[:]

        for i in range(0, self.n_views):
            # Get the necessary information (spatial transformation
            # and spacing from the images
            self.data.set_active_image(i, "registered")
            image_spacing = self.data.get_voxel_size()
            transform = self.data.get_transform()

            # In case that each view has a PSF, new data will be read at
            # every loop iteration. For a single PSF situation one read
            # in the beginning is enough.
            if n_psfs > 1:
                self.data.set_active_image(i, 0, 100, "psf")
                psf_orig = self.data[:]
                psf_spacing = self.data.get_voxel_size()

            # Zoom to the same voxel size
            zoom_factors = psf_spacing / image_spacing
            psf_new = zoom(psf_orig, zoom_factors)

            # Rotate PSF with the spatial transformation from the
            # image registration task.
            if i > 0:
                psf_new = itkutils.rotate_psf(
                    psf_new,
                    transform,
                    spacing=image_spacing,
                    return_numpy=True
                )

            # Save the zoomed and rotated PSF, as well as its mirrored version
            self.psfs.append(psf_new)
            self.adj_psfs.append(psf_new[::-1, ::-1, ::-1])

    def save_to_hdf(self):
        self.data.set_active_image(0, "registered")
        spacing = self.data.get_voxel_size()
        self.data.add_fused_image(self.estimate, spacing)

    def save_to_tiff(self, filename):
        tiffile.imsave(filename, self.estimate)

    def show_result(self):
        show.evaluate_3d_image(self.estimate)
