"""
fusion.py

Copyright (C) 2014 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

The functions in this file can be used to combine or fuse multiple 3D image
volumes. The code is based on Richardson-Lucy 3D deconvolution.

"""

import numpy
import os
import tempfile
import sys
import time
from numpy.testing import utils as numpy_utils

from supertomo.reconstruction import ops_ext
from supertomo.cli import utils
import supertomo.io.image_data as image_data


class MultiViewFusionRL():
    """
    The Richardson-Lucy fusion process is based on the Deconvolve
    superclass, implemented in iocbio.microscope.deconvolution
    module. The difference is that the fusion
    is a result of simultaneous deblurring of several 3D volumes.
    Also, as a difference to the deconvolution functions in IOCBIO,
    the whole fusion process is contained in this class

    The projections and their corresponding point-spread-functions
    (PSF) should be input as a list of IOCBIO ImageStack objects.
    The fusion result will be returned as an ImageStack object as
    well.
    """
    def __init__(self, data, options):
        """
        :param psfs:    a list of 3D PSFs (each of which should match
                        their corresponding projection), as ImageStack
                        objects
        :param images:  a list of ImageStack objects of the 3D projections
                        to be fused
        :param options: command line options that control the behavior
                        of the fusion algorithm
        """
        assert isinstance(data, image_data.ImageData)

        self.data_type = images[0].get_data_type()
        self.float_type = utils.float2dtype(options.float_type)

        self.options = options



        for i in range(len(images)):
            psfs[i].images, images[i].images = deconvolution.get_coherent_images(
                psfs[i],
                images[i],
                self.float_type
            )
            self.data.append(utils.rescale_to_min_max(
                images[i].images.astype(options.float_type), 0, 1)
            )
            self.psfs.append(utils.rescale_to_min_max(
                psfs[i].images.astype(options.float_type), 0, 1)
            )

        # A 3D shape needs to be passed for the Deconvolution
        deconvolution.Deconvolve.__init__(self, self.data[0].shape, options=options)

        self.voxel_sizes = [s * 1000 for s in images[0].get_voxel_sizes()]
        self.pathinfo = images[0].get_pathinfo()
        self.final_shape = images[0].get_shape()

        self.alpha = options.rltv_alpha

        self.psf_adj_psf_f = None
        self.psf_adj_data = None
        self.lambda_lsq = None
        self.lambda_lsq_coeff = None

        self.set_cache_dir(tempfile.mkdtemp('-iocbio.fuse'))
        self.set_save_data(self.pathinfo, self.shape, self.data_type)
        self.set_test_data()

        self.estimate = None

        self.virtual_psfs = []

        if "opt" in self.options.fusion_method:
            self._compute_virtual_psfs()
        else:
            pass

    def compute_estimate(self):
        """
        Calculates a single RL fusion estimate. There is no reason to call this
        function -- it is used internally by the class during fusion process.
        """

        if self.options.verbose:
            print 'Entering %s.compute_estimate' % self.__class__.__name__

        if "multiplicative" in self.options.fusion_method:
            estimate_new = numpy.ones(self.data[0].shape, dtype=self.data[0].dtype)
        else:
            estimate_new = numpy.zeros(self.data[0].shape, dtype=self._cache.dtype)

        for i in range(len(self.data)):
            self.set_convolve_kernel(self.psfs[i])

            psf_f = self.convolve_kernel_fourier
            adj_psf_f = self.convolve_kernel_fourier_conj

            cache = self._cache
            cache[:] = self.estimate

            #print cache.max()

            # Execute: cache = convolve(PSF, estimate), non-normalized
            self._fft_plan.execute()
            cache *= psf_f
            self._ifft_plan.execute()

            # Execute: cache = data/cache
            ops_ext.inverse_division_inplace(cache, self.data[i])

            # Execute: cache = convolve(PSF(-), cache), inverse of non-normalized
            # Convolution with virtual PSFs is performed here as well, if
            # necessary
            self._fft_plan.execute()
            if "opt" in self.options.fusion_method:
                cache *= self.virtual_psfs[i]
            else:
                cache *= adj_psf_f
            self._ifft_plan.execute()

            if "multiplicative" in self.options.fusion_method:
                estimate_new *= cache.real
            else:
                estimate_new += cache

        # Divide with the number of projections

        #print estimate_new.max()

        if "summative" in self.options.fusion_method:
            estimate_new *= (1.0/len(self.data))
        else:
            estimate_new = utils.nroot(estimate_new, len(self.data))


        #The deconvolution term has now been calculated. The next step is to
        #calculate the Total Variation based regularisation term

        if self.options.rltv_estimate_lambda:
                dv_est = self._compute_dv_estimate()
                self.lambda_ = self.lambda_lsq
                estimate_new /= (1.0 - self.lambda_lsq * dv_est)
        elif self.lambda_> 0:
                dv_est = self._compute_dv_estimate()
                estimate_new /= (1.0 - self.lambda_ * dv_est)
        else:
                pass

        return ops_ext.update_estimate_poisson(self.estimate,
                                               estimate_new,
                                               self.convergence_epsilon)

    def _compute_virtual_psfs(self):
        """
        Implements a Virtual PSF calculation routine, as described in "Efficient
        Bayesian-based multiview deconvolution" by Preibich et al in Nature
        Methods 11/6 (2014)
        """

        if self.options.verbose:
            print "Caclulating Virtual PSFs"

        for i in range(len(self.data)):
            self.set_convolve_kernel(self.psfs[i])
            adj_psf_current = self.convolve_kernel_fourier_conj
            virtual_new = numpy.ones(self.psfs[0].shape, dtype=self.psfs[0].dtype)
            for j in range(len(self.data)):
                if j == i:
                    pass
                else:
                    self.set_convolve_kernel(self.psfs[j])
                    psf_not_current = self.convolve_kernel_fourier
                    adj_psf_not_current = self.convolve_kernel_fourier_conj

                    cache = self._cache
                    cache[:] = adj_psf_current*psf_not_current*adj_psf_not_current
                    self._ifft_plan.execute()

                    virtual_new *= cache.real

            virtual_new = self.psfs[i][::-1, ::-1, ::-1]*virtual_new
            virtual_new = utils.rescale_to_min_max(virtual_new, 0, 1)
            self.set_convolve_kernel(virtual_new)
            self.virtual_psfs.append(self.convolve_kernel_fourier)

    def _compute_dv_estimate(self):
        """
        An internal function for calculating the total variation term.
        """

        dv_estimate = None

        # In case lambda estimation option is selected

        if self.options.rltv_compute_lambda_lsq or self.options.rltv_estimate_lambda:

            dv_estimate = ops_ext.div_unit_grad(self.estimate, self.voxel_sizes)
            lambda_lsq = ((1.0 - self.estimate.real) * dv_estimate).sum() / (dv_estimate * dv_estimate).sum()

            if self.lambda_lsq_coeff is None:
                lambda_lsq_coeff_path = os.path.join(self.cache_dir, 'lambda_lsq_coeff.txt')
                lambda_lsq_coeff = self.options.rltv_lambda_lsq_coeff

                if (lambda_lsq_coeff in [None, 0.0]
                    and self.options.first_estimate == 'last result'
                    and os.path.isfile(lambda_lsq_coeff_path)):

                    try:
                        lambda_lsq_coeff = float(open(lambda_lsq_coeff_path).read())
                    except Exception, msg:
                        print 'Failed to read lambda_lsq_coeff cache: %s' % (msg)
                        return -1

                if lambda_lsq_coeff == 0.0:
                    # C * lambda_0 = 50/SNR
                    lambda_lsq_coeff = 50.0 / self.snr / lambda_lsq

                if lambda_lsq_coeff < 0:
                    print 'Negative lambda_lsq, skip storing lambda_lsq_coeff'
                else:
                    self.lambda_lsq_coeff = lambda_lsq_coeff
                    f = open(lambda_lsq_coeff_path, 'w')
                    f.write(str(lambda_lsq_coeff))
                    f.close()

                print 'lambda-opt=', 50.0 / self.snr
                print 'lambda-lsq-0=', lambda_lsq
                print 'lambda-lsq-coeff=', lambda_lsq_coeff

            else:
                lambda_lsq_coeff = self.lambda_lsq_coeff

            lambda_lsq *= lambda_lsq_coeff
            self.lambda_lsq = lambda_lsq

        elif self.lambda_ > 0:
            dv_estimate = ops_ext.div_unit_grad(self.estimate, self.voxel_sizes)

        else:
            pass

        return dv_estimate

    def deconvolve(self):
        """
        This is the main fusion function
        """
        if self.options.verbose:
            print 'Entering %s.deconvolve' % self.__class__.__name__

        save_intermediate_results = self.options.save_intermediate_results

        data_to_save = ('count', 't', 'mn', 'mx', 'tau1', 'tau2', 'leak', 'e',
                        's', 'u', 'n', 'u_esu', 'mem')

        # The deconvolution data, together with intermediate results (if
        # requested) are saved to a temporary folder -- /tmp on Linux.
        data_file_name = os.path.join(self.cache_dir, 'deconvolve_data.txt')

        count = -1
        append_data_file = False

        # There are various alternatives for the first fusion estimate. The selection
        # will be made through the command line option interface. The default is
        # 'input image' which will select the regular STED image (presuming that it
        # is the first image in the projections list). With low SNR data
        # 'convolved input image' option often works better.
        first_estimate = self.options.first_estimate
        if first_estimate == 'input image':
            self.estimate = self.data[0]
        elif first_estimate == 'image_mean':
            self.estimate = numpy.full(self.data[0].shape, numpy.mean(self.data[0]), dtype=self.data[0].dtype)
        elif first_estimate == 'convolved input image':
            self.set_convolve_kernel(self.psfs[0])
            self.estimate = self.convolve(self.data[0])
        elif first_estimate == 'sum of all projections':
            self.estimate = 0.5*self.data[0]+0.5*self.data[1]
        elif first_estimate == 'stupid tomo':
            self.estimate = (self.data[0]-(self.data[0]-self.data[1]).clip(min=0)).clip(min=0)
        elif first_estimate == 'average':
            self.estimate = numpy.zeros(self.data[0].shape, dtype=self.data[0].dtype)
            for i in range(len(self.data)):
                self.estimate += (self.data[i] / len(self.data))
        elif first_estimate == '2x convolved input image':
            self.estimate = self.convolve(self.convolve(self.data[0]))
        elif first_estimate == 'last result':
            if os.path.isfile(data_file_name):
                data_file = RowFile(data_file_name)
                data, data_titles = data_file.read(with_titles=True)
                data_file.close()
                counts = map(int, data['count'])
                for count in reversed(counts):
                    fn = os.path.join(self.cache_dir, 'result_%s.tif' % count)
                    if os.path.isfile(fn):
                        append_data_file = True
                        break
                if append_data_file:
                    print 'Loading the last result from %r.' % fn
                    stack = image_stack.ImageStack.load(fn)
                    self.estimate = numpy.array(
                        stack.images,
                        dtype=self.float_type
                    )
                    f = open(os.path.join(
                        self.cache_dir,
                        'deconvolve_data_%s_%s.txt' % (counts[0], count)), 'w'
                    )
                    fi = open(data_file_name)
                    f.write(fi.read())
                    fi.close()
                    f.close()
                    if count != counts[-1]:
                        print 'Expected result %s but got %s, ' \
                              'fixing %r' % (counts[-1], count, data_file_name)
                        data_file = RowFile(data_file_name, titles=data_titles)
                        for c in range(count + 1):
                            data_file.write(
                                ', '.join([str(data[t][c]) for t in data_titles]))
                        data_file.close()

            if not append_data_file:
                print 'Found no results in %r, ' \
                      'using input image as estimate.' % self.cache_dir
                count = -1
                self.estimate = self.data[0]
        else:
            raise NotImplementedError(`first_estimate`)

        prev_estimate = self.estimate
        initial_photon_count = self.data[0].sum()

        print 'Initial photon count: %.3f' % initial_photon_count
        print 'Initial minimum: %.3f' % (self.estimate.min())
        print 'Initial maximum: %.3f' % (self.estimate.max())

        max_count = self.options.max_nof_iterations
        bar = utils.ProgressBar(0,
                                max_count,
                                totalWidth=40,
                                show_percentage=False)

        if self.options.rltv_estimate_lambda or \
                self.options.rltv_compute_lambda_lsq:
            data_to_save += ('lambda_lsq',)

        if self.test_data is not None:
            data_to_save += ('mseo',)
            test_data_norm2 = (self.test_data ** 2).sum()

        data_file = RowFile(data_file_name,
                            titles=data_to_save,
                            append=append_data_file)
        data_file.comment(
            'Fusion Command: %s' % (' '.join(map(str, sys.argv)))
        )

        if 'mseo' in data_file.extra_titles and 'mseo' not in data_to_save:
            data_to_save += ('mseo',)

        stop_message = ''
        stop = count >= max_count
        if stop:
            stop_message = 'The number of iterations reached ' \
                           'to maximal count: %s' % max_count
        else:
            if save_intermediate_results:
                self.save(self.estimate, 'result_%sm1.tif' % (count + 1))
        # The Fusion calculation starts here
        # ====================================================================
        try:
            min_mse = 1e300
            min_mseo = 1e300
            min_tau = 1e300
            max_lambda = 0.0
            while not stop:
                count += 1
                self.count = count
                info_map = {}
                ittime = time.time()

                prev2_estimate = prev_estimate.copy()
                prev_estimate = self.estimate.copy()

                e, s, u, n = self.compute_estimate()

                info_map['E/S/U/N=%s/%s/%s/%s'] = int(e), int(s), int(u), int(n)
                photon_leak = 1.0 - (e + s + u) / initial_photon_count
                info_map['LEAK=%s%%'] = 100 * photon_leak

                if 'leak' in data_to_save:
                    leak = 100 * photon_leak

                if 'u_esu' in data_to_save:
                    u_esu = u / (e + s + u)
                    info_map['U/ESU=%s'] = u_esu

                if 'mn' in data_to_save:
                    mn, mx = self.estimate.min(), self.estimate.max()

                if 'mse' in data_to_save:
                    self.set_convolve_kernel(self.psfs[0])
                    eh = self.convolve(self.estimate, inplace=False)
                    mse = ((eh - self.data[0]) ** 2).sum() / self.data[0].size
                    info_map['MSE=%s'] = mse

                if 'klic' in data_to_save:
                    klic = ops_ext.kullback_leibler_divergence(self.data[0], eh, 1.0)
                    info_map['KLIC=%s'] = klic

                if 'mseo' in data_to_save:
                    mseo = ((self.estimate - self.test_data) ** 2).sum() / self.data[0].size
                    info_map['MSEO=%s'] = mseo

                if 'tau1' in data_to_save:
                    tau1 = abs(self.estimate - prev_estimate).sum() / abs(prev_estimate).sum()
                    tau2 = abs(self.estimate - prev2_estimate).sum() / abs(prev2_estimate).sum()
                    info_map['TAU1/2=%s/%s'] = (tau1, tau2)

                if 'lambda_lsq' in data_to_save:
                    lambda_lsq = self.lambda_lsq
                    if lambda_lsq > max_lambda:
                        max_lambda = lambda_lsq
                    info_map['LAM/MX=%s/%s'] = lambda_lsq, max_lambda

                if 'mem' in data_to_save:
                    mem = int(numpy_utils.memusage() / 2 ** 20)
                    info_map['MEM=%sMB'] = mem

                info_map['TIME=%ss'] = t = time.time() - ittime

                bar.updateComment(' ' + ', '.join([k % (utils.tostr(info_map[k])) for k in sorted(info_map)]))
                bar(count)

                if 'mse' in data_to_save and mse < min_mse:
                    min_mse = mse
                    #self.save(discretize(self.estimate), 'deconvolved_%s_min_mse.tif' % (count))

                if 'mseo' in data_to_save and mseo < min_mseo:
                    min_mseo = mseo
                    #self.save(discretize(self.estimate), 'deconvolved_%s_min_mseo.tif' % (count))

                if save_intermediate_results:
                    self.save(
                        utils.cast_to_dtype(self.estimate, self.data_type, remove_outliers=True),
                        'result_%s.tif' % count
                    )

                # Stopping criteria:
                stop = True
                #if abs(photon_leak) > 0.6:
                #   stop_message = 'Photons leak is too large: %.3f%%>20%%' % (photon_leak * 100)
                if int(u) == 0 and int(n) == 0:
                    stop_message = 'The number of non converging photons reached to zero.'
                elif count >= max_count:
                    stop_message = 'The number of iterations reached to maximal count: %s' % (max_count)
                elif 'tau1' in data_to_save and tau1 <= self.options.rltv_stop_tau:
                    stop_message = 'Desired tau-threshold achieved'
                else:
                    stop = False

                exec 'data_file.write(%s)' % (', '.join(data_to_save))
                if not save_intermediate_results and stop:
                    self.save(
                        utils.cast_to_dtype(self.estimate, self.data_type, remove_outliers=False),
                        'result_%s.tif' % count
                    )

        except KeyboardInterrupt:
            stop_message = 'Iteration was interrupted by user.'

        print
        bar.updateComment(' ' + stop_message)
        bar(count)
        print

        data_file.close()

        if self.options.show_plots:
            plots.plot_rowfile(data_file_name, shape=440, tight=False)

        if self.options.output_cast:
            return image_stack.ImageStack(
                utils.contract_to_shape(
                    utils.cast_to_dtype(self.estimate, self.data_type, remove_outliers=False),
                    self.final_shape),
                self.save_pathinfo,
                options=self.options
            )
        else:
            return image_stack.ImageStack(
                utils.contract_to_shape(self.estimate, self.final_shape),
                self.save_pathinfo,
                options=self.options
            )