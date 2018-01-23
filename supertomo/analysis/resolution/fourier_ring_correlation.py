"""
Sami Koho 01/2017

Image resolution measurement by Fourier Ring Correlation.

The code in this file was modified from FRC implementation by Filippo Arcadu

The original file header is shown below:

#######                                                                      #######
#######         FOURIER RING CORRELATION ANALYSIS FOR RESOLUTION             #######
#######                                                                      #######
#######  This routine evaluates the resol by means of the fourier            #######
#######  ring correlation (FRC). The inputs are two reconstructions made     #######
#######  with the same algorithm on a sinogram storing the odd-map_disted    #######
#######  projections and on an other one storing the even-map_disted projec.-#######
#######  tions. The two images are transformed with the FFT and their        #######
#######  transform centered. Then, rings of increasing radius R are selec-   #######
#######  ted, starting from the origin of the Fourier space, and the         #######
#######  Fourier coefficients lying inside the ring are used to calculate    #######
#######  the FRC at R, that is FRC(R), with the following formula:           #######
#######                                                                      #######
####### FRC(R)=(sum_{i in R}I_{1}(r_{i})*I_{2}(r_{i}))/sqrt((sum_{i in R}    #######
#######        ||I_{1}(r_{i})||^{2})*(sum_{i in R}||I_{2}(r_{i})||^{2}))     #######
#######                                                                      #######
#######  At the same time, the so-called '2*sigma' curve is calculated at    #######
#######  each step R:                                                        #######
#######                F_{2*sigma}(R) = 2/sqrt(N_{p}(R)/2)                   #######
#######  where N_{p} is the number of pixels in the ring of radius R.        #######
#######  Then, the crossing point between FRC(R) and 2*sigma(R) is found     #######
#######  as the first zero crossing point with negative slope of the dif-    #######
#######  ference curve:                                                      #######
#######                D(R) = FRC(R) - F_{2*sigma}(R)                        #######
#######  The resol is calculated as real space distance correspon-           #######
#######  to this intersection point.                                         ####### 
#######                                                                      #######
#######  Reference:                                                          #######
#######  "Fourier Ring Correlation as a resol criterion for super-           #######
#######  resol microscopy", N. Banterle et al., 2013, Journal of             #######
#######  Structural Biology, 183  363-367.                                   #######
#######                                                                      #######                                    
#######        Author: Filippo Arcadu, arcusfil@gmail.com, 16/09/2013        #######
#######                                                                      #######
####################################################################################
####################################################################################
####################################################################################
"""

import numpy as np
from scipy import optimize
import datetime
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import rc, s

from pydeconvolution.io.myimage import MyImage as Image


class FRC(object):
    """
    A class for calcuating 2D Fourier ring correlation. Contains
    methods to calculate the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, args):
        assert isinstance(image1, Image)
        assert isinstance(image2, Image)

        assert image1.get_dimensions() == image2.get_dimensions()
        #TODO: Could calculate the Nyquist according to the pixel size here. Also the sphere should
        # be in physical units, instead of pixel numbers
        if image1.get_dimensions()[0] == 1 and image1.ndim() == 3:
            image1 = Image(image1[0], image1.get_spacing()[1:])
            image2 = Image(image2[0], image1.get_spacing()[1:])

        self.pixel_size = image1.get_spacing()[0]
        self.args = args

        self.ndims = image1.ndim()

        # Get the Nyquist frequency
        dims = image1.get_dimensions()

        nmax = max(dims)
        self.freq_nyq = int(np.floor(nmax / 2.0))

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in dims)
        grid = np.meshgrid(*axes)
        self.map_dist = np.sqrt(reduce(lambda x, y: x+y, (i**2 for i in grid)))

        # FFT transforms of the input images
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1.get_array())).real
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2.get_array())).real

        # Get thickness of the rings
        width_ring = args.width_ring
        #print "Ring width: %i" % width_ring

        self.frc = None
        self.two_sigma = None
        self.resolution = None
        self.n_points = None
        self.spatial_freq = None
        self.all = None

    def set_images(self, image1, image2):
        """
        It is possible to reinitialize and object with a new set of images.
        This functionality was added with batch processing tasks in mind.

        :param image1: Should be a myimage.MyImage object
        :param image2:  Should be a myimage.MyImage object
        """
        args = self.args
        self.__init__(image1, image2, args)

    def get_frc(self):
        """
        Get the FRC points. In case they have not been calculated already,
        the FRC calculation will be run first.

        :return: Returns a dictionary {y:frc_values, x:frequencies,
                 fit:curve fit to the y values, equation:the equation for the
                 fitted function.
        """
        if self.frc is None:
            return self.calculate_frc()
        else:
            return self.frc

    def get_two_sigma(self):
        """
        Get the Two Sigma curve. In case it has not been calculated already,
        the curve will be calculated first

        :return: Returns a dictinary with the following keys:
                 y:
                 The calculated two sigma points calculated at every
                 fourier ring bin

                 x:
                 The spatial frequencies from 0 to 1 (1 meaning the
                 Nyquist frequency.

                 fit:
                 A polynomial curve fit on the *y* values with the values *x*

                 eq: equation of the fitted function,

                 crit:the employed resolution criterion.
        """
        if self.two_sigma is None:
            return self.calculate_two_sigma()
        else:
            return self.two_sigma

    def get_resolution(self):
        """
        Get the resolution measurement results.

        :return: Returns a dictinary with the following keys:

        resolution:
        The resolution in real spatial scale. This assumes that the pixel
        size of the image has been set.

        x:
        The resolution point on the frequency axis

        y:
        The y coordinate of the cross section of the resolution and two-sigma curves

        crit:
        the selected resolution criterion.
        """
        if self.resolution is None:
            return self.calculate_resolution()
        else:
            return self.resolution

    def get_all(self):
        """
        Get all the curves in a single dictionary.
        :return: The individual curves resolution, twosigma and resolution packed into
                 a single structure.
        """
        if self.all is None:
            if self.resolution is None:
                self.calculate_resolution()
            self.all = {'resolution': self.frc, 'twosigma': self.two_sigma, 'resolution': self.resolution}

        return self.all

    def calculate_frc(self):
        """
        Calculate the FRC
        :return: Returns the FRC results. They are also saved inside the class.
                 The return value is just for convenience.
        """
        width_ring = self.args.width_ring
        radii = np.arange(0, self.freq_nyq, width_ring)
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for idx, radius in enumerate(radii):
            ind_ring = self.find_points_interval(self.map_dist, radius,
                                                 radius + width_ring)

            #subset1 = self.fft_image1[ind_ring[:, 0], ind_ring[:, 1]]
            #subset2 = self.fft_image2[ind_ring[:, 0], ind_ring[:, 1]]

            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[idx] = np.sum(subset1 * np.conjugate(subset2))
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        self.spatial_freq = radii.astype(np.float32) / self.freq_nyq
        self.n_points = np.array(points)

        # Calculate FRC
        frc = np.abs(c1) / np.sqrt(c2 * c3)

        # Calculate least-squares fit
        poldeg = self.args.polynomial_degree
        coeff = np.polyfit(self.spatial_freq, frc, poldeg)
        equation = np.poly1d(coeff)

        self.frc = {'y': frc, 'x': self.spatial_freq, 'fit': equation(self.spatial_freq), 'eq': equation}
        return self.frc

    def calculate_two_sigma(self):
        """
        Calculate the two sigma curve. The FRC should be run first, as the results of the two sigma
        depend on the number of points on the fourier rings.

        :return:  Returns the Two sigma results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        if self.frc is None:
            self.calculate_frc()

        crit = self.args.resolution_criterion
        n = self.n_points
        poldeg = self.args.polynomial_degree

        if crit == 'one-bit':
            points = (0.5 + 2.4142 / np.sqrt(n)) / (1.5 + 1.4142 / np.sqrt(n))
        elif crit == 'half-bit':
            points = (0.2071 + 1.9102 / np.sqrt(n)) / (1.2071 + 0.9102 / np.sqrt(n))
        elif crit == 'half-height':
            points = 0.5 * np.ones(len(n))
        else:
            raise AttributeError()

        if crit == 'one-bit' or crit == 'half-bit':
            coeff = np.polyfit(self.spatial_freq, points, poldeg)
            equation = np.poly1d(coeff)
            curve = equation(points)
        else:
            equation = None
            curve = points

        self.two_sigma = {'y': points, 'x': self.spatial_freq, 'fit': curve, 'eq': equation, 'crit': crit}
        return self.two_sigma

    def calculate_resolution(self, eps=1e-1):
        """
        Calculate the spatial resolution as a cross-section of the FRC and Two-sigma curves.

        :return: Returns the calculation results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        if self.frc is None:
            self.calculate_frc()

        if self.two_sigma is None:
            self.calculate_two_sigma()

        frc_eq = self.frc['eq']
        two_sigma_eq = self.two_sigma['eq']

        def pdiff1(x):
            return frc_eq(x) - two_sigma_eq(x)

        def pdiff2(x):
            return frc_eq(x) - 0.5

        crit = self.args.resolution_criterion

        if crit == 'one-bit' or crit == 'half-bit':
            for x0 in self.spatial_freq:
                root, infodict, ier, mesg = optimize.fsolve(pdiff1, x0,
                                                            full_output=True)
                if (ier == 1) and (0 < root < self.spatial_freq[-1]):
                    root = root[0]
                    break

            if np.abs(frc_eq(root) - two_sigma_eq(root)) > eps:
                success = 0
            else:
                success = 1

        else:
            if pdiff2(0.0) * pdiff2(s) < 0:
                root = optimize.bisect(pdiff2, 0.0, self.spatial_freq[-1])
                success = 1
            else:
                success = 0

        if success:
            self.resolution = {'resolution': 2 * self.pixel_size / root,
                               'x': root, 'y': frc_eq(root), 'crit': crit}
            return self.resolution
        else:
            print "Could not find an intersection for the curves."
            self.resolution = None
            return {'resolution': 0, 'x': 0, 'y': 0, 'crit': crit}

    @staticmethod
    def find_points_interval(distance_map, start, stop):
        """
        Find the indexes of points located within a fourier ring

        :param distance_map: Is a numpy.array consisting of distances of pixels
                             from the 0-frequency center in the centered 2D FFT
                             image
        :param start:        the ring starts at a radius r = start from the center
        :param stop:         the ring stops at a radious r = stop from the center
        :return:             returns a mask to select the indexes within the
                             specified interval.
        """
        arr_inf = distance_map >= start
        arr_sup = distance_map < stop
        ind = np.where(arr_inf * arr_sup)
        return ind


class SoloFRC(FRC):
    def __init__(self, image, args):
        assert isinstance(image, Image)

        if image.get_dimensions()[0] == 1 and image.ndim() == 3:
            data = image.get_array()[0]
        else:
            data = image.get_array()

        shape = data.shape

        odd_index = list(np.arange(1, shape[i], 2) for i in range(len(shape)))
        even_index = list(np.arange(0, shape[i], 2) for i in range(len(shape)))

        #odd_index = np.arange(1, data.shape[0], 2)
        #even_index = np.arange(0, data.shape[0], 2)

        # image1 = Image(data[block_size:2*block_size, block_size:2*block_size], image.get_spacing())
        # image2 = Image(data[block_size-10:2*block_size-10, block_size-10:2*block_size-10], image.get_spacing())

        if len(shape) == 2:
            image1 = Image(data[odd_index[0], :][:, odd_index[1]], image.get_spacing()[-2:])
            image2 = Image(data[even_index[0], :][:, even_index[1]], image.get_spacing()[-2:])
        elif len(shape) == 3:
            image1 = Image(data[odd_index[0], :, :][:, odd_index[1], :][:, :, odd_index[2]], image.get_spacing())
            image2 = Image(data[even_index[0], :, :][:, even_index[1], :][:, :, even_index[2]], image.get_spacing())
        else:
            raise AttributeError("The image dimensions (%i) are wrong" % len(shape))

        FRC.__init__(self, image1, image2, args)

        if args.normalize_power:
            mean_im1 = np.mean(image1[:])
            mean_im2 = np.mean(image2[:])
            self.fft_image1 /= (np.array(image1.ndim()).sum() * mean_im1)
            self.fft_image2 /= (np.array(image2.ndim()).sum() * mean_im2)

    def calculate_frc_simple(self):

        width_ring = self.args.width_ring
        radii = np.arange(0, self.freq_nyq, width_ring, dtype=np.float32)
        sum1 = np.zeros(len(radii), dtype=np.float32)
        sum2 = np.zeros(len(radii), dtype=np.float32)

        for idx, radius in enumerate(radii):
            ind_ring = self.find_points_interval(self.map_dist, radius, radius + width_ring)
            sum1[idx] = np.absolute(self.fft_image1[ind_ring]).sum()
            sum2[idx] = np.absolute(self.fft_image2[ind_ring]).sum()

        nominator = sum1 * sum2
        denominator = np.sqrt(np.sum(sum1 ** 2) + np.sum(sum2 ** 2))

        self.FRC = nominator / denominator
        self.spatial_freq = radii / self.freq_nyq


def plot_frc_curves(curve_list, mode='resolution', title="FRC curve"):
    """
    Creates a plot of the FRC curves in the curve_list. Single or multiple vurves can
    be plotted.
    """
    ##  Define figure enviroment
    fig = plt.figure(1)
    rect = fig.patch
    rect.set_facecolor('white')
    axescolor = '#f6f6f6'
    ax = plt.subplot(111, facecolor=axescolor)

    ##  Font setting
    font0 = FontProperties()
    font1 = font0.copy()
    font1.set_size('medium')
    font = font1.copy()
    font.set_family('sans')
    rc('text', usetex=True)

    ##  Enable grid
    gridLineWidth = 0.2
    ax.yaxis.grid(True, linewidth=gridLineWidth, linestyle='-', color='0.05')

    ##  Marker setup
    colorArray = ['blue', 'green', 'red', 'orange', 'brown', 'black', 'violet', 'pink']
    marker_array = ['^', 's', 'o', 'd', '1', 'v', '*', 'p']

    ##  Axis limits
    plt.ylim([0, 1.2])

    ##  Axis labelling
    xlabel = 'Spatial frequencies'
    ylabel = 'Fourier Ring Correlation (FRC)'
    plt.xlabel(xlabel, fontsize=12, position=(0.5, -0.2))
    plt.ylabel(ylabel, fontsize=12, position=(0.5, 0.5))

    ##  Title
    plt.text(0.5, 1.06, title, horizontalalignment='center',
             fontsize=18, transform=ax.transAxes)

    ##  1) Modality single: only 1 FRC curve
    if mode == 'resolution':
        y = curve_list['resolution']['y']
        x = curve_list['resolution']['x']
        plt.plot(x, y, marker_array[0], markersize=6, color=colorArray[0], label='FRC')


    # 2) Modality resol: FRC curve + fit of the curve + criterion curve +
    # esol. point
    elif mode == 'fit':
        y = curve_list['resolution']['y']
        x = curve_list['resolution']['x']
        plt.plot(x, y, marker_array[0], markersize=6, color=colorArray[0],
                 label='FRC')

        #plt.hold('True')

        y = curve_list['resolution']['fit']
        plt.plot(x, y, linewidth=3, color=colorArray[1],
                 label='Least-squares fit')

        #plt.hold('True')

        y = curve_list['twosigma']['y']
        res_crit = curve_list['twosigma']['crit']
        if res_crit == 'one-bit':
            label = 'One-bit curve'
        elif res_crit == 'half-bit':
            label = 'Half-bit curve'
        elif res_crit == 'half-height':
            label = 'y = 0.5'
        else:
            raise AttributeError()
        plt.plot(x, y, marker_array[3], color=colorArray[3], markersize=7,
                 label=label)

        #plt.hold('True')

        # Plot resolution point
        y0 = curve_list['resolution']['y']
        x0 = curve_list['resolution']['x']

        plt.plot(x0, y0, 'ro', markersize=8, label='Resolution point')

        verts = [(x0, 0), (x0, y0)]
        xs, ys = zip(*verts)

        ax.plot(xs, ys, 'x--', lw=5, color='red', ms=10)
        ax.text(x0, y0 + 0.10, 'RESOL-FREQ', fontsize=12)


        #         ##  3) Modality multi:
        # elif mode == 'multi':
        #     num_curves = len(curve_list)
        #     for i in range(num_curves):
        #         y = curve_list[i]
        #         if labels is not None:
        #             plt.plot(x, y, color=colorArray[i], label=labels[i],
        #                      linewidth=5)
        #         else:
        #             plt.plot(x, y, color=colorArray[i], linewidth=5)
        #         plt.hold('True')
        #


        ##  Add legend
    plt.legend(loc='best')

    plt.show()
################################################################
################################################################
####                                                        ####
####                      WRITE LOG FILE                    ####
####                                                        ####
################################################################
################################################################

def write_log_file(resol, args, pathout, prefix, image_name):
    print('pathout:\n', pathout)
    print('prefix:\n', prefix)
    file_log = prefix + 'frc_log.txt'
    fp = open(pathout + file_log, 'w')
    fp.write('FRC analysis log file')
    today = datetime.datetime.today()
    fp.write('\n\nCaculation done on the ' + str(today))
    fp.write('\n\nInput images:\n1) ' + image_name[0] + '\n2) ' + image_name[1])
    if args.resol_square is True:
        fp.write('\nComputation done inside the resolution circle')
    if args.hanning is True:
        fp.write('\nHanning pre-filter activated')
    fp.write('\nRing thickness: ' + str(args.width_ring))

    fp.write('\n\n\nResolution results:')
    fp.write('\n1) Criterion one-bit: ' + str(resol[0]) + ' pixels')
    fp.write('\n2) Criterion half-bit: ' + str(resol[1]) + ' pixels')
    fp.write('\n3) Criterion half-height: ' + str(resol[2]) + ' pixels')

    fp.close()
