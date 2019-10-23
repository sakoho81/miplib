import numpy as np
import scipy.ndimage as ndimage
import scipy.optimize as optimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import medfilt, savgol_filter
import miplib.processing.ndarray as arrayutils
from miplib.data.containers.fourier_correlation_data import FourierCorrelationDataCollection, FourierCorrelationData
import miplib.processing.converters as converters


def fit_frc_curve(data_set, degree, fit_type='spline'):
    """
    Calculate a least squares curve fit to the FRC Data
    :return: None. Will modify the frc argument in place
    """
    assert isinstance(data_set, FourierCorrelationData)

    data = data_set.correlation["correlation"]

    if fit_type == 'smooth-spline':
        equation = UnivariateSpline(data_set.correlation["frequency"],
                                    data)
        equation.set_smoothing_factor(0.25)
        # equation = interp1d(data_set.correlation["frequency"],
        #                     data, kind='slinear')

    elif fit_type == 'spline':
        equation = interp1d(data_set.correlation["frequency"],
                            data, kind='slinear')

    elif fit_type == 'polynomial':

        coeff = np.polyfit(data_set.correlation["frequency"],
                           data,
                           degree,
                           w=1 - data_set.correlation["frequency"] ** 3)
        equation = np.poly1d(coeff)
    else:
        raise AttributeError(fit_type)

    data_set.correlation["curve-fit"] = equation(data_set.correlation["frequency"])

    return equation


def calculate_snr_threshold_value(points_x_bin, snr):
    """
    A function to calculate a SNR based resolution threshold, as described
    in ...

    :param points_x_bin: a 1D Array containing the numbers of points at each
    FRC/FSC ring/shell
    :param snr: the expected SNR value
    :return:
    """
    nominator = snr + arrayutils.safe_divide(
            2.0 * np.sqrt(snr) + 1,
            np.sqrt(points_x_bin)
        )
    denominator = snr + 1 + arrayutils.safe_divide(
        2.0 * np.sqrt(snr),
        np.sqrt(points_x_bin)
    )
    return arrayutils.safe_divide(nominator, denominator)



def calculate_resolution_threshold_curve(data_set, criterion, threshold, snr):
    """
    Calculate the two sigma curve. The FRC should be run first, as the results of the two sigma
    depend on the number of points on the fourier rings.

    :return:  Adds the
    """
    assert isinstance(data_set, FourierCorrelationData)

    points_x_bin = data_set.correlation["points-x-bin"]

    if points_x_bin[-1] == 0:
        points_x_bin[-1] = points_x_bin[-2]

    if criterion == 'one-bit':
        nominator = 0.5 + arrayutils.safe_divide(
            2.4142,
            np.sqrt(points_x_bin)
        )
        denominator = 1.5 + arrayutils.safe_divide(
            1.4142,
            np.sqrt(points_x_bin)
        )
        points = arrayutils.safe_divide(nominator, denominator)

    elif criterion == 'half-bit':
        nominator = 0.2071 + arrayutils.safe_divide(
            1.9102,
            np.sqrt(points_x_bin)
        )
        denominator = 1.2071 + arrayutils.safe_divide(
            0.9102,
            np.sqrt(points_x_bin)
        )
        points = arrayutils.safe_divide(nominator, denominator)

    elif criterion == 'three-sigma':
        points = arrayutils.safe_divide(np.full(points_x_bin.shape, 3.0), (np.sqrt(points_x_bin) + 3.0 - 1))


    elif criterion == 'fixed':
        points = threshold * np.ones(len(data_set.correlation["points-x-bin"]))
    elif criterion == 'snr':
        points = calculate_snr_threshold_value(points_x_bin, snr)

    else:
        raise AttributeError()

    if criterion != 'fixed':
        #coeff = np.polyfit(data_set.correlation["frequency"], points, 3)
        #equation = np.poly1d(coeff)
        equation = interp1d(data_set.correlation["frequency"], points, kind='slinear')
        curve = equation(data_set.correlation["frequency"])
    else:
        curve = points
        equation = None

    data_set.resolution["threshold"] = curve
    return equation


class FourierCorrelationAnalysis(object):
    def __init__(self, data, spacing, args):

        assert isinstance(data, FourierCorrelationDataCollection)

        self.data_collection = data
        self.args = args
        self.spacing = spacing

    def execute(self, z_correction=1):
        """
        Calculate the spatial resolution as a cross-section of the FRC and Two-sigma curves.

        :return: Returns the calculation results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        criterion = self.args.resolution_threshold_criterion
        threshold = self.args.resolution_threshold_value
        snr = self.args.resolution_snr_value
        tolerance = self.args.resolution_point_sigma
        degree = self.args.frc_curve_fit_degree
        fit_type = self.args.frc_curve_fit_type
        verbose = self.args.verbose

        def pdiff1(x):
            return abs(frc_eq(x) - two_sigma_eq(x))

        def pdiff2(x):
            return abs(frc_eq(x) - threshold)

        def first_guess(x, y, threshold):
            #y_smooth = savgol_filter(y, 5, 2)
            #return x[np.argmin(np.abs(y_smooth - threshold))]

            difference = y - threshold

            return x[np.where(difference <= 0)[0][0] - 1]
            #return x[np.argmin(np.abs(y - threshold))]

        for key, data_set in self.data_collection:

            if verbose:
                print("Calculating resolution point for dataset {}".format(key))
            frc_eq = fit_frc_curve(data_set, degree, fit_type)
            two_sigma_eq = calculate_resolution_threshold_curve(data_set, criterion, threshold, snr)

            """
            Todo: Make the first quess adaptive. For example find the data point at which FRC
            value is closest to the mean of the threshold
            """


            # Find intersection
            fit_start = first_guess(data_set.correlation["frequency"],
                                    data_set.correlation["correlation"],
                                    np.mean(data_set.resolution["threshold"])
            )
            if self.args.verbose:
                print("Fit starts at {}".format(fit_start))
                disp = 1
            else:
                disp = 0
            root = optimize.fmin(pdiff2 if criterion == 'fixed' else pdiff1, fit_start, disp=disp)[0]
            data_set.resolution["resolution-point"] = (frc_eq(root), root)
            data_set.resolution["criterion"] = criterion

            angle = converters.degrees_to_radians(int(key))
            z_correction_multiplier = (1+(z_correction-1)*np.abs(np.sin(angle)))
            resolution =z_correction_multiplier*(2*self.spacing / root)

            data_set.resolution["resolution"] = resolution
            data_set.resolution["spacing"] = self.spacing*z_correction_multiplier

            self.data_collection[int(key)] = data_set


            # # # Find intersection
            # root, result = optimize.brentq(
            #     pdiff2 if criterion == 'fixed' else pdiff1,
            #     0.0, 1.0, xtol=tolerance, full_output=True)
            #
            # # Save result, if intersection was found
            # if result.converged is True:
            #     data_set.resolution["resolution-point"] = (frc_eq(root), root)
            #     data_set.resolution["criterion"] = criterion
            #     resolution = 2 * self.spacing / root
            #     data_set.resolution["resolution"] = resolution
            #     self.data_collection[int(key)] = data_set
            # else:
            #     print "Could not find an intersection for the curves for the dataset %s." % key

        return self.data_collection
