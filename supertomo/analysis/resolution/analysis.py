import numpy as np
import scipy.optimize as optimize
from supertomo.data.containers.fourier_correlation_data import FourierCorrelationData, FourierCorrelationDataCollection
import supertomo.processing.ndarray as arrayutils
import scipy.ndimage as ndimage


class FourierCorrelationAnalysis(object):
    def __init__(self, data, args):

        assert isinstance(data, FourierCorrelationDataCollection)

        self.data_collection = data
        self.args = args
        self.data_set = None

    def __fit_least_squares(self):
        """
        Calculate a least squares curve fit to the FRC Data
        :param frc: A FourierCorrelationData structure
        :param degree: The degree of the polynomial function
        :return: None. Will modify the frc argument in place
        """
        # Calculate least-squares fit

        degree = self.args.frc_curve_fit_degree
        if self.args.min_filter:
            data = ndimage.minimum_filter1d(self.data_set.correlation["correlation"], 3)
        else:
            data = self.data_set.correlation["correlation"]

        if data[-1] > data[-2]:
            data[-1] = data[-2]

        coeff = np.polyfit(self.data_set.correlation["frequency"],
                           data,
                           degree,
                           w=1-self.data_set.correlation["frequency"]**2)
        equation = np.poly1d(coeff)

        self.data_set.correlation["curve-fit"] = equation(self.data_set.correlation["frequency"])
        self.data_set.correlation["curve-fit-coefficients"] = coeff

    def __calculate_resolution_threshold(self):
        """
        Calculate the two sigma curve. The FRC should be run first, as the results of the two sigma
        depend on the number of points on the fourier rings.

        :return:  Adds the
        """
        criterion = self.args.resolution_threshold_criterion
        threshold = self.args.resolution_threshold_value
        degree = self.args.resolution_threshold_curve_fit_degree

        if criterion == 'one-bit':
            nominator = 0.5 + arrayutils.safe_divide(
                2.4142,
                np.sqrt(self.data_set.correlation["points-x-bin"])
            )
            denominator = 1.5 + arrayutils.safe_divide(
                1.4142,
                np.sqrt(self.data_set.correlation["points-x-bin"])
            )
            points = arrayutils.safe_divide(nominator, denominator)

        elif criterion == 'half-bit':
            nominator = 0.2071 + arrayutils.safe_divide(
                1.9102,
                np.sqrt(self.data_set.correlation["points-x-bin"])
            )
            denominator = 1.2071 + arrayutils.safe_divide(
                0.9102,
                np.sqrt(self.data_set.correlation["points-x-bin"])
            )
            points = arrayutils.safe_divide(nominator, denominator)

        elif criterion == 'fixed':
            points = threshold * np.ones(len(self.data_set.correlation["points-x-bin"]))
        else:
            raise AttributeError()

        if criterion == 'one-bit' or criterion == 'half-bit':
            coeff = np.polyfit(self.data_set.correlation["frequency"], points, degree)
            equation = np.poly1d(coeff)
            curve = equation(points)
        else:
            coeff = None
            curve = points

        self.data_set.resolution["threshold"] = curve
        self.data_set.resolution["resolution-threshold-coefficients"] = coeff
        self.data_set.resolution["criterion"] = criterion

    def calculate_resolution(self, pixel_size):
        """
        Calculate the spatial resolution as a cross-section of the FRC and Two-sigma curves.

        :return: Returns the calculation results. They are also saved inside the class.
                 The return value is just for convenience.
        """

        criterion = self.args.resolution_threshold_criterion
        threshold = self.args.resolution_threshold_value
        tolerance = self.args.resolution_point_sigma

        def pdiff1(x):
            return frc_eq(x) - two_sigma_eq(x)

        def pdiff2(x):
            return frc_eq(x) - threshold

        for key, value in self.data_collection:
            self.data_set = value

            self.__fit_least_squares()
            self.__calculate_resolution_threshold()

            frc_eq = np.poly1d(self.data_set.correlation["curve-fit-coefficients"])
            two_sigma_eq = np.poly1d(self.data_set.resolution["resolution-threshold-coefficients"])

            # Find intersection
            root, result = optimize.brentq(
                pdiff2 if criterion == 'fixed' else pdiff1,
                0.0, 0.9, xtol=tolerance, full_output=True)

            # Save result, if intersection was found
            if result.converged is True:
                self.data_set.resolution["resolution-point"] = (frc_eq(root), root)
                self.data_set.resolution["criterion"] = criterion
                resolution = 2 * pixel_size / root
                self.data_set.resolution["resolution"] = resolution
                self.data_collection[int(key)] = self.data_set
            else:
                print "Could not find an intersection for the curves for the dataset %s." % key

        return self.data_collection
