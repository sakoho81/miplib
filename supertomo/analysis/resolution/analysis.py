import numpy as np
import scipy.optimize as optimize
from supertomo.data.containers.fourier_correlation import FourierCorrelationData, FourierCorrelationDataCollection
import scipy.signal as signal

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
        data = signal.medfilt(self.data_set.correlation["correlation"], 5)
        coeff = np.polyfit(self.data_set.correlation["frequency"],
                           data,
                           degree,
                           w=np.sqrt(self.data_set.correlation["correlation"]))
        equation = np.poly1d(coeff)

        self.data_set.correlation["curve-fit"] = equation(self.data_set.correlation["frequency"])
        self.data_set.correlation["curve-fit-eq"] = equation

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
            points = (0.5 + 2.4142 / np.sqrt(self.data_set.correlation["points-x-bin"])) / \
                     (1.5 + 1.4142 / np.sqrt(self.data_set.correlation["points-x-bin"]))
        elif criterion == 'half-bit':
            points = (0.2071 + 1.9102 / np.sqrt(self.data_set.correlation["points-x-bin"])) / \
                     (1.2071 + 0.9102 / np.sqrt(self.data_set.correlation["points-x-bin"]))
        elif criterion == 'fixed':
            points = threshold * np.ones(len(self.data_set.correlation["points-x-bin"]))
        else:
            raise AttributeError()

        if criterion == 'one-bit' or criterion == 'half-bit':
            coeff = np.polyfit(self.data_set.correlation["frequency"], points, degree)
            equation = np.poly1d(coeff)
            curve = equation(points)
        else:
            equation = None
            curve = points

        self.data_set.resolution["threshold"] = curve
        self.data_set.resolution["threshold-eq"] = equation
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

            frc_eq = self.data_set.correlation["curve-fit-eq"]
            two_sigma_eq = self.data_set.resolution["threshold-eq"]

            if criterion == 'one-bit' or criterion == 'half-bit':
                for x0 in self.data_set.correlation["frequency"]:
                    root, infodict, ier, mesg = optimize.fsolve(pdiff1, x0,
                                                                full_output=True)
                    if (ier == 1) and (0 < root < self.data_set.correlation["frequency"][-1]):
                        root = root[0]
                        break

                if np.abs(frc_eq(root) - two_sigma_eq(root)) > tolerance:
                    success = False
                else:
                    success = True

            else:
                if pdiff2(0.0) * pdiff2(self.data_set.correlation["frequency"][-1]) < 0:
                    root = optimize.bisect(pdiff2, 0.0, self.data_set.correlation["frequency"][-1])
                    success = True
                else:
                    success = False

            if success:
                self.data_set.resolution["resolution-point"] = (frc_eq(root), root)
                self.data_set.resolution["criterion"] = criterion
                resolution = 2 * pixel_size / root
                self.data_set.resolution["resolution"] = resolution

            else:
                print "Could not find an intersection for the curves for the dataset %s." % key

            self.data_collection[int(key)] = self.data_set

        return self.data_collection
