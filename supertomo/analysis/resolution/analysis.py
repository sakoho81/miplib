import numpy as np
import scipy.optimize as optimize
from supertomo.data.containers.fourier_correlation import FourierCorrelationData


def fit_least_squares(frc, degree):
    """
    Calculate a least squares curve fit to the FRC Data
    :param frc: A FourierCorrelationData structure
    :param degree: The degree of the polynomial function
    :return: None. Will modify the frc argument in place
    """
    # Calculate least-squares fit

    coeff = np.polyfit(frc.correlation["frequency"], frc.correlation, degree)
    equation = np.poly1d(coeff)

    frc.correlation["curve-fit"] = equation(frc.correlation["frequency"])
    frc.correlation["curve-fit-eq"] = equation

    return frc


def calculate_resolution_threshold(frc, criterion, degree, threshold=0.5):
    """
    Calculate the two sigma curve. The FRC should be run first, as the results of the two sigma
    depend on the number of points on the fourier rings.

    :return:  Adds the
    """
    assert isinstance(frc, FourierCorrelationData)

    if criterion == 'one-bit':
        points = (0.5 + 2.4142 / np.sqrt(frc.correlation["points"])) / (1.5 + 1.4142 / np.sqrt(frc.correlation["points"]))
    elif criterion == 'half-bit':
        points = (0.2071 + 1.9102 / np.sqrt(frc.correlation["points"])) / (1.2071 + 0.9102 / np.sqrt(frc.correlation["points"]))
    elif criterion == 'threshold':
        points = threshold * np.ones(len(frc.correlation["points"]))
    else:
        raise AttributeError()

    if criterion == 'one-bit' or criterion == 'half-bit':
        coeff = np.polyfit(frc.correlation["frequency"], points, degree)
        equation = np.poly1d(coeff)
        curve = equation(points)
    else:
        equation = None
        curve = points

    frc.resolution["threshold"] = curve
    frc.resolution["threshold-eq"] = equation

    return frc


def calculate_resolution(frc, criterion, pixel_size, tolerance=1e-1):
    """
    Calculate the spatial resolution as a cross-section of the FRC and Two-sigma curves.

    :return: Returns the calculation results. They are also saved inside the class.
             The return value is just for convenience.
    """

    assert isinstance(frc, FourierCorrelationData)

    frc_eq = frc.correlation["curve-fit-eq"]
    two_sigma_eq = frc.resolution["threshold-eq"]

    def pdiff1(x):
        return frc_eq(x) - two_sigma_eq(x)

    def pdiff2(x):
        return frc_eq(x) - 0.5

    if criterion == 'one-bit' or criterion == 'half-bit':
        for x0 in frc.correlation["frequency"]:
            root, infodict, ier, mesg = optimize.fsolve(pdiff1, x0,
                                                        full_output=True)
            if (ier == 1) and (0 < root < frc.correlation["frequency"][-1]):
                root = root[0]
                break

        if np.abs(frc_eq(root) - two_sigma_eq(root)) > tolerance:
            success = False
        else:
            success = True

    else:
        if pdiff2(0.0) * pdiff2(frc.correlation["frequency"][-1]) < 0:
            root = optimize.bisect(pdiff2, 0.0, frc.correlation["frequency"][-1])
            success = True
        else:
            success = False

    if success:
        frc.resolution["resolution-point"] = (frc_eq(root), root)
        frc.resolution["criterion"] = criterion
        resolution = 2 * pixel_size / root
        frc.resolution["resolution"] = resolution
        return frc

    else:
        print "Could not find an intersection for the curves."
        return frc
