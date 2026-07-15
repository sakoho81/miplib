import logging
from enum import Enum

import numpy as np
import scipy.optimize as optimize
from scipy.interpolate import UnivariateSpline, interp1d

import miplib.processing.converters as converters
import miplib.processing.ndarray as arrayutils
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)

logger = logging.getLogger(__name__)


class FitType(Enum):
    """Curve-fitting method for FRC data."""

    SMOOTH_SPLINE = "smooth-spline"
    SPLINE = "spline"
    POLYNOMIAL = "polynomial"


class ResolutionCriterion(Enum):
    """Threshold criterion for determining the resolution limit.

    ONE_BIT: 1-bit information content criterion.
    HALF_BIT: ½-bit information content criterion.
    THREE_SIGMA: 3σ significance threshold.
    FIXED: Fixed threshold value (e.g. 1/7).
    SNR: SNR-based adaptive threshold.
    """

    ONE_BIT = "one-bit"
    HALF_BIT = "half-bit"
    THREE_SIGMA = "three-sigma"
    FIXED = "fixed"
    SNR = "snr"


def fit_frc_curve(
    data_set: FourierCorrelationData, degree: int, fit_type: FitType = FitType.SPLINE
) -> UnivariateSpline | interp1d | np.poly1d:
    data = data_set.correlation["correlation"]

    if fit_type == FitType.SMOOTH_SPLINE:
        equation = UnivariateSpline(data_set.correlation["frequency"], data)
        equation.set_smoothing_factor(0.25)
    elif fit_type == FitType.SPLINE:
        equation = interp1d(data_set.correlation["frequency"], data, kind="slinear")
    elif fit_type == FitType.POLYNOMIAL:
        coeff = np.polyfit(
            data_set.correlation["frequency"],
            data,
            degree,
            w=1 - data_set.correlation["frequency"] ** 3,
        )
        equation = np.poly1d(coeff)
    else:
        raise ValueError(f"Unknown fit_type: {fit_type!r}")

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
        2.0 * np.sqrt(snr) + 1, np.sqrt(points_x_bin)
    )
    denominator = (
        snr + 1 + arrayutils.safe_divide(2.0 * np.sqrt(snr), np.sqrt(points_x_bin))
    )
    return arrayutils.safe_divide(nominator, denominator)


def calculate_resolution_threshold_curve(
    data_set: FourierCorrelationData,
    criterion: ResolutionCriterion,
    threshold: float,
    snr: float,
) -> interp1d | None:
    points_x_bin = data_set.correlation["points-x-bin"]

    if points_x_bin[-1] == 0:
        points_x_bin[-1] = points_x_bin[-2]

    if criterion == ResolutionCriterion.ONE_BIT:
        nominator = 0.5 + arrayutils.safe_divide(2.4142, np.sqrt(points_x_bin))
        denominator = 1.5 + arrayutils.safe_divide(1.4142, np.sqrt(points_x_bin))
        points = arrayutils.safe_divide(nominator, denominator)
    elif criterion == ResolutionCriterion.HALF_BIT:
        nominator = 0.2071 + arrayutils.safe_divide(1.9102, np.sqrt(points_x_bin))
        denominator = 1.2071 + arrayutils.safe_divide(0.9102, np.sqrt(points_x_bin))
        points = arrayutils.safe_divide(nominator, denominator)
    elif criterion == ResolutionCriterion.THREE_SIGMA:
        points = arrayutils.safe_divide(
            np.full(points_x_bin.shape, 3.0), (np.sqrt(points_x_bin) + 3.0 - 1)
        )
    elif criterion == ResolutionCriterion.FIXED:
        points = threshold * np.ones(len(data_set.correlation["points-x-bin"]))
    elif criterion == ResolutionCriterion.SNR:
        points = calculate_snr_threshold_value(points_x_bin, snr)
    else:
        raise ValueError(f"Unknown criterion: {criterion!r}")

    if criterion != ResolutionCriterion.FIXED:
        equation = interp1d(data_set.correlation["frequency"], points, kind="slinear")
        curve = equation(data_set.correlation["frequency"])
    else:
        curve = points
        equation = None

    data_set.resolution["threshold"] = curve
    return equation


def _first_guess(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Return the frequency just before the FRC curve crosses the threshold.

    Returns None if the curve never crosses the threshold.
    """
    candidates = np.where((y - threshold) <= 0)[0]
    if len(candidates) == 0:
        return None
    idx = max(0, candidates[0] - 1)
    return x[idx]


class FourierCorrelationAnalysis:
    def __init__(
        self,
        data: FourierCorrelationDataCollection,
        spacing: float,
        options: object,
    ) -> None:
        if not isinstance(data, FourierCorrelationDataCollection):
            raise TypeError(
                f"Expected FourierCorrelationDataCollection, got {type(data).__name__}"
            )

        self.data_collection = data
        self.options = options
        self.spacing = spacing

    def execute(self, z_correction: float = 1) -> FourierCorrelationDataCollection:
        criterion = self.options.resolution_threshold_criterion  # type: ignore[attr-defined]
        threshold = self.options.resolution_threshold_value  # type: ignore[attr-defined]
        snr = self.options.resolution_snr_value  # type: ignore[attr-defined]
        degree = self.options.frc_curve_fit_degree  # type: ignore[attr-defined]
        fit_type = self.options.frc_curve_fit_type  # type: ignore[attr-defined]

        def _pdiff1(x: float) -> float:
            return abs(frc_eq(x) - two_sigma_eq(x))

        def _pdiff2(x: float) -> float:
            return abs(frc_eq(x) - threshold)

        for key, data_set in self.data_collection:
            logger.debug("Calculating resolution point for dataset %s", key)
            frc_eq = fit_frc_curve(data_set, degree, fit_type)
            two_sigma_eq = calculate_resolution_threshold_curve(
                data_set, criterion, threshold, snr
            )

            fit_start = _first_guess(
                data_set.correlation["frequency"],
                data_set.correlation["correlation"],
                np.mean(data_set.resolution["threshold"]),
            )
            if fit_start is None:
                logger.debug("No intersection found for dataset %s", key)
                continue

            logger.debug("Fit starts at %s", fit_start)

            root = optimize.minimize_scalar(
                _pdiff2 if criterion == ResolutionCriterion.FIXED else _pdiff1,
                bounds=(0, 1),
                method="bounded",
            ).x
            data_set.resolution["resolution-point"] = (frc_eq(root), root)
            data_set.resolution["criterion"] = criterion.value

            angle = converters.degrees_to_radians(int(key))
            z_correction_multiplier = 1 + (z_correction - 1) * np.abs(np.sin(angle))
            resolution = z_correction_multiplier * (2 * self.spacing / root)

            data_set.resolution["resolution"] = resolution
            data_set.resolution["spacing"] = self.spacing * z_correction_multiplier

            self.data_collection[int(key)] = data_set

        return self.data_collection
