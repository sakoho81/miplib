from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import miplib.data.iterators.fourier_ring_iterators as iterators
import miplib.data.iterators.fourier_shell_iterators as shell_iterators
import miplib.processing.ndarray as arrayutils
from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
)

from . import analysis as fsc_analysis


def _cutoff_correction(x: float) -> float:
    return 0.95988146 * np.exp(13.90441896 * (x - 0.97979108)) + 0.55146136


@dataclass
class FRCOptions:
    d_bin: float = 1.0
    use_hamming: bool = True
    d_angle: float = 45.0
    d_extract_angle: float = 5.0
    frc_curve_fit_degree: int = 8
    frc_curve_fit_type: fsc_analysis.FitType = fsc_analysis.FitType.SPLINE
    resolution_threshold_criterion: fsc_analysis.ResolutionCriterion = (
        fsc_analysis.ResolutionCriterion.FIXED
    )
    resolution_threshold_value: float = 1.0 / 7
    resolution_snr_value: float = 0.25


def namespace_to_frc_options(ns: object) -> FRCOptions:
    fit_type_raw = getattr(ns, "frc_curve_fit_type", "spline")
    if isinstance(fit_type_raw, fsc_analysis.FitType):
        fit_type = fit_type_raw
    else:
        fit_type = fsc_analysis.FitType(fit_type_raw)

    criterion_raw = getattr(ns, "resolution_threshold_criterion", "fixed")
    if isinstance(criterion_raw, fsc_analysis.ResolutionCriterion):
        criterion = criterion_raw
    else:
        criterion = fsc_analysis.ResolutionCriterion(criterion_raw)

    return FRCOptions(
        d_bin=getattr(ns, "d_bin", 1.0),
        use_hamming=getattr(ns, "use_hamming", True),
        d_angle=getattr(ns, "d_angle", 45.0),
        d_extract_angle=getattr(ns, "d_extract_angle", 5.0),
        frc_curve_fit_degree=getattr(ns, "frc_curve_fit_degree", 8),
        frc_curve_fit_type=fit_type,
        resolution_threshold_criterion=criterion,
        resolution_threshold_value=getattr(ns, "resolution_threshold_value", 1.0 / 7),
        resolution_snr_value=getattr(ns, "resolution_snr_value", 0.25),
    )


def create_fourier_iterator(
    shape: tuple[int, ...], d_bin: float = 1.0
) -> iterators.FourierRingIterator | shell_iterators.FourierShellIterator:
    if len(shape) == 2:
        return iterators.FourierRingIterator(shape, d_bin)
    if len(shape) == 3:
        return shell_iterators.FourierShellIterator(shape, d_bin)
    raise ValueError(f"Unsupported dimensionality: {len(shape)}D")


def accumulate_fourier_correlation(
    fft1: np.ndarray,
    fft2: np.ndarray,
    iterator: iterators.FourierRingIterator | shell_iterators.FourierShellIterator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii = iterator.radii
    c1 = np.zeros(radii.shape, dtype=np.float32)
    c2 = np.zeros(radii.shape, dtype=np.float32)
    c3 = np.zeros(radii.shape, dtype=np.float32)
    n_points = np.zeros(radii.shape, dtype=np.float32)

    for indices, idx in iterator:
        subset1 = fft1[indices]
        subset2 = fft2[indices]
        c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
        c2[idx] = np.sum(np.abs(subset1) ** 2)
        c3[idx] = np.sum(np.abs(subset2) ** 2)
        n_points[idx] = len(subset1)

    return c1, c2, c3, n_points


def accumulate_sectioned_fourier_correlation(
    fft1: np.ndarray,
    fft2: np.ndarray,
    iterator: shell_iterators.SectionedFourierShellIterator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radii, angles = iterator.steps
    shape = (angles.shape[0], radii.shape[0])
    c1 = np.zeros(shape, dtype=np.float32)
    c2 = np.zeros(shape, dtype=np.float32)
    c3 = np.zeros(shape, dtype=np.float32)
    n_points = np.zeros(shape, dtype=np.float32)

    for indices, shell_idx, rotation_idx in iterator:  # type: ignore[misc]
        subset1 = fft1[indices]
        subset2 = fft2[indices]
        c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2)).real
        c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
        c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)
        n_points[rotation_idx, shell_idx] = len(subset1)

    return c1, c2, c3, n_points


def build_correlation_curve(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    radii: np.ndarray,
    nyquist: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_freq = _radii_to_spatial_freq(radii, nyquist)
    frc = arrayutils.safe_divide(np.abs(c1), np.sqrt(c2 * c3))
    return spatial_freq, frc


def make_correlation_data(
    correlation: np.ndarray,
    frequency: np.ndarray,
    points_per_bin: np.ndarray,
) -> FourierCorrelationData:
    data = FourierCorrelationData()
    data.correlation["correlation"] = correlation
    data.correlation["frequency"] = frequency
    data.correlation["points-x-bin"] = points_per_bin
    return data


def _radii_to_spatial_freq(radii: np.ndarray, nyquist: float) -> np.ndarray:
    return radii.astype(np.float32) / nyquist
