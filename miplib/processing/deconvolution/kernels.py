from __future__ import annotations

from typing import Callable

import numpy as np

from miplib.processing.deconvolution.backends import convolve_cpu
from miplib.processing.ndarray import nroot, safe_divide
from miplib.processing.ops_ext import div_unit_grad, update_estimate_poisson


def rl_single_view(
    estimate: np.ndarray,
    image: np.ndarray,
    psf: np.ndarray,
    adj_psf: np.ndarray,
    *,
    convolve: Callable[[np.ndarray, np.ndarray], np.ndarray] = convolve_cpu,
    tv_lambda: float = 0.0,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, float, float, float, float]:
    """One Richardson-Lucy iteration for a single view.

    Parameters
    ----------
    estimate : ndarray
        Current estimate (modified in-place).
    image : ndarray
        Observed image.
    psf : ndarray
        Point-spread function (normalized, sum=1).
    adj_psf : ndarray
        Adjoint (mirrored) PSF.
    convolve : callable
        Convolution backend, e.g. ``convolve_cpu``.
    tv_lambda : float
        Total-variation regularization weight.
    epsilon : float
        Convergence parameter for photon classification.

    Returns
    -------
    estimate : ndarray
        Updated estimate.
    e, s, u, n : float
        Exact, stable, unstable, negative photon counts.
    """
    forward = convolve(estimate, psf)
    ratio = safe_divide(image, forward)
    update = convolve(ratio, adj_psf)

    if tv_lambda > 0:
        update += tv_lambda * div_unit_grad(estimate, (1.0,) * estimate.ndim)

    new_est = estimate.copy()
    e, s, u, n = update_estimate_poisson(new_est, update, epsilon)
    return new_est, e, s, u, n


def rl_multi_view(
    estimate: np.ndarray,
    images: list[np.ndarray],
    psfs: list[np.ndarray],
    adj_psfs: list[np.ndarray],
    *,
    weights: list[float] | None = None,
    backgrounds: list[float] | None = None,
    fusion_mode: str = "summative",
    convolve: Callable[[np.ndarray, np.ndarray], np.ndarray] = convolve_cpu,
    tv_lambda: float = 0.0,
    epsilon: float = 1e-7,
) -> tuple[np.ndarray, float, float, float, float]:
    """One Richardson-Lucy iteration for multi-view fusion.

    All views share the same estimate and update it jointly.

    Parameters
    ----------
    estimate : ndarray
        Current estimate (modified in-place).
    images : list of ndarray
        Observed views.
    psfs : list of ndarray
        Per-view PSFs (normalized, sum=1).
    adj_psfs : list of ndarray
        Per-view adjoint (mirrored) PSFs.
    weights : list of float, optional
        Per-view intensity weights (default: all ones).
    backgrounds : list of float, optional
        Per-view constant background offsets (default: all zeros).
    fusion_mode : {"summative", "multiplicative"}
        View combination strategy.
    convolve : callable
        Convolution backend.
    tv_lambda : float
        Total-variation regularization weight.
    epsilon : float
        Convergence parameter.

    Returns
    -------
    estimate : ndarray
        Updated estimate.
    e, s, u, n : float
        Exact, stable, unstable, negative photon counts.
    """
    n_views = len(images)
    if not (len(psfs) == len(adj_psfs) == n_views):
        raise ValueError("images, psfs, and adj_psfs must have the same length")

    if weights is None:
        weights = [1.0] * n_views
    if backgrounds is None:
        backgrounds = [0.0] * n_views

    if fusion_mode == "summative":
        correction = np.zeros_like(estimate)
        for img, psf, adj, w, bg in zip(images, psfs, adj_psfs, weights, backgrounds):
            forward = convolve(estimate, psf)
            forward = forward * w + bg
            ratio = safe_divide(img, forward)
            correction += convolve(ratio, adj)
        correction /= n_views
    elif fusion_mode == "multiplicative":
        correction = np.ones_like(estimate)
        for img, psf, adj, w, bg in zip(images, psfs, adj_psfs, weights, backgrounds):
            forward = convolve(estimate, psf)
            forward = forward * w + bg
            ratio = safe_divide(img, forward)
            correction *= convolve(ratio, adj)
        correction = np.asarray(nroot(correction, n_views))
    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode!r}")

    if tv_lambda > 0:
        correction += tv_lambda * div_unit_grad(estimate, (1.0,) * estimate.ndim)

    new_est = estimate.copy()
    e, s, u, n = update_estimate_poisson(new_est, correction, epsilon)
    return new_est, e, s, u, n
