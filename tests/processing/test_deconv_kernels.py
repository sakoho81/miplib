from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.signal

from miplib.processing import ops_ext
from miplib.processing.deconvolution.kernels import rl_multi_view, rl_single_view


def _gaussian_psf(shape, sigma=2.0):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    psf = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    return psf / psf.sum()


def _blur(image, psf):
    return scipy.signal.fftconvolve(image, psf, mode="same")


@pytest.fixture
def small_image():
    shape = (16, 16)
    obj = np.zeros(shape, dtype=np.float64)
    obj[4:12, 4:12] = np.linspace(0.2, 2.0, 8)[:, None]
    return obj


@pytest.fixture
def small_psf():
    return _gaussian_psf((16, 16), sigma=1.5)


@pytest.fixture
def small_adj_psf(small_psf):
    psf = small_psf
    slices = tuple(slice(None, None, -1) for _ in range(psf.ndim))
    return psf[slices]


# ---------------------------------------------------------------------------
# rl_single_view
# ---------------------------------------------------------------------------


def test_rl_recovers_blurred_object(small_image, small_psf, small_adj_psf):
    """RL iterations on a noise-free, PSF-blurred image should reduce MSE toward zero."""
    blurred = _blur(small_image, small_psf)
    est = np.ones_like(small_image) * small_image.mean()

    mse_before = np.mean((est - small_image) ** 2)

    for _ in range(10):
        est, *_ = rl_single_view(est, blurred, small_psf, small_adj_psf, epsilon=0.1)

    mse_after = np.mean((est - small_image) ** 2)
    assert mse_after < mse_before * 0.5, (
        f"After 10 RL iterations, MSE should decrease substantially: "
        f"{mse_after:.6f} not < {mse_before * 0.5:.6f}"
    )


def test_constant_image_approaches_fixed_point():
    """On a large image with narrow PSF, a constant should be close to a fixed point."""
    shape = (64, 64)
    psf = _gaussian_psf(shape, sigma=1.0)
    adj_psf = psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))]
    img = np.ones(shape, dtype=np.float64)
    est = img.copy()

    est_new, *_ = rl_single_view(est, img, psf, adj_psf, epsilon=0.1)

    interior = tuple(slice(12, -12) for _ in range(img.ndim))
    npt.assert_allclose(est_new[interior], img[interior], rtol=1e-4)


def test_rl_single_view_updates_estimate(small_image, small_psf, small_adj_psf):
    blurred = _blur(small_image, small_psf)
    est = np.ones_like(small_image)

    est_one = est.copy()
    est_one, *_ = rl_single_view(est_one, blurred, small_psf, small_adj_psf)

    assert not np.allclose(est_one, small_image), "One iteration should not be perfect"
    assert est_one.max() > est.max() * 0.5
    assert est_one.min() >= 0


def test_monotonic_kullback_leibler(small_image, small_psf, small_adj_psf):
    blurred = _blur(small_image, small_psf)
    est = np.ones_like(small_image)

    kl_values = []
    for _ in range(5):
        est, *_ = rl_single_view(est, blurred, small_psf, small_adj_psf, epsilon=0.1)
        fwd = _blur(est, small_psf)
        kl = ops_ext.kullback_leibler_divergence(blurred, fwd, 0.1)
        kl_values.append(kl)

    for i in range(len(kl_values) - 1):
        assert kl_values[i] > kl_values[i + 1], (
            f"K-L divergence must decrease monotonically: "
            f"{kl_values[i]:.6f} !> {kl_values[i + 1]:.6f}"
        )


def test_rl_epsilon_validates():
    img = np.ones((8, 8), dtype=np.float64)
    est = img.copy()
    psf = _gaussian_psf((8, 8), sigma=1.0)
    adj = psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))]

    with pytest.raises(ValueError):
        rl_single_view(est, img, psf, adj, epsilon=-0.1)

    with pytest.raises(ValueError):
        rl_single_view(est, img, psf, adj, epsilon=0.6)


# ---------------------------------------------------------------------------
# rl_multi_view
# ---------------------------------------------------------------------------


def test_multi_view_equals_single_view(small_image, small_psf, small_adj_psf):
    blurred = _blur(small_image, small_psf)
    est_single = np.ones_like(small_image)
    est_multi = np.ones_like(small_image)

    for _ in range(3):
        est_single, *_ = rl_single_view(
            est_single, blurred, small_psf, small_adj_psf, epsilon=0.1
        )

    n_views = 3
    images = [blurred] * n_views
    psfs = [small_psf] * n_views
    adjs = [small_adj_psf] * n_views

    for _ in range(3):
        est_multi, *_ = rl_multi_view(
            est_multi, images, psfs, adjs, fusion_mode="summative", epsilon=0.1
        )

    npt.assert_allclose(est_single, est_multi, rtol=1e-6)


def test_multi_view_summative_vs_multiplicative_one_view(
    small_image, small_psf, small_adj_psf
):
    blurred = _blur(small_image, small_psf)
    est_sum = np.ones_like(small_image)
    est_mul = np.ones_like(small_image)

    for _ in range(3):
        est_sum, *_ = rl_multi_view(
            est_sum,
            [blurred],
            [small_psf],
            [small_adj_psf],
            fusion_mode="summative",
            epsilon=0.1,
        )
        est_mul, *_ = rl_multi_view(
            est_mul,
            [blurred],
            [small_psf],
            [small_adj_psf],
            fusion_mode="multiplicative",
            epsilon=0.1,
        )

    npt.assert_allclose(est_sum, est_mul, rtol=1e-6)


def test_multi_view_length_mismatch():
    img = np.ones((8, 8), dtype=np.float64)
    est = img.copy()
    psf = _gaussian_psf((8, 8), sigma=1.0)
    adj = psf[tuple(slice(None, None, -1) for _ in range(psf.ndim))]

    with pytest.raises(ValueError, match="must have the same length"):
        rl_multi_view(est, [img], [psf, psf], [adj])


def test_multi_view_invalid_mode(small_image, small_psf, small_adj_psf):
    with pytest.raises(ValueError, match="Unknown fusion_mode"):
        rl_multi_view(
            small_image,
            [small_image],
            [small_psf],
            [small_adj_psf],
            fusion_mode="invalid",
        )


def test_multi_view_weights_and_backgrounds(small_image, small_psf, small_adj_psf):
    blurred = _blur(small_image, small_psf)
    est_w = np.ones_like(small_image)
    est_now = np.ones_like(small_image)

    n_views = 2
    images = [blurred] * n_views
    psfs = [small_psf] * n_views
    adjs = [small_adj_psf] * n_views

    for _ in range(3):
        est_now, *_ = rl_multi_view(
            est_now, images, psfs, adjs, fusion_mode="summative", epsilon=0.1
        )
        est_w, *_ = rl_multi_view(
            est_w,
            images,
            psfs,
            adjs,
            weights=[0.5, 0.5],
            backgrounds=[0.01, 0.01],
            fusion_mode="summative",
            epsilon=0.1,
        )

    assert not np.allclose(est_w, est_now), "weights/backgrounds should change result"


def test_multi_view_default_weights_and_backgrounds(
    small_image, small_psf, small_adj_psf
):
    blurred = _blur(small_image, small_psf)
    est1 = np.ones_like(small_image)
    est2 = np.ones_like(small_image)

    for _ in range(3):
        est1, *_ = rl_multi_view(
            est1, [blurred], [small_psf], [small_adj_psf], epsilon=0.1
        )
        est2, *_ = rl_multi_view(
            est2,
            [blurred],
            [small_psf],
            [small_adj_psf],
            weights=[1.0],
            backgrounds=[0.0],
            epsilon=0.1,
        )

    npt.assert_allclose(est1, est2, rtol=1e-6)
