from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.psf_utils import (
    compute_virtual_psfs,
    prepare_psf,
    prepare_psfs,
)


def _gaussian_psf_img(shape, sigma=2.0, spacing=(1.0, 1.0)):
    coords = [np.arange(s) - s // 2 for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    r_sq = sum(g**2 for g in grid)
    data = np.exp(-r_sq / (2 * sigma**2)).astype(np.float64)
    return Image(data, spacing=list(spacing))


def test_prepare_psf_normalizes():
    psf_img = _gaussian_psf_img((16, 16))
    psf_norm, adj = prepare_psf(psf_img, psf_img.spacing)

    assert pytest.approx(psf_norm.sum()) == 1.0


def test_prepare_psf_adjoint_is_mirror():
    psf_img = _gaussian_psf_img((16, 16))
    psf_norm, adj = prepare_psf(psf_img, psf_img.spacing)

    expected_adj = psf_img.view(np.ndarray)[::-1, ::-1] / psf_img.sum()
    npt.assert_allclose(adj, expected_adj, atol=1e-10)


def test_prepare_psf_resamples():
    psf_img = _gaussian_psf_img((32, 32))
    target_spacing = (2.0, 2.0)
    psf_norm, adj = prepare_psf(psf_img, target_spacing)

    assert psf_norm.shape == (16, 16)
    assert pytest.approx(psf_norm.sum()) == 1.0


def test_prepare_psfs_multiple():
    psf1 = _gaussian_psf_img((16, 16))
    psf2 = _gaussian_psf_img((16, 16), sigma=3.0)
    norms, adjs = prepare_psfs([psf1, psf2], psf1.spacing)

    assert len(norms) == 2
    assert len(adjs) == 2
    assert pytest.approx(norms[0].sum()) == 1.0
    assert pytest.approx(norms[1].sum()) == 1.0


def test_compute_virtual_psfs_normalizes():
    psf1 = _gaussian_psf_img((16, 16)).view(np.ndarray)
    psf2 = _gaussian_psf_img((16, 16), sigma=3.0).view(np.ndarray)

    psf1 /= psf1.sum()
    psf2 /= psf2.sum()
    adj1 = psf1[::-1, ::-1].copy()
    adj2 = psf2[::-1, ::-1].copy()

    virtual = compute_virtual_psfs([psf1, psf2], [adj1, adj2])

    for vp in virtual:
        assert pytest.approx(vp.sum()) == 1.0


def test_compute_virtual_psfs_single_view_raises():
    psf = _gaussian_psf_img((8, 8)).view(np.ndarray)
    psf /= psf.sum()
    with pytest.raises(ValueError, match="at least 2 views"):
        compute_virtual_psfs([psf], [psf])


def test_compute_virtual_psfs_length_mismatch():
    psf1 = _gaussian_psf_img((8, 8)).view(np.ndarray)
    psf2 = _gaussian_psf_img((8, 8)).view(np.ndarray)
    psf1 /= psf1.sum()
    psf2 /= psf2.sum()
    with pytest.raises(ValueError, match="same length"):
        compute_virtual_psfs([psf1, psf2], [psf1])
