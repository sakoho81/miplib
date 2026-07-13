from __future__ import annotations

import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.psf_utils import (
    compute_virtual_psfs,
    prepare_psf,
    prepare_psfs,
)
from tests.conftest import gaussian_spot


def test_prepare_psf_normalizes():
    psf_img = Image(gaussian_spot((16, 16), sigma=2.0), spacing=(1.0, 1.0))
    psf_norm, adj = prepare_psf(psf_img, psf_img.spacing)

    assert pytest.approx(psf_norm.sum()) == 1.0


def test_prepare_psf_adjoint_is_mirror():
    psf_img = Image(gaussian_spot((16, 16), sigma=2.0), spacing=(1.0, 1.0))
    psf_norm, adj = prepare_psf(psf_img, psf_img.spacing)

    expected_adj = gaussian_spot((16, 16), sigma=2.0)[::-1, ::-1]
    expected_adj /= expected_adj.sum()
    npt.assert_allclose(adj, expected_adj, atol=1e-12)


def test_prepare_psf_resamples():
    psf_img = Image(gaussian_spot((32, 32), sigma=2.0), spacing=(1.0, 1.0))
    target_spacing = (2.0, 2.0)
    psf_norm, adj = prepare_psf(psf_img, target_spacing)

    assert psf_norm.shape == (16, 16)
    assert pytest.approx(psf_norm.sum()) == 1.0


def test_prepare_psfs_multiple():
    psf1 = Image(gaussian_spot((16, 16), sigma=2.0), spacing=(1.0, 1.0))
    psf2 = Image(gaussian_spot((16, 16), sigma=3.0), spacing=(1.0, 1.0))
    norms, adjs = prepare_psfs([psf1, psf2], psf1.spacing)

    assert len(norms) == 2
    assert len(adjs) == 2
    assert pytest.approx(norms[0].sum()) == 1.0
    assert pytest.approx(norms[1].sum()) == 1.0


def test_compute_virtual_psfs_normalizes():
    psf1 = gaussian_spot((16, 16), sigma=2.0)
    psf2 = gaussian_spot((16, 16), sigma=3.0)

    psf1 /= psf1.sum()
    psf2 /= psf2.sum()
    adj1 = psf1[::-1, ::-1].copy()
    adj2 = psf2[::-1, ::-1].copy()

    virtual = compute_virtual_psfs([psf1, psf2], [adj1, adj2])

    for vp in virtual:
        assert pytest.approx(vp.sum()) == 1.0
        assert vp.min() >= 0


def test_compute_virtual_psfs_single_view_raises():
    psf = gaussian_spot((8, 8), sigma=2.0)
    psf /= psf.sum()
    with pytest.raises(ValueError, match="at least 2 views"):
        compute_virtual_psfs([psf], [psf])


def test_compute_virtual_psfs_length_mismatch():
    psf1 = gaussian_spot((8, 8), sigma=2.0)
    psf2 = gaussian_spot((8, 8), sigma=2.0)
    psf1 /= psf1.sum()
    psf2 /= psf2.sum()
    with pytest.raises(ValueError, match="same length"):
        compute_virtual_psfs([psf1, psf2], [psf1])
