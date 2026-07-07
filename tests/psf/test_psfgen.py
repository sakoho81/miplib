import math

import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.psf.psfgen import PsfFromFwhm


def test_psf_fwhm_to_sigma_conversion():
    """FWHM → sigma conversion: sigma_um = FWHM / (2*sqrt(2*ln(2)))."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128), dims=(4.0, 4.0))
    spacing = 4.0 / 128.0
    expected_sigma_um = 2.0 / (2 * math.sqrt(2 * math.log(2)))
    expected_sigma_px = expected_sigma_um / spacing
    assert psf.sigma_px[0] == pytest.approx(expected_sigma_px)
    assert psf.sigma_px[1] == pytest.approx(expected_sigma_px)


def test_psf_xy_shape():
    """xy() returns 2D mirror-symmetry image: (2*shape[1]-1, 2*shape[1]-1)."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128))
    result = psf.xy()
    assert result.shape == (255, 255)
    assert result.ndim == 2
    assert isinstance(result, Image)


def test_psf_xy_spacing():
    """xy() spacing is isotropic lateral spacing."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128), dims=(4.0, 4.0))
    spacing = 4.0 / 128.0
    result = psf.xy()
    assert result.spacing == pytest.approx([spacing, spacing])


def test_psf_xy_center_peak_normalized():
    """The PSF center is normalized to 1.0."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128))
    result = psf.xy()
    center = result.shape[0] // 2
    assert result[center, center] == pytest.approx(1.0)


def test_psf_xy_gaussian_profile():
    """Value at 1-sigma from center ≈ e^(-0.5) ≈ 0.606."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128), dims=(4.0, 4.0))
    result = psf.xy()
    center = result.shape[0] // 2
    sigma_px = psf.sigma_px[1]  # lateral sigma in pixels
    val = result[center, center + int(sigma_px)]
    npt.assert_approx_equal(val, 0.606, significant=2)


def test_psf_volume_shape():
    """volume() returns 3D symmetrized PSF."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128))
    result = psf.volume()
    assert result.ndim == 3
    assert result.shape == (255, 255, 255)
    assert isinstance(result, Image)


def test_psf_volume_spacing():
    """volume() spacing includes axial + lateral."""
    psf = PsfFromFwhm(fwhm=[2.0, 2.0], shape=(128, 128), dims=(4.0, 4.0))
    spacing = 4.0 / 128.0
    result = psf.volume()
    assert result.spacing == pytest.approx([spacing, spacing, spacing])


def test_psf_single_fwhm_replicates():
    """A single FWHM value is replicated for both axial and lateral."""
    psf = PsfFromFwhm(fwhm=[2.0], shape=(128, 128))
    assert len(psf.sigma_px) == 2
    assert psf.sigma_px[0] == pytest.approx(psf.sigma_px[1])


def test_psf_rejects_non_list_fwhm():
    """Non-list FWHM raises TypeError."""
    with pytest.raises(TypeError, match="fwhm must be a list"):
        PsfFromFwhm(fwhm=(2.0, 2.0))  # type: ignore[arg-type]
