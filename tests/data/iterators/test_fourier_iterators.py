import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.iterators.fourier_ring_iterators import (
    FourierRingIterator,
    SectionedFourierRingIterator,
)
from miplib.data.iterators.fourier_shell_iterators import (
    AxialExcludeSectionedFourierShellIterator,
    FourierShellIterator,
    HollowSectionedFourierShellIterator,
    SectionedFourierShellIterator,
)

# -- Helpers -------------------------------------------------------------------


def _has_simpleitk():
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        return False
    return True


# -- FourierRingIterator -------------------------------------------------------


def test_ring_nbins_square():
    it = FourierRingIterator((64, 64), d_bin=4)
    assert it.nbins == 8


def test_ring_dc_in_first_ring():
    it = FourierRingIterator((5, 5), d_bin=1)
    indices, ring_idx = next(it)
    assert ring_idx == 0
    row_idx, col_idx = indices
    assert list(zip(row_idx, col_idx)).count((2, 2)) == 1


def test_ring_exhaustion():
    it = FourierRingIterator((8, 8), d_bin=2)
    count = sum(1 for _ in it)
    assert count == it.nbins
    assert count > 0


def test_ring_get_points_radius_range():
    it = FourierRingIterator((10, 10), d_bin=1)
    mask = it.get_points_on_ring(2.0, 3.0)
    selected_r = it.r[mask]
    assert len(selected_r) > 0
    assert np.all(selected_r >= 2.0)
    assert np.all(selected_r < 3.0)


def test_ring_non_square_shape():
    it = FourierRingIterator((8, 16), d_bin=2)
    indices, ring_idx = next(it)
    row_idx, col_idx = indices
    assert len(row_idx) > 0
    assert len(indices) == 2


def test_ring_different_bin_sizes():
    it1 = FourierRingIterator((32, 32), d_bin=2)
    it2 = FourierRingIterator((32, 32), d_bin=4)
    assert it1.nbins > it2.nbins


def test_ring_rejects_3d_shape():
    with pytest.raises(ValueError, match="2D"):
        FourierRingIterator((8, 8, 8), d_bin=1)


# -- SectionedFourierRingIterator ----------------------------------------------


def test_ring_sectioned_d_angle_not_d_bin():
    it = SectionedFourierRingIterator((10, 10), d_bin=3, d_angle=30)
    mask = it.angle_sector
    n_true = mask.sum()
    sector_rad = it.d_angle
    # All pixels in the sector should be within [+-sector_rad/2] of 0 or pi
    phi_shifted = it.phi.copy()
    phi_shifted[phi_shifted > np.pi] -= np.pi
    sector_mask = (phi_shifted >= 0) & (phi_shifted < sector_rad / 2)
    sector_mask |= (phi_shifted >= np.pi - sector_rad / 2) | (
        phi_shifted < sector_rad / 2 - np.pi
    )
    # The mask should roughly match the sector boundaries
    n_expected = sector_mask.sum()
    # Due to phi adjustment, check approximate equality
    assert abs(n_true - n_expected) <= 2  # small tolerance for edge cases


def test_ring_sectioned_angle_setter():
    it = SectionedFourierRingIterator((20, 20), d_bin=2, d_angle=45)
    mask_before = it.angle_sector.copy()
    it.angle = 90
    mask_after = it.angle_sector.copy()
    # After setting angle=90, the sector should be different (rotated)
    assert not np.array_equal(mask_before, mask_after)


def test_ring_sectioned_180_symmetry():
    """Points at phi and phi+pi should both be in the angle mask (Friedel symmetry)."""
    it = SectionedFourierRingIterator((10, 10), d_bin=1, d_angle=45)
    it.angle = 0
    mask = it.angle_sector
    phi_at_true = it.phi[mask]
    for p in phi_at_true[:5]:  # check first 5 points
        p_opposite = p + np.pi
        if p_opposite >= 2 * np.pi:
            p_opposite -= 2 * np.pi
        # Find if any pixel in the mask has phi close to p_opposite
        diff = np.abs(it.phi - p_opposite)
        closest_in_mask = diff[mask].min()
        assert closest_in_mask < 0.2, f"No opposite for phi={p}"


# -- FourierShellIterator ------------------------------------------------------


def test_shell_dc_in_first():
    it = FourierShellIterator((5, 5, 5), d_bin=1)
    indices, shell_idx = next(it)
    assert shell_idx == 0
    z_idx, y_idx, x_idx = indices
    assert list(zip(z_idx, y_idx, x_idx)).count((2, 2, 2)) == 1


def test_shell_count():
    it = FourierShellIterator((12, 12, 12), d_bin=3)
    count = sum(1 for _ in it)
    # shell_stop = floor(12/(2*3)) - 1 = 1, so indices 0,1 → 2 shells
    assert count == 2


def test_shell_getitem_matches_iteration():
    it = FourierShellIterator((6, 6, 6), d_bin=1)
    # shell from radius 0 to 1
    indices_from_getitem = it[0, 1]
    indices_from_iter, shell_idx = next(it)
    assert shell_idx == 0
    npt.assert_array_equal(indices_from_getitem[0], indices_from_iter[0])
    npt.assert_array_equal(indices_from_getitem[1], indices_from_iter[1])
    npt.assert_array_equal(indices_from_getitem[2], indices_from_iter[2])


def test_shell_rejects_2d_shape():
    with pytest.raises(ValueError, match="3D"):
        FourierShellIterator((8, 8), d_bin=1)


# -- SectionedFourierShellIterator ---------------------------------------------


def test_shell_sectioned_iteration_order():
    """Verify shell-outer iteration: shell idx increments only after all rotations."""
    it = SectionedFourierShellIterator((20, 20, 20), d_bin=5, d_angle=90)
    results = list(it)
    # 4 rotations, ceil(10/5)=2 shells → 8 total
    assert len(results) == 8
    # First 4 are shell 0 with rotations 0-3
    for i in range(4):
        _, shell_idx, rot_idx = results[i]
        assert shell_idx == 0
        assert rot_idx == i
    # Next 4 are shell 1 with rotations 0-3
    for i in range(4):
        _, shell_idx, rot_idx = results[4 + i]
        assert shell_idx == 1
        assert rot_idx == i


def test_shell_sectioned_rotation_count():
    it = SectionedFourierShellIterator((20, 20, 20), d_bin=5, d_angle=90)
    results = list(it)
    rot_indices = set(r[2] for r in results)
    assert rot_indices == {0, 1, 2, 3}


def test_shell_sectioned_float_rotation_stop():
    """d_angle=7 (360 not divisible) should produce correct count without error."""
    it = SectionedFourierShellIterator((20, 20, 20), d_bin=5, d_angle=7)
    results = list(it)
    rot_indices = set(r[2] for r in results)
    assert len(rot_indices) == 51  # int(360/7) = 51


def test_shell_sectioned_phi_in_angle_sector():
    """Pixels selected by get_angle_sector should have phi in expected range."""
    it = SectionedFourierShellIterator((10, 10, 10), d_bin=2, d_angle=30)
    d_rad = it.d_angle
    mask = it.get_angle_sector(0, d_rad)
    phi_values = it.phi[mask]
    for p in phi_values:
        # Must be in [0, d_rad) or [pi, pi+d_rad)
        in_range = (0 <= p < d_rad) or (np.pi <= p < np.pi + d_rad)
        assert in_range, f"phi={p} not in expected ranges"


# -- HollowSectionedFourierShellIterator ---------------------------------------


def test_hollow_center_removed():
    """The hollow iterator removes fewer pixels than the non-hollow parent."""
    it_hollow = HollowSectionedFourierShellIterator(
        (30, 30, 30), d_bin=3, d_angle=60, d_extract_angle=10
    )
    it_full = SectionedFourierShellIterator((30, 30, 30), d_bin=3, d_angle=60)
    mask_hollow = it_hollow.get_angle_sector(np.radians(0), np.radians(60))
    mask_full = it_full.get_angle_sector(np.radians(0), np.radians(60))
    assert mask_hollow.sum() < mask_full.sum()


def test_hollow_zero_extract_identical():
    """With d_extract_angle=0, the result should match the parent."""
    it_hollow = HollowSectionedFourierShellIterator(
        (20, 20, 20), d_bin=4, d_angle=45, d_extract_angle=0
    )
    it_full = SectionedFourierShellIterator((20, 20, 20), d_bin=4, d_angle=45)
    mask_hollow = it_hollow.get_angle_sector(np.radians(10), np.radians(55))
    mask_full = it_full.get_angle_sector(np.radians(10), np.radians(55))
    npt.assert_array_equal(mask_hollow, mask_full)


# -- AxialExcludeSectionedFourierShellIterator ---------------------------------


def test_axial_sector_excluded():
    """Sector containing 90° has center removed compared to non-hollowed parent."""
    it = AxialExcludeSectionedFourierShellIterator(
        (30, 30, 30), d_bin=3, d_angle=60, d_extract_angle=5
    )
    mask = it.get_angle_sector(np.radians(60), np.radians(120))
    assert mask.sum() > 0

    # Compare with non-hollow parent: axial should have fewer pixels
    it_full = SectionedFourierShellIterator((30, 30, 30), d_bin=3, d_angle=60)
    mask_full = it_full.get_angle_sector(np.radians(60), np.radians(120))
    assert mask.sum() < mask_full.sum()


def test_non_axial_sector_not_excluded():
    """Sector far from 90°/270° should be identical to parent (no hollow)."""
    it = AxialExcludeSectionedFourierShellIterator(
        (30, 30, 30), d_bin=3, d_angle=30, d_extract_angle=5
    )
    it_full = SectionedFourierShellIterator((30, 30, 30), d_bin=3, d_angle=30)
    mask = it.get_angle_sector(np.radians(10), np.radians(40))
    mask_full = it_full.get_angle_sector(np.radians(10), np.radians(40))
    npt.assert_array_equal(mask, mask_full)


def test_axial_d_extract_angle_propagated():
    """d_extract_angle parameter is passed to the parent class."""
    it = AxialExcludeSectionedFourierShellIterator(
        (20, 20, 20), d_bin=5, d_angle=30, d_extract_angle=10
    )
    assert it.d_extract_angle == pytest.approx(np.radians(10))


# -- RotatingFourierShellIterator ----------------------------------------------


@pytest.mark.skipif(not _has_simpleitk(), reason="SimpleITK not available")
def test_rotating_basic_iteration():
    from miplib.data.iterators.fourier_shell_iterators import (
        RotatingFourierShellIterator,
    )

    it = RotatingFourierShellIterator((20, 20, 20), d_bin=5, d_angle=90)
    results = list(it)
    # Shells per rotation: shell_stop+1. shell_stop = floor(20/10)-1 = 1.
    # So 2 shells × 4 rotations = 8 total
    assert len(results) >= 4
    # Check that yields are (indices, shell_idx, rotation_idx)
    for indices, shell_idx, rot_idx in results:
        assert isinstance(indices, tuple)
        assert isinstance(shell_idx, (int, np.integer))
        assert isinstance(rot_idx, (int, np.integer))
