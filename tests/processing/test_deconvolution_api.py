from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.containers.image import Image
from miplib.processing.deconvolution import (
    FirstEstimate,
    RLOptions,
    richardson_lucy_deconvolution,
    richardson_lucy_fusion,
)


@pytest.fixture
def camera_crop(camera_image):
    """128x128 crop of the camera image at the ISM pixel spacing (0.1 µm)."""
    data = np.asarray(camera_image)[:128, :128]
    return Image(data.copy(), spacing=(0.1, 0.1))


def test_fusion_identical_views_matches_single_view(camera_crop, psf_gaussian_2d):
    opts = RLOptions(max_iterations=5, stop_tau=0.0)

    single = richardson_lucy_deconvolution(camera_crop, psf_gaussian_2d, options=opts)
    fused = richardson_lucy_fusion(
        [camera_crop, camera_crop],
        [psf_gaussian_2d, psf_gaussian_2d],
        options=opts,
    )

    npt.assert_allclose(fused, single, rtol=1e-5)


def test_estimate_constant_reaches_initial_estimate(camera_crop, psf_gaussian_2d):
    def run(constant):
        opts = RLOptions(
            max_iterations=1,
            stop_tau=0.0,
            first_estimate=FirstEstimate.CONSTANT,
            estimate_constant=constant,
        )
        return richardson_lucy_deconvolution(camera_crop, psf_gaussian_2d, options=opts)

    # A zero initial estimate stays zero under the multiplicative RL update;
    # a non-zero constant does not. This fails if the constant is ignored.
    assert not np.asarray(run(0.0)).any()
    assert np.asarray(run(1.0)).max() > 0


def test_estimate_path_matches_in_memory(camera_crop, psf_gaussian_2d, tmp_path):
    opts = RLOptions(max_iterations=2, stop_tau=0.0)
    estimate_file = tmp_path / "estimate.dat"

    in_memory = richardson_lucy_deconvolution(
        camera_crop, psf_gaussian_2d, options=opts
    )
    mapped = richardson_lucy_deconvolution(
        camera_crop, psf_gaussian_2d, options=opts, estimate_path=estimate_file
    )

    assert estimate_file.is_file()
    assert estimate_file.stat().st_size > 0
    npt.assert_allclose(mapped, in_memory, rtol=1e-6)


def test_on_iteration_reports_each_step(camera_crop, psf_gaussian_2d):
    opts = RLOptions(max_iterations=3, stop_tau=0.0)
    observed = []

    def record(deconv):
        observed.append(deconv.tracker.to_dataframe().shape[0])

    plain = richardson_lucy_deconvolution(camera_crop, psf_gaussian_2d, options=opts)
    result = richardson_lucy_deconvolution(
        camera_crop, psf_gaussian_2d, options=opts, on_iteration=record
    )

    assert observed == [1, 2, 3]
    npt.assert_allclose(result, plain)


def test_psf_count_mismatch_raises(camera_crop, psf_gaussian_2d):
    with pytest.raises(ValueError, match=r"Expected 1 PSF\(s\), got 2"):
        richardson_lucy_fusion(
            [camera_crop],
            [psf_gaussian_2d, psf_gaussian_2d],
        )


def test_invalid_backend_raises(camera_crop, psf_gaussian_2d):
    with pytest.raises(ValueError, match="Unknown backend"):
        richardson_lucy_deconvolution(camera_crop, psf_gaussian_2d, backend="tpu")
