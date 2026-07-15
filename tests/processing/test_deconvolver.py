from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import (
    CPUBackend,
    ViewData,
    convolve_cpu,
)
from miplib.processing.deconvolution.deconvolver import (
    FusionMode,
    RLDeconvolver,
    RLOptions,
)
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psf
from tests.conftest import checkerboard_pattern, gaussian_spot, impulse


def _make_view_data(n_views=1, shape=(32, 32), seed=42):
    rng = np.random.default_rng(seed)
    images: list[Image] = []
    psfs: list[np.ndarray] = []
    adjs: list[np.ndarray] = []
    for i in range(n_views):
        obj = np.abs(rng.normal(loc=10, scale=2, size=shape).astype(np.float64))
        psf_img = Image(gaussian_spot(shape, sigma=1.5 + i * 0.5), spacing=(1.0, 1.0))
        psf, adj = prepare_psf(psf_img, psf_img.spacing)
        blurred = convolve_cpu(obj, psf)
        images.append(Image(blurred, spacing=(1.0, 1.0)))
        psfs.append(psf)
        adjs.append(adj)
    source = ArrayDataSource(images)
    return ViewData(
        source=source,
        psfs=psfs,
        adj_psfs=adjs,
        weights=[1.0] * n_views,
        backgrounds=[0.0] * n_views,
    )


def test_manual_step_loop():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5, stop_tau=0.0)
    )

    for _ in range(5):
        converged = deconv.step()
    assert converged

    result = deconv.result()
    assert result.shape == vd.source.shape
    assert result.min() >= 0
    assert result.max() > 0


def test_iteration_protocol():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5)
    )

    results = list(deconv)
    assert len(results) == 5


def test_for_loop_syntax():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=3, stop_tau=1.0)
    )

    count = 0
    for _ in deconv:
        count += 1
    assert 1 <= count <= 3


def test_run_convenience():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5, stop_tau=1.0)
    )
    result = deconv.run()
    assert result.shape == vd.source.shape
    assert len(deconv.tracker.to_dataframe()) >= 1


def test_result_before_iteration():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=5)
    )
    result = deconv.result()
    assert result.shape == vd.source.shape


def test_multi_view_identity():
    shape = (32, 32)
    rng = np.random.default_rng(42)
    obj = np.abs(rng.normal(loc=10, scale=2, size=shape).astype(np.float64))
    psf_img = Image(gaussian_spot(shape, sigma=1.5), spacing=(1.0, 1.0))
    psf, adj = prepare_psf(psf_img, psf_img.spacing)
    blurred = convolve_cpu(obj, psf)
    img = Image(blurred, spacing=(1.0, 1.0))

    vd1 = ViewData(
        source=ArrayDataSource([img]),
        psfs=[psf],
        adj_psfs=[adj],
        weights=[1.0],
        backgrounds=[0.0],
    )
    vd3 = ViewData(
        source=ArrayDataSource([img, img, img]),
        psfs=[psf, psf, psf],
        adj_psfs=[adj, adj, adj],
        weights=[1.0, 1.0, 1.0],
        backgrounds=[0.0, 0.0, 0.0],
    )

    backend1 = CPUBackend(vd1)
    backend3 = CPUBackend(vd3)
    opts = RLOptions(max_iterations=3, stop_tau=0.0)

    est1 = create_estimate(vd1.source, FirstEstimate.IMAGE_MEAN)
    est3 = create_estimate(vd3.source, FirstEstimate.IMAGE_MEAN)
    deconv1 = RLDeconvolver(backend1, estimate=est1, options=opts)
    deconv3 = RLDeconvolver(backend3, estimate=est3, options=opts)

    for _ in deconv1:
        pass
    for _ in deconv3:
        pass

    npt.assert_allclose(deconv1.result(), deconv3.result(), rtol=1e-5)


def test_progress_tracking():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=3, stop_tau=0.0)
    )

    for _ in deconv:
        pass

    df = deconv.tracker.to_dataframe()
    assert len(df) == 3
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]


def test_external_estimate_array():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    estimate = np.zeros(vd.source.shape, dtype=np.float32)
    deconv = RLDeconvolver(
        backend, estimate=estimate, options=RLOptions(max_iterations=1)
    )
    deconv.step()


def test_estimate_shape_mismatch():
    vd = _make_view_data(n_views=1)
    backend = CPUBackend(vd)
    est_null = np.zeros(vd.source.shape, dtype=np.float32)
    wrong = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="estimate shape"):
        RLDeconvolver(backend, estimate=wrong, options=RLOptions(max_iterations=1))
    RLDeconvolver(backend, estimate=est_null, options=RLOptions(max_iterations=1))


def test_rloptions_validation():
    with pytest.raises(ValueError, match="n_blocks"):
        RLOptions(n_blocks=0)
    with pytest.raises(ValueError, match="epsilon"):
        RLOptions(epsilon=-0.1)
    with pytest.raises(ValueError, match="max_iterations"):
        RLOptions(max_iterations=0)
    opts = RLOptions(fusion_mode="summative")
    assert opts.fusion_mode == FusionMode.SUMMATIVE
    with pytest.raises(ValueError):
        RLOptions(fusion_mode="invalid")


# ---------------------------------------------------------------------------
# Deconvolution of a known grating
# ---------------------------------------------------------------------------


def test_deconvolve_blurred_checkerboard():
    """RL deconvolution sharpens a Gaussian-blurred checkerboard.

    After deconvolving, the result should have larger gradients (sharper
    edges) than the blurred input, moving it closer to the original.
    """
    shape = (64, 64)
    obj = checkerboard_pattern(shape)
    psf_data = gaussian_spot((16, 16), sigma=2.0)
    psf_data /= psf_data.sum()

    blurred_data = convolve_cpu(obj, psf_data)
    psf_img = Image(psf_data, spacing=(1.0, 1.0))

    images = [Image(blurred_data, spacing=(1.0, 1.0))]
    psfs = [psf_img.view(np.ndarray)]
    adjs = [psf_data[::-1, ::-1].copy()]

    vd = ViewData(
        source=ArrayDataSource(images),
        psfs=psfs,
        adj_psfs=adjs,
        weights=[1.0],
        backgrounds=[0.0],
    )
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN)
    deconv = RLDeconvolver(
        backend,
        estimate=estimate,
        options=RLOptions(max_iterations=30, stop_tau=0.0),
    )
    for _ in deconv:
        pass

    result = deconv.result()

    def gradient_magnitude(arr):
        gy, gx = np.gradient(arr.astype(np.float64))
        return (gy**2 + gx**2).mean()

    grad_blurred = gradient_magnitude(blurred_data)
    grad_result = gradient_magnitude(result.view(np.ndarray))
    grad_original = gradient_magnitude(obj)

    # Deconvolution sharpens the image — gradients move toward original
    assert abs(grad_result - grad_original) < abs(grad_blurred - grad_original) * 0.6

    # Tracker recorded meaningful statistics
    df = deconv.tracker.to_dataframe()
    assert len(df) == 30
    assert list(df.columns) == ["t", "tau1", "leak", "e", "s", "u", "n"]
    assert df["tau1"].iloc[-1] < df["tau1"].iloc[0]  # tau1 decreases
    assert (df["t"] > 0).all()


def test_deconvolve_impulse_recovery():
    """RL deconvolution recovers a Gaussian-blurred impulse.

    Starting from a constant estimate, after a few RL iterations the
    pixel-level error vs the original impulse should drop well below the
    blurred input's error.
    """
    shape = (32, 32)
    original = impulse(shape)
    psf_data = gaussian_spot((8, 8), sigma=1.0)
    psf_data /= psf_data.sum()

    blurred_data = convolve_cpu(original, psf_data)
    psf_img = Image(psf_data, spacing=(1.0, 1.0))

    images = [Image(blurred_data, spacing=(1.0, 1.0))]
    psfs = [psf_img.view(np.ndarray)]
    adjs = [psf_data[::-1, ::-1].copy()]

    vd = ViewData(
        source=ArrayDataSource(images),
        psfs=psfs,
        adj_psfs=adjs,
        weights=[1.0],
        backgrounds=[0.0],
    )
    backend = CPUBackend(vd)
    estimate = create_estimate(vd.source, FirstEstimate.IMAGE_MEAN, constant=0.0)
    deconv = RLDeconvolver(
        backend,
        estimate=estimate,
        options=RLOptions(max_iterations=10, stop_tau=0.0),
    )
    for _ in deconv:
        pass

    result = deconv.result()

    def mse(a, b):
        return ((a - b) ** 2).mean()

    mse_blurred = mse(blurred_data, original)
    mse_deconv = mse(result.view(np.ndarray), original)

    # Deconvolution should reduce MSE — narrower PSF and more iterations
    # produce a sharper recovery
    assert mse_deconv < mse_blurred * 0.7
    assert result.min() >= 0


@pytest.mark.integration
def test_deconvolution_pipeline_on_real_ism_image():
    """RL deconv with FRC-estimated PSF improves resolution on real ISM image."""
    from pathlib import Path

    from skimage import io

    from miplib.analysis.resolution.common import FRCOptions
    from miplib.analysis.resolution.fourier_ring_correlation import (
        calculate_single_image_frc,
    )
    from miplib.data.adapters.image_data import ArrayDataSource
    from miplib.data.containers.image import Image
    from miplib.processing.deconvolution.backends import ViewData, resolve_backend
    from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
    from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate
    from miplib.processing.deconvolution.psf_utils import prepare_psf
    from miplib.psf.psfgen import generate_frc_based_psf

    path = Path(__file__).parent.parent / "testdata" / "ism_dendrite.tiff"
    if not path.is_file():
        pytest.skip(f"Test image not found: {path}")

    data = io.imread(str(path)).astype(np.float64)
    spacing = (0.1, 0.1)
    img = Image(data[200:456, 200:456], spacing=spacing)

    psf = generate_frc_based_psf(img)
    assert psf.ndim == 2
    assert psf[psf.shape[0] // 2, psf.shape[1] // 2] == pytest.approx(1.0)

    psf_arr, adj_psf_arr = prepare_psf(psf, spacing)

    source = ArrayDataSource([img])
    vd = ViewData(
        source=source,
        psfs=[psf_arr],
        adj_psfs=[adj_psf_arr],
        weights=[1.0],
        backgrounds=[0.0],
    )
    backend = resolve_backend("cpu", vd, vd.source.shape)
    estimate = create_estimate(source, FirstEstimate.IMAGE)
    options = RLOptions(max_iterations=5, stop_tau=1e-8, tracker_type="frc")

    initial = calculate_single_image_frc(img, FRCOptions()).resolution["resolution"]

    deconv = RLDeconvolver(backend, estimate=estimate, options=options)
    for _ in deconv:
        pass

    result_img = deconv.result()
    assert result_img.shape == img.shape
    assert np.isfinite(result_img).all()

    final = calculate_single_image_frc(result_img, FRCOptions()).resolution[
        "resolution"
    ]
    assert final is not None, "FRC resolution not found on deconvolved image"
    assert final < initial, f"FRC did not improve: {initial:.4f} → {final:.4f} µm"
