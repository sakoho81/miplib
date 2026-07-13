"""Tests for miplib/processing/registration/."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from miplib.data.adapters.registration_data import (
    ArrayDetectorDataSource,
    ArrayRegistrationDataSource,
)
from miplib.data.containers.image import Image
from miplib.processing.registration import (
    Metric,
    MultiViewRegistration,
    RegistrationMethod,
    RegistrationOptions,
    register,
    register_iter,
)
from miplib.processing.registration.methods import (
    ITKRegistration,
    PhaseCorrelationRegistration,
)
from miplib.processing.registration.stack import (
    register_stack_slices,
    register_stack_slices_with_reference,
    shift_stack_slices,
)
from tests.conftest import gaussian_spot

# ---------------------------------------------------------------------------
# Phase correlation — known-shift recovery
# ---------------------------------------------------------------------------


def test_phase_correlation_identity_2d(camera_image: Image):
    opts = RegistrationOptions(method=RegistrationMethod.PHASE_CORRELATION)
    transform = PhaseCorrelationRegistration(opts).register(camera_image, camera_image)
    assert isinstance(transform, sitk.TranslationTransform)
    assert np.allclose(transform.GetParameters(), 0, atol=1e-3)


def test_phase_correlation_identity_3d(blobs_3d: Image):
    opts = RegistrationOptions(method=RegistrationMethod.PHASE_CORRELATION)
    transform = PhaseCorrelationRegistration(opts).register(blobs_3d, blobs_3d)
    assert isinstance(transform, sitk.TranslationTransform)
    assert np.allclose(transform.GetParameters(), 0, atol=1e-3)


def test_phase_correlation_known_shift(shepp_logan: Image):
    shifted = np.roll(shepp_logan, shift=(-5, 7), axis=(0, 1))
    moving = Image(shifted, shepp_logan.spacing)

    opts = RegistrationOptions(method=RegistrationMethod.PHASE_CORRELATION)
    transform = PhaseCorrelationRegistration(opts).register(shepp_logan, moving)
    params = np.asarray(transform.GetParameters())

    # Roll (-5, 7) px, spacing=1 → ITK (x, y) = (7.0, -5.0)
    assert np.allclose(params, [7.0, -5.0], atol=1e-2)


def test_phase_correlation_subpixel_accuracy(camera_image: Image):
    """Higher subpixel factor should improve shift accuracy."""
    shifted = np.roll(camera_image, shift=(-2, 3), axis=(0, 1))
    moving = Image(shifted, camera_image.spacing)

    opts_coarse = RegistrationOptions(
        method=RegistrationMethod.PHASE_CORRELATION, subpixel=1
    )
    opts_fine = RegistrationOptions(
        method=RegistrationMethod.PHASE_CORRELATION, subpixel=100
    )
    t_coarse = PhaseCorrelationRegistration(opts_coarse).register(camera_image, moving)
    t_fine = PhaseCorrelationRegistration(opts_fine).register(camera_image, moving)

    expected = np.array([3.0, -2.0])
    err_coarse = np.abs(np.asarray(t_coarse.GetParameters()) - expected).sum()
    err_fine = np.abs(np.asarray(t_fine.GetParameters()) - expected).sum()
    assert err_fine <= err_coarse + 1e-2


def test_phase_correlation_rejects_non_image():
    opts = RegistrationOptions(method=RegistrationMethod.PHASE_CORRELATION)
    with pytest.raises(TypeError):
        PhaseCorrelationRegistration(opts).register(
            np.zeros((32, 32)),
            np.zeros((32, 32)),  # type: ignore[arg-type]
        )


def test_phase_correlation_window_hamming(camera_image: Image):
    """Hamming window — should still recover the shift on a known image."""
    shifted = np.roll(camera_image, shift=(0, 4), axis=(0, 1))
    moving = Image(shifted, camera_image.spacing)

    opts = RegistrationOptions(
        method=RegistrationMethod.PHASE_CORRELATION, window="hamming"
    )
    transform = PhaseCorrelationRegistration(opts).register(camera_image, moving)
    params = np.asarray(transform.GetParameters())
    # Expect (x=4, y=0) with spacing=1
    assert abs(params[0] - 4.0) < 1.0
    assert abs(params[1]) < 1.0


# ---------------------------------------------------------------------------
# ITK registration — known-shift recovery, convergence, transform types
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_img() -> Image:
    return Image(gaussian_spot((64, 64), sigma=5), spacing=(0.1, 0.1))


def test_itk_identity_near_zero(gaussian_img: Image):
    backend = ITKRegistration(
        "rigid", RegistrationOptions(min_step_length=0.1, max_iterations=10)
    )
    transform = backend.register(gaussian_img, gaussian_img)
    params = np.asarray(transform.GetParameters())
    assert np.allclose(params, 0, atol=0.5)


def test_itk_recovers_shift_with_convergence(camera_image: Image):
    """Register an image shifted by 3 px along x — verify shift and metric convergence."""
    shifted = np.roll(camera_image, shift=(0, 3), axis=(0, 1))
    moving = Image(shifted, camera_image.spacing)

    backend = ITKRegistration(
        "rigid",
        RegistrationOptions(
            max_iterations=50,
            min_step_length=1e-5,
            metric=Metric.CORRELATION,
            translate_only=True,
        ),
    )
    states = list(backend.register_iter(camera_image, moving))
    assert len(states) >= 2
    # Correlation cost decreases (more negative = better fit)
    assert states[-1].metric_value <= states[0].metric_value + 0.05

    transform = backend.register(camera_image, moving)
    params = np.asarray(transform.GetParameters())
    # ITK (x, y), spacing=1, roll (0, 3) → expect x ≈ 3.0
    assert abs(params[0] - 3.0) < 1.0
    assert abs(params[1]) < 1.0


@pytest.mark.parametrize(
    "transform_type,opts,expected_class",
    [
        ("affine", {}, sitk.AffineTransform),
        ("similarity", {}, sitk.Similarity2DTransform),
        ("rigid", {"translate_only": True}, sitk.TranslationTransform),
    ],
)
def test_itk_transform_types(gaussian_img, transform_type, opts, expected_class):
    reg_opts = RegistrationOptions(max_iterations=5, **opts)
    backend = ITKRegistration(transform_type, reg_opts)
    transform = backend.register(gaussian_img, gaussian_img)
    assert isinstance(transform, expected_class)


def test_itk_rejects_invalid_transform_type():
    with pytest.raises(ValueError):
        ITKRegistration("unknown", RegistrationOptions())


def test_metric_rejects_invalid_string():
    with pytest.raises(ValueError, match="Unknown metric"):
        Metric.from_string("invalid_metric")


# ---------------------------------------------------------------------------
# register() convenience — identity + shift recovery + overrides
# ---------------------------------------------------------------------------


def test_register_rigid_recovers_shift(camera_image: Image):
    shifted = np.roll(camera_image, shift=(5, 0), axis=(0, 1))
    moving = Image(shifted, camera_image.spacing)

    transform = register(
        camera_image,
        moving,
        RegistrationMethod.ITERATIVE_RIGID,
        translate_only=True,
        max_iterations=30,
        min_step_length=1e-5,
    )
    params = np.asarray(transform.GetParameters())
    # ITK (x, y), spacing=1, roll (5, 0) down → y ≈ 5.0
    assert abs(params[0]) < 1.0
    assert abs(params[1] - 5.0) < 1.0


def test_register_affine_identity(camera_image: Image):
    transform = register(
        camera_image,
        camera_image,
        RegistrationMethod.ITERATIVE_AFFINE,
        max_iterations=5,
    )
    assert isinstance(transform, sitk.AffineTransform)
    params = np.asarray(transform.GetParameters())
    assert len(params) == 6  # 2D affine


def test_register_invalid_override_raises_typeerror(camera_image: Image):
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        register(
            camera_image,
            camera_image,
            RegistrationMethod.ITERATIVE_RIGID,
            nonexistent_field=42,
        )


# ---------------------------------------------------------------------------
# register_iter — convergence validation
# ---------------------------------------------------------------------------


def test_register_iter_metric_converges(camera_image: Image):
    shifted = np.roll(camera_image, shift=(1, 2), axis=(0, 1))
    moving = Image(shifted, camera_image.spacing)

    states = list(
        register_iter(
            camera_image,
            moving,
            RegistrationMethod.ITERATIVE_RIGID,
            max_iterations=40,
            min_step_length=1e-5,
            metric=Metric.CORRELATION,
        )
    )
    assert len(states) >= 2
    first, last = states[0].metric_value, states[-1].metric_value
    assert last <= first + 0.05  # correlation cost improves


# ---------------------------------------------------------------------------
# MultiViewRegistration — known shifts + iterator
# ---------------------------------------------------------------------------


def test_multiview_recovers_known_shifts(camera_image: Image):
    shifted = np.roll(camera_image, shift=(-3, 4), axis=(0, 1))
    images = [
        camera_image.copy(),
        camera_image.copy(),
        Image(shifted, camera_image.spacing),
    ]
    source = ArrayRegistrationDataSource(images)
    mv = MultiViewRegistration(source, method=RegistrationMethod.PHASE_CORRELATION)
    transforms = mv.execute()

    assert np.allclose(transforms[0].GetParameters(), 0, atol=1e-3)  # identity
    assert np.allclose(transforms[1].GetParameters(), 0, atol=1e-3)  # identical copy
    params2 = np.asarray(transforms[2].GetParameters())
    assert abs(params2[0] - 4.0) < 0.5  # recovered x-shift
    assert abs(params2[1] + 3.0) < 0.5  # recovered y-shift


def test_multiview_fixed_image_identity(camera_image: Image):
    images = [camera_image.copy() for _ in range(3)]
    source = ArrayRegistrationDataSource(images)
    mv = MultiViewRegistration(
        source, fixed_idx=1, method=RegistrationMethod.PHASE_CORRELATION
    )
    transforms = mv.execute()
    assert np.allclose(transforms[1].GetParameters(), 0, atol=1e-6)


def test_multiview_rejects_fixed_idx_out_of_range(camera_image: Image):
    source = ArrayRegistrationDataSource([camera_image.copy() for _ in range(2)])
    with pytest.raises(ValueError, match="fixed_idx"):
        MultiViewRegistration(source, fixed_idx=5)


# ---------------------------------------------------------------------------
# Stack registration — known shifts + error handling
# ---------------------------------------------------------------------------


def test_stack_register_slices_recovers_shifts():
    """Stack of gaussian spots where slice i is shifted by (i*3, i*4) px."""
    n_slices = 4
    spot = gaussian_spot((128, 128), sigma=10)
    data = np.array(
        [np.roll(spot, shift=(i * 3, i * 4), axis=(0, 1)) for i in range(n_slices)],
        dtype=np.float64,
    )
    stack = Image(data, spacing=(0.5, 1.0, 1.0))
    shifts = register_stack_slices(stack)

    assert abs(shifts[0]).sum() < 1e-3
    for i in range(1, n_slices):
        assert abs(shifts[i, 0] - 3 * i) < 0.5
        assert abs(shifts[i, 1] - 4 * i) < 0.5


def test_stack_register_with_reference_offset():
    """Gaussian spots shifted by 0, 5, 10, 15 px along y, registered to first slice."""
    n_slices = 4
    spot = gaussian_spot((128, 128), sigma=10)
    data = np.array(
        [np.roll(spot, shift=(i * 5, 0), axis=(0, 1)) for i in range(n_slices)],
        dtype=np.float64,
    )
    stack = Image(data, spacing=(0.5, 1.0, 1.0))
    fixed = Image(stack[0], stack.spacing[1:])
    shifts = register_stack_slices_with_reference(stack, fixed)

    assert abs(shifts[0]).sum() < 1e-3
    for i in range(1, n_slices):
        assert abs(shifts[i, 0] - 5 * i) < 0.5
        assert abs(shifts[i, 1]) < 0.5


def test_stack_shift_zero_shifts_preserves_shape():
    spot = gaussian_spot((64, 64), sigma=5)
    data = np.array([spot for _ in range(4)])
    stack = Image(data, spacing=(0.5, 1.0, 1.0))
    result = shift_stack_slices(stack, np.zeros((4, 2)))
    assert result.shape == data.shape
    assert result.spacing == stack.spacing


def test_stack_rejects_2d(camera_image: Image):
    with pytest.raises(ValueError, match="Expected 3D stack"):
        register_stack_slices(camera_image)


def test_stack_rejects_mismatched_shifts(camera_image: Image):
    stack = Image(
        np.array([np.asarray(camera_image) for _ in range(4)]),
        spacing=(0.5, 1.0, 1.0),
    )
    with pytest.raises(ValueError, match="Shift array does not match"):
        shift_stack_slices(stack, np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# RegistrationDataSource adapters
# ---------------------------------------------------------------------------


def test_array_source_returns_images(camera_image: Image):
    images = [camera_image.copy() for _ in range(3)]
    source = ArrayRegistrationDataSource(images)
    assert source.n_views == 3
    assert source.spacing == tuple(camera_image.spacing)
    assert source.get_image(1) is images[1]
    assert source.exists(0) is False
    source.save_result(0, images[0], sitk.TranslationTransform(2))


def test_array_source_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        ArrayRegistrationDataSource([])


def test_detector_source_basic():
    """ArrayDetectorDataSource wraps ArrayDetectorData as a view (no copy)."""
    from miplib.data.containers.array_detector_data import ArrayDetectorData

    data = ArrayDetectorData(detectors=3, gates=1)
    img = Image(
        np.random.default_rng(42).random((32, 32)).astype(np.float64),
        spacing=(0.1, 0.1),
    )
    for i in range(3):
        data[0, i] = img

    source = ArrayDetectorDataSource(data, photosensor=0)
    assert source.n_views == 3
    assert source.spacing == (0.1, 0.1)
    assert source.get_image(0) is data[0, 0]
    assert source.exists(0) is False
    source.save_result(0, img, sitk.TranslationTransform(2))


def test_detector_source_rejects_invalid_photosensor():
    from miplib.data.containers.array_detector_data import ArrayDetectorData

    data = ArrayDetectorData(detectors=2, gates=1)
    with pytest.raises(ValueError, match="photosensor"):
        ArrayDetectorDataSource(data, photosensor=5)


def test_detector_source_find_shifts_with_phase_correlation():
    """End-to-end: find_image_shifts with phase correlation via ArrayDetectorDataSource."""
    from miplib.data.containers.array_detector_data import ArrayDetectorData
    from miplib.processing.ism.reconstruction import find_image_shifts

    data = ArrayDetectorData(detectors=3, gates=1)
    rng = np.random.default_rng(42)
    img = Image(rng.random((64, 64)).astype(np.float64), spacing=(0.1, 0.1))
    data[0, 0] = img
    data[0, 1] = img.copy()
    data[0, 2] = Image(
        np.roll(np.asarray(img), shift=(2, 0), axis=(0, 1)), spacing=(0.1, 0.1)
    )

    shifts, transforms = find_image_shifts(
        data,
        method=RegistrationMethod.PHASE_CORRELATION,
        fixed_idx=0,
    )
    assert len(transforms) == 3
    # View 0 is fixed → identity
    assert np.allclose(transforms[0].GetParameters(), 0, atol=1e-3)
    # View 1 is identical → near zero
    assert np.allclose(transforms[1].GetParameters(), 0, atol=1e-1)
    # View 2 is shifted by 2 px down → y-shift ≈ -0.2 in ITK
    params2 = np.asarray(transforms[2].GetParameters())
    assert abs(params2[1] - 0.2) < 0.1


# ---------------------------------------------------------------------------
# RegistrationOptions
# ---------------------------------------------------------------------------


def test_options_default_values():
    opts = RegistrationOptions()
    assert opts.method == RegistrationMethod.ITERATIVE_RIGID
    assert opts.learning_rate == 0.7
    assert opts.metric == Metric.CORRELATION
    assert opts.subpixel == 100
    assert opts.window is None


def test_options_custom_values():
    opts = RegistrationOptions(
        method=RegistrationMethod.PHASE_CORRELATION,
        max_iterations=500,
        metric=Metric.MATTES,
        subpixel=200,
        window="hamming",
    )
    assert opts.method == RegistrationMethod.PHASE_CORRELATION
    assert opts.max_iterations == 500
    assert opts.metric == Metric.MATTES
    assert opts.subpixel == 200
    assert opts.window == "hamming"


def test_registration_method_enum_from_string():
    assert RegistrationMethod.from_string("rigid") == RegistrationMethod.ITERATIVE_RIGID
    assert (
        RegistrationMethod.from_string("phase_correlation")
        == RegistrationMethod.PHASE_CORRELATION
    )
    with pytest.raises(ValueError, match="Unknown registration method"):
        RegistrationMethod.from_string("invalid")


# ---------------------------------------------------------------------------
# Top-level error handling
# ---------------------------------------------------------------------------


def test_register_rejects_non_image():
    with pytest.raises(TypeError):
        register(np.zeros((32, 32)), np.zeros((32, 32)))  # type: ignore[arg-type]
