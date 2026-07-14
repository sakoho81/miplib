"""Registration method backends.

Implements ``ITKRegistration`` (iterative 2D/3D rigid/similarity/affine) and
``PhaseCorrelationRegistration`` (single-step frequency-domain). Both conform
to the ``RegistrationBackend`` protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation

from miplib.data.containers.image import Image
from miplib.processing.itk import calculate_center_of_image, convert_to_itk_image
from miplib.processing.windowing import apply_hamming_window, apply_tukey_window

from .options import Metric, RegistrationMethod, RegistrationOptions

logger = logging.getLogger(__name__)


@dataclass
class RegistrationState:
    """Progress state yielded during registration."""

    iteration: int
    metric_value: float


class RegistrationBackend(Protocol):
    """Backend protocol for registration implementations."""

    def register(self, fixed: Image, moving: Image) -> sitk.Transform:
        """Register moving image to fixed image."""

    def register_iter(self, fixed: Image, moving: Image) -> Iterator[RegistrationState]:
        """Register and yield progress states."""


class ITKRegistration(RegistrationBackend):
    """Unified ITK-based registration for 2D/3D images.

    Supports rigid, similarity, and affine transforms. Dimensionality is
    auto-detected from the input images.
    """

    def __init__(
        self,
        transform_type: str,
        options: RegistrationOptions,
    ) -> None:
        if transform_type not in ("rigid", "similarity", "affine"):
            raise ValueError(
                f"Unknown transform type: {transform_type!r}. "
                f"Supported: rigid, similarity, affine"
            )
        self.transform_type = transform_type
        self.options = options
        self._result: sitk.Transform | None = None

    def register(self, fixed: Image, moving: Image) -> sitk.Transform:
        for _ in self.register_iter(fixed, moving):
            pass
        if self._result is None:
            raise RuntimeError("Registration produced no result")
        return self._result

    def register_iter(self, fixed: Image, moving: Image) -> Iterator[RegistrationState]:
        ndim = fixed.ndim

        fixed_itk = sitk.Cast(convert_to_itk_image(fixed), sitk.sitkFloat32)
        moving_itk = sitk.Cast(convert_to_itk_image(moving), sitk.sitkFloat32)

        reg = sitk.ImageRegistrationMethod()
        self._configure_optimizer(reg, ndim)
        self._configure_metric(reg)
        self._configure_interpolator(reg)

        initial_transform = self._create_initial_transform(moving_itk, fixed_itk, ndim)
        reg.SetInitialTransform(initial_transform)

        raw_states: list[RegistrationState] = []
        iteration_count = 0

        def _iteration_observer() -> None:
            nonlocal iteration_count
            raw_states.append(
                RegistrationState(
                    iteration=iteration_count,
                    metric_value=reg.GetMetricValue(),
                )
            )
            iteration_count += 1

        reg.AddCommand(sitk.sitkIterationEvent, _iteration_observer)

        if self.options.enable_observers:
            _start_observer_plot(reg)

        self._result = reg.Execute(fixed_itk, moving_itk)

        logger.info("Final metric value: %s", reg.GetMetricValue())
        logger.info("Stop condition: %s", reg.GetOptimizerStopConditionDescription())

        yield from raw_states

    def _configure_optimizer(
        self, reg: sitk.ImageRegistrationMethod, ndim: int
    ) -> None:
        reg.SetOptimizerAsRegularStepGradientDescent(
            self.options.learning_rate,
            self.options.min_step_length,
            self.options.max_iterations,
            relaxationFactor=self.options.relaxation_factor,
            estimateLearningRate=reg.EachIteration,
        )

        if self.transform_type == "rigid":
            if self.options.translate_only or ndim == 3:
                reg.SetOptimizerScalesFromJacobian()
            else:
                tscale = 1.0 / self.options.translation_scale
                reg.SetOptimizerScales([1.0, tscale, tscale])
        else:
            sscale = 1.0 / self.options.scaling_scale
            tscale = 1.0 / self.options.translation_scale
            reg.SetOptimizerScales([sscale, 1.0, tscale, tscale])

    def _configure_metric(self, reg: sitk.ImageRegistrationMethod) -> None:
        metric = self.options.metric
        if metric == Metric.MATTES:
            reg.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=self.options.mattes_histogram_bins
            )
        elif metric == Metric.CORRELATION:
            reg.SetMetricAsCorrelation()
        elif metric == Metric.MEAN_SQUARED_DIFFERENCE:
            reg.SetMetricAsMeanSquares()
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(self.options.sampling_percentage)

    @staticmethod
    def _configure_interpolator(reg: sitk.ImageRegistrationMethod) -> None:
        reg.SetInterpolator(sitk.sitkLinear)

    def _create_initial_transform(
        self,
        moving_itk: sitk.Image,
        fixed_itk: sitk.Image,
        ndim: int,
    ) -> sitk.Transform:
        if self.options.translate_only:
            return sitk.TranslationTransform(ndim)

        tx: sitk.Transform
        if self.transform_type == "rigid":
            if ndim == 2:
                tx = sitk.Euler2DTransform()
            else:
                tx = sitk.Euler3DTransform()
        elif self.transform_type == "similarity":
            tx = sitk.Similarity2DTransform()
            tx.SetScale(self.options.initial_scale)
        elif self.transform_type == "affine":
            tx = sitk.AffineTransform(ndim)
        else:
            raise ValueError(f"Unknown transform type: {self.transform_type!r}")

        if hasattr(tx, "SetAngle"):
            tx.SetAngle(self.options.initial_rotation)  # type: ignore[union-attr]

        if self.options.use_initializer:
            initialized = sitk.CenteredTransformInitializer(
                fixed_itk,
                moving_itk,
                tx,
                sitk.CenteredTransformInitializerFilter.MOMENTS,
            )
            return initialized

        offsets = [self.options.y_offset, self.options.x_offset]
        if ndim == 3:
            offsets.append(self.options.z_offset)
        tx.SetTranslation(offsets)
        tx.SetCenter(calculate_center_of_image(moving_itk))

        return tx


class PhaseCorrelationRegistration(RegistrationBackend):
    """Frequency-domain phase correlation registration."""

    _WINDOW_FUNCTIONS = {
        "hamming": apply_hamming_window,
        "tukey": apply_tukey_window,
    }

    def __init__(self, options: RegistrationOptions) -> None:
        self.subpixel = options.subpixel
        self.window = options.window
        self._result: sitk.Transform | None = None

    def register(self, fixed: Image, moving: Image) -> sitk.Transform:
        for _ in self.register_iter(fixed, moving):
            pass
        if self._result is None:
            raise RuntimeError("Registration produced no result")
        return self._result

    def register_iter(self, fixed: Image, moving: Image) -> Iterator[RegistrationState]:
        if not isinstance(fixed, Image):
            raise TypeError(f"Expected Image, got {type(fixed).__name__}")
        if not isinstance(moving, Image):
            raise TypeError(f"Expected Image, got {type(moving).__name__}")

        fixed_arr = np.asarray(fixed)
        moving_arr = np.asarray(moving)
        if self.window:
            fixed_arr = self._apply_window(fixed_arr, self.window)
            moving_arr = self._apply_window(moving_arr, self.window)

        shift, *_ = phase_cross_correlation(
            fixed_arr,
            moving_arr,
            upsample_factor=self.subpixel,
        )

        scaled_shifts = [
            -offset * spacing
            for offset, spacing in zip(shift, fixed.spacing, strict=False)
        ]

        self._result = sitk.TranslationTransform(len(shift))
        self._result.SetParameters(scaled_shifts[::-1])

        yield RegistrationState(iteration=0, metric_value=0.0)

    @classmethod
    def _apply_window(cls, data: np.ndarray, window_type: str) -> np.ndarray:
        if window_type not in cls._WINDOW_FUNCTIONS:
            raise ValueError(
                f"Unknown window type: {window_type!r}. "
                f"Supported: {list(cls._WINDOW_FUNCTIONS)}"
            )
        return cls._WINDOW_FUNCTIONS[window_type](data)  # type: ignore[operator]


def create_backend(options: RegistrationOptions) -> RegistrationBackend:
    """Create a registration backend from options."""
    if options.method == RegistrationMethod.PHASE_CORRELATION:
        return PhaseCorrelationRegistration(options)

    transform_map = {
        RegistrationMethod.ITERATIVE_RIGID: "rigid",
        RegistrationMethod.ITERATIVE_SIMILARITY: "similarity",
        RegistrationMethod.ITERATIVE_AFFINE: "affine",
    }
    transform_type = transform_map[options.method]
    return ITKRegistration(transform_type, options)


# ---- Observer callbacks for plotting ----

_metric_values: list[float] = []


def _start_observer_plot(reg: sitk.ImageRegistrationMethod) -> None:
    """Wire up a per-iteration callback that records metric values to a global list."""
    _metric_values.clear()

    def _on_iteration() -> None:
        _metric_values.append(reg.GetMetricValue())

    reg.AddCommand(sitk.sitkIterationEvent, _on_iteration)
