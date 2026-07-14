from dataclasses import dataclass
from enum import Enum


class Metric(Enum):
    """ITK similarity metric for iterative registration."""

    CORRELATION = "correlation"
    MATTES = "mattes"
    MEAN_SQUARED_DIFFERENCE = "mean-squared-difference"

    @classmethod
    def from_string(cls, value: str) -> "Metric":
        mapping = {m.value: m for m in cls}
        if value not in mapping:
            raise ValueError(f"Unknown metric: {value!r}. Valid: {list(mapping)}")
        return mapping[value]


class RegistrationMethod(Enum):
    """Registration method presets with sensible defaults."""

    ITERATIVE_RIGID = "iterative_rigid"
    ITERATIVE_SIMILARITY = "iterative_similarity"
    ITERATIVE_AFFINE = "iterative_affine"
    PHASE_CORRELATION = "phase_correlation"

    @classmethod
    def from_string(cls, value: str) -> "RegistrationMethod":
        mapping = {
            "rigid": cls.ITERATIVE_RIGID,
            "similarity": cls.ITERATIVE_SIMILARITY,
            "affine": cls.ITERATIVE_AFFINE,
            "phase_correlation": cls.PHASE_CORRELATION,
        }
        if value not in mapping:
            raise ValueError(
                f"Unknown registration method: {value!r}. Valid: {list(mapping)}"
            )
        return mapping[value]


@dataclass
class RegistrationOptions:
    """Registration configuration.

    Most users only need to specify ``method``. Advanced users can override
    specific parameters for fine-tuned control.
    """

    method: RegistrationMethod = RegistrationMethod.ITERATIVE_RIGID

    # Optimizer (iterative methods only)
    learning_rate: float = 0.7
    min_step_length: float = 0.001
    max_iterations: int = 200
    relaxation_factor: float = 0.7

    # Metric (iterative methods only)
    metric: Metric = Metric.CORRELATION
    sampling_percentage: float = 1.0
    mattes_histogram_bins: int = 15

    # Transform initialization (iterative methods only)
    translate_only: bool = False
    initial_rotation: float = 0.0
    initial_scale: float = 1.0
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.0
    use_initializer: bool = False

    # Optimizer scaling (iterative methods only)
    translation_scale: float = 1.0
    scaling_scale: float = 10.0

    # Phase correlation
    subpixel: int = 100

    # Preprocessing
    window: str | None = None  # "hamming", "tukey", or None

    # Observers
    enable_observers: bool = False
