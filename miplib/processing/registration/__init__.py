"""Image registration module.

Public API:
    ``register()`` — register two images and return the transform.
    ``register_iter()`` — register two images and yield progress states.

    ``RegistrationMethod`` — enum of preset registration methods.
    ``RegistrationOptions`` — dataclass for fine-grained control.
"""

from __future__ import annotations

from collections.abc import Iterator

import SimpleITK as sitk

from miplib.data.containers.image import Image

from .methods import (
    ITKRegistration,
    PhaseCorrelationRegistration,
    RegistrationBackend,
    RegistrationState,
    create_backend,
)
from .multiview import MultiViewRegistration
from .options import Metric, RegistrationMethod, RegistrationOptions

__all__ = [
    "ITKRegistration",
    "Metric",
    "MultiViewRegistration",
    "PhaseCorrelationRegistration",
    "RegistrationBackend",
    "RegistrationMethod",
    "RegistrationOptions",
    "RegistrationState",
    "register",
    "register_iter",
]


def register(
    fixed: Image,
    moving: Image,
    method: RegistrationMethod = RegistrationMethod.ITERATIVE_RIGID,
    **overrides: object,
) -> sitk.Transform:
    """Register moving image to fixed image.

    Args:
        fixed: Reference image.
        moving: Image to register.
        method: Registration method preset.
        **overrides: Override specific ``RegistrationOptions`` fields.

    Returns:
        Transform that aligns *moving* to *fixed*.

    Examples:
        >>> transform = register(fixed, moving)
        >>> transform = register(fixed, moving, RegistrationMethod.PHASE_CORRELATION)
        >>> transform = register(
        ...     fixed, moving,
        ...     RegistrationMethod.ITERATIVE_AFFINE,
        ...     max_iterations=500,
        ...     metric="mattes",
        ... )
    """
    options = _build_options(method, **overrides)
    backend = create_backend(options)
    return backend.register(fixed, moving)


def register_iter(
    fixed: Image,
    moving: Image,
    method: RegistrationMethod = RegistrationMethod.ITERATIVE_RIGID,
    **overrides: object,
) -> Iterator[RegistrationState]:
    """Register and yield progress states.

    Args:
        fixed: Reference image.
        moving: Image to register.
        method: Registration method preset.
        **overrides: Override specific ``RegistrationOptions`` fields.

    Yields:
        ``RegistrationState`` at each iteration.

    Examples:
        >>> for state in register_iter(fixed, moving):
        ...     print(f"Iteration {state.iteration}, metric = {state.metric_value:.4f}")
        >>> transform = register(fixed, moving)  # or call register() separately
    """
    options = _build_options(method, **overrides)
    backend = create_backend(options)
    yield from backend.register_iter(fixed, moving)


def _build_options(
    method: RegistrationMethod, **overrides: object
) -> RegistrationOptions:
    """Build RegistrationOptions from a method preset and optional overrides."""
    field_names = {f.name for f in RegistrationOptions.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    valid_overrides = {k: v for k, v in overrides.items() if k in field_names}
    invalid = set(overrides) - field_names
    if invalid:
        raise TypeError(
            f"Unexpected keyword arguments: {invalid}. "
            f"Valid options: {sorted(field_names)}"
        )
    return RegistrationOptions(method=method, **valid_overrides)  # type: ignore[arg-type]
