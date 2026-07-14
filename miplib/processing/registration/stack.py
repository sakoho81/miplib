"""Registration of slices within an image stack."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import fourier_shift

from miplib.data.containers.image import Image

from . import register
from .options import RegistrationMethod


def register_stack_slices(
    stack: Image, method: RegistrationMethod = RegistrationMethod.PHASE_CORRELATION
) -> np.ndarray:
    """Register adjacent slices in a 3D stack.

    Args:
        stack: A 3D image stack.
        method: Registration method for slice-to-slice alignment.

    Returns:
        Array of shape ``(n_slices, 2)`` with (y, x) shifts in pixels
        for each slice relative to slice 0.
    """
    if not isinstance(stack, Image):
        raise TypeError(f"Expected Image, got {type(stack).__name__}")
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {stack.ndim}D")

    shifts = np.zeros((stack.shape[0], 2), dtype=np.float64)

    for f_idx, m_idx in zip(
        range(0, stack.shape[0] - 1), range(1, stack.shape[0]), strict=False
    ):
        fixed = Image(stack[f_idx], stack.spacing[1:])
        moving = Image(stack[m_idx], stack.spacing[1:])

        offset = register(fixed, moving, method)
        # register returns ITK-ordered (x, y) params; convert to numpy (y, x)
        shifts[m_idx] = shifts[f_idx] + np.asarray(offset.GetParameters())[::-1]

    return shifts


def register_stack_slices_with_reference(
    stack: Image,
    fixed: Image,
    method: RegistrationMethod = RegistrationMethod.PHASE_CORRELATION,
) -> np.ndarray:
    """Register each slice in a 3D stack to a provided reference.

    Args:
        stack: A 3D image stack.
        fixed: Reference 2D image.
        method: Registration method.

    Returns:
        Array of shape ``(n_slices, 2)`` with (y, x) shifts in pixels.
    """
    if not isinstance(stack, Image):
        raise TypeError(f"Expected Image, got {type(stack).__name__}")
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {stack.ndim}D")

    shifts = np.zeros((stack.shape[0], 2), dtype=np.float64)

    for i in range(stack.shape[0]):
        moving = Image(stack[i], stack.spacing[1:])
        offset = register(fixed, moving, method)
        # ITK (x, y) → numpy (y, x)
        shifts[i] = np.asarray(offset.GetParameters())[::-1]

    return shifts


def shift_stack_slices(stack: Image, shifts: np.ndarray) -> Image:
    """Shift each slice in a 3D stack.

    Args:
        stack: A 3D image stack.
        shifts: Array of shape ``(n_slices, 2)`` with (y, x) shifts in pixels.

    Returns:
        Shift-aligned 3D image.
    """
    if not isinstance(stack, Image):
        raise TypeError(f"Expected Image, got {type(stack).__name__}")
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {stack.ndim}D")
    if stack.shape[0] != shifts.shape[0]:
        raise ValueError("Shift array does not match stack depth")

    resampled = Image(np.zeros_like(stack), spacing=stack.spacing)

    for idx, image_slice in enumerate(stack):
        resampled[idx] = np.abs(
            np.fft.ifftn(fourier_shift(np.fft.fftn(image_slice), shifts[idx])).real
        )

    return resampled
