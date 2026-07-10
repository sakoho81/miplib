from __future__ import annotations

import numpy as np

from miplib.processing.deconvolution.types import FirstEstimate


def create_estimate(
    source,
    strategy: FirstEstimate = FirstEstimate.IMAGE_MEAN,
    *,
    constant: float = 1.0,
    out: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Allocate and initialise an estimate array for RL deconvolution."""
    shape = source.shape

    if out is None:
        out = np.zeros(shape, dtype=dtype)
    elif out.shape != shape:
        raise ValueError(f"out shape {out.shape} does not match source shape {shape}")

    if strategy == FirstEstimate.CONSTANT:
        out[:] = constant
    elif strategy == FirstEstimate.IMAGE:
        out[:] = source.get_full_image(0)
    elif strategy == FirstEstimate.IMAGE_MEAN:
        out[:] = source.get_full_image(0).mean()
    elif strategy == FirstEstimate.AVERAGE:
        out[:] = 0
        for v in range(source.n_views):
            out += source.get_full_image(v)
        out /= source.n_views
    elif strategy == FirstEstimate.SUM:
        out[:] = 0
        for v in range(source.n_views):
            out += source.get_full_image(v)

    return out
