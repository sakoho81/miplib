from __future__ import annotations

from pathlib import Path

import numpy as np

from miplib.processing.deconvolution.types import FirstEstimate


def create_estimate(
    source,
    strategy: FirstEstimate = FirstEstimate.IMAGE_MEAN,
    *,
    constant: float = 1.0,
    path: str | Path | None = None,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Allocate and initialise an estimate array for RL deconvolution.

    If *path* is given, the array is backed by a memory-mapped file.
    """
    shape = source.shape

    if path is not None:
        out: np.ndarray = np.memmap(str(path), dtype=dtype, mode="w+", shape=shape)
    else:
        out = np.zeros(shape, dtype=dtype)

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
