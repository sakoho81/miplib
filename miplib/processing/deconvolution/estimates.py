from __future__ import annotations

from enum import Enum

import numpy as np

from miplib.processing.deconvolution.blocks import BlockSpec


class FirstEstimate(Enum):
    CONSTANT = "constant"
    IMAGE = "image"
    IMAGE_MEAN = "image_mean"
    AVERAGE = "average"
    SUM = "sum"


def create_estimate(
    source,
    strategy: FirstEstimate = FirstEstimate.IMAGE_MEAN,
    *,
    constant: float = 1.0,
    out: np.ndarray | None = None,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Allocate and initialise an estimate array for RL deconvolution.

    Parameters
    ----------
    source
        Duck-typed data source with ``.shape`` and ``.get_image_block()``.
    strategy
        Initialisation strategy (default: ``IMAGE_MEAN``).
    constant
        Value used when ``strategy == CONSTANT``.
    out
        Pre-allocated array to fill (must match ``source.shape``).
    dtype
        Data type for newly allocated arrays.
    """
    shape = source.shape

    if out is None:
        out = np.zeros(shape, dtype=dtype)
    elif out.shape != shape:
        raise ValueError(f"out shape {out.shape} does not match source shape {shape}")

    if strategy == FirstEstimate.CONSTANT:
        out[:] = constant
    elif strategy == FirstEstimate.IMAGE:
        blk = _full_block(shape)
        out[:] = source.get_image_block(0, blk)
    elif strategy == FirstEstimate.IMAGE_MEAN:
        blk = _full_block(shape)
        out[:] = source.get_image_block(0, blk).mean()
    elif strategy == FirstEstimate.AVERAGE:
        out[:] = 0
        blk = _full_block(shape)
        for v in range(source.n_views):
            out += source.get_image_block(v, blk)
        out /= source.n_views
    elif strategy == FirstEstimate.SUM:
        out[:] = 0
        blk = _full_block(shape)
        for v in range(source.n_views):
            out += source.get_image_block(v, blk)

    return out


def _full_block(shape: tuple[int, ...]) -> BlockSpec:
    start = np.zeros(len(shape), dtype=int)
    size = np.array(shape, dtype=int)
    return BlockSpec(start=start, size=size, pad=0)
