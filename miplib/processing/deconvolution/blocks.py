from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from miplib.processing.ndarray import start_to_stop_idx


@dataclass(frozen=True)
class BlockSpec:
    """A spatial block of an n-dimensional array with padding.

    Attributes
    ----------
    start : ndarray
        Start index of the padded block in the source array.
    size : ndarray
        Size of the padded block (includes padding on all sides).
    pad : int
        Number of padding voxels on each side of the block.
    """

    start: np.ndarray
    size: np.ndarray
    pad: int

    @property
    def inner_start(self) -> np.ndarray:
        return self.start + self.pad

    @property
    def inner_size(self) -> np.ndarray:
        return self.size - 2 * self.pad

    @property
    def padded_slice(self) -> tuple[slice, ...]:
        return start_to_stop_idx(self.start, self.start + self.size)

    @property
    def inner_slice(self) -> tuple[slice, ...]:
        return start_to_stop_idx(self.inner_start, self.inner_start + self.inner_size)


def iter_blocks(
    image_shape: tuple[int, ...],
    n_blocks: int = 1,
    pad: int = 0,
) -> Iterator[BlockSpec]:
    """Yield ``n_blocks ** ndim`` blocks whose inner regions tile *image_shape*.

    Each block carries *pad* pixels of zero-padding on every side.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

    ndim = len(image_shape)
    for coords in itertools.product(range(n_blocks), repeat=ndim):
        inner_start = np.zeros(ndim, dtype=int)
        inner_size = np.zeros(ndim, dtype=int)

        for ax in range(ndim):
            base = image_shape[ax] // n_blocks
            rem = image_shape[ax] % n_blocks
            inner_start[ax] = coords[ax] * base + min(coords[ax], rem)
            inner_size[ax] = base + (1 if coords[ax] < rem else 0)

        padded_start = inner_start - pad
        padded_size = inner_size + 2 * pad

        yield BlockSpec(start=padded_start, size=padded_size, pad=pad)


def extract_padded_block(
    data: np.ndarray,
    block: BlockSpec,
) -> np.ndarray:
    """Extract a padded block from *data*, zero-padding at boundaries."""
    result = np.zeros(block.size.tolist(), dtype=data.dtype)
    src_slice, dst_slice = _compute_src_dst_slices(data.shape, block.start, block.size)
    result[dst_slice] = data[src_slice]
    return result


def _compute_src_dst_slices(
    data_shape: tuple[int, ...],
    block_start: np.ndarray,
    block_size: np.ndarray,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    src_slices = []
    dst_slices = []
    for dim in range(len(data_shape)):
        src_start = max(0, block_start[dim])
        src_end = min(data_shape[dim], block_start[dim] + block_size[dim])
        dst_start = src_start - block_start[dim]
        dst_end = dst_start + (src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))
    return tuple(src_slices), tuple(dst_slices)
