from __future__ import annotations

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


def calculate_block_layout(
    image_shape: tuple[int, ...],
    n_blocks: int = 1,
    pad: int = 0,
) -> list[BlockSpec]:
    """Split each axis of *image_shape* into *n_blocks* pieces, producing
    ``n_blocks ** ndim`` BlockSpecs whose inner (unpadded) regions tile the
    image without gaps or overlaps.
    """
    if n_blocks < 1:
        raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")

    ndim = len(image_shape)
    blocks: list[BlockSpec] = []

    for coords in np.ndindex(*([n_blocks] * ndim)):
        inner_start = np.zeros(ndim, dtype=int)
        inner_size = np.zeros(ndim, dtype=int)

        for ax in range(ndim):
            base = image_shape[ax] // n_blocks
            rem = image_shape[ax] % n_blocks
            inner_start[ax] = coords[ax] * base + min(coords[ax], rem)
            inner_size[ax] = base + (1 if coords[ax] < rem else 0)

        padded_start = inner_start - pad
        padded_size = inner_size + 2 * pad

        blocks.append(BlockSpec(start=padded_start, size=padded_size, pad=pad))

    return blocks


def extract_padded_block(
    data: np.ndarray,
    block: BlockSpec,
) -> np.ndarray:
    """Extract a padded block from *data*, zero-padding where the block extends
    beyond the array boundaries.
    """
    result = np.zeros(block.size.tolist(), dtype=data.dtype)
    src_slice, dst_slice = _compute_src_dst_slices(data.shape, block.start, block.size)
    result[dst_slice] = data[src_slice]
    return result


def _compute_src_dst_slices(
    data_shape: tuple[int, ...],
    block_start: np.ndarray,
    block_size: np.ndarray,
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Compute source and destination slices for padded block extraction."""
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


def reconstruct_from_blocks(
    blocks: list[BlockSpec],
    block_results: list[np.ndarray],
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """Reconstruct a full image from per-block results by copying the inner
    (unpadded) region of each block result into the output array.

    The block inner regions must tile *out_shape* exactly.
    """
    result = np.zeros(out_shape, dtype=block_results[0].dtype)
    for block, block_data in zip(blocks, block_results):
        p = block.pad
        inner_data = block_data[tuple(slice(p, p + s) for s in block.inner_size)]
        result[block.inner_slice] = inner_data
    return result
