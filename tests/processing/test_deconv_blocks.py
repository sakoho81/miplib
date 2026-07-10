from __future__ import annotations

import itertools

import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing.deconvolution.blocks import (
    BlockSpec,
    _compute_src_dst_slices,
    extract_padded_block,
    iter_blocks,
)


def test_block_spec_properties():
    start = np.array([10, 20])
    size = np.array([30, 40])
    pad = 4
    block = BlockSpec(start=start, size=size, pad=pad)

    npt.assert_array_equal(block.inner_start, np.array([14, 24]))
    npt.assert_array_equal(block.inner_size, np.array([22, 32]))
    assert block.padded_slice == (slice(10, 40), slice(20, 60))
    assert block.inner_slice == (slice(14, 36), slice(24, 56))


def test_iter_blocks_invalid():
    with pytest.raises(ValueError, match="n_blocks must be >= 1"):
        list(iter_blocks((10, 10), 0))


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_iter_blocks_tiles_image(n_blocks):
    shape = (50, 40)
    coverage = np.zeros(shape, dtype=int)

    for block in iter_blocks(shape, n_blocks, pad=0):
        coverage[block.inner_slice] += 1

    assert (coverage == 1).all(), f"{n_blocks} blocks didn't tile {shape} exactly once"


@pytest.mark.parametrize("n_blocks", [2, 3])
def test_iter_blocks_correct_count(n_blocks):
    shape = (101, 81)
    blocks = list(iter_blocks(shape, n_blocks, pad=0))
    assert len(blocks) == n_blocks * n_blocks


def test_iter_blocks_iterator_is_lazy():
    it = iter_blocks((100, 100), 4, pad=0)
    assert len(list(itertools.islice(it, 2))) == 2


def test_extract_padded_block_interior():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    block = BlockSpec(start=np.array([2, 2]), size=np.array([6, 6]), pad=1)

    result = extract_padded_block(data, block)

    assert result.shape == (6, 6)
    npt.assert_array_equal(result[1:5, 1:5], data[3:7, 3:7])


def test_extract_padded_block_negative_start():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    block = BlockSpec(start=np.array([-3, -3]), size=np.array([8, 8]), pad=3)

    result = extract_padded_block(data, block)

    assert result.shape == (8, 8)
    npt.assert_array_equal(result[:3, :], 0)
    npt.assert_array_equal(result[:, :3], 0)
    npt.assert_array_equal(result[3:8, 3:8], data[:5, :5])


def test_extract_padded_block_beyond_end():
    data = np.arange(100, dtype=np.float64).reshape(10, 10)
    block = BlockSpec(start=np.array([8, 8]), size=np.array([5, 5]), pad=1)

    result = extract_padded_block(data, block)

    assert result.shape == (5, 5)
    npt.assert_array_equal(result[:2, :2], data[8:10, 8:10])
    npt.assert_array_equal(result[2:, :], 0)
    npt.assert_array_equal(result[:, 2:], 0)


def test_compute_src_dst_slices_negative_start():
    src, dst = _compute_src_dst_slices((10, 10), np.array([-3, -3]), np.array([8, 8]))
    assert src == (slice(0, 5), slice(0, 5))
    assert dst == (slice(3, 8), slice(3, 8))
