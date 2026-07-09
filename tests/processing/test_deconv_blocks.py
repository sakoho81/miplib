from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.processing.deconvolution.blocks import (
    BlockSpec,
    _compute_src_dst_slices,
    calculate_block_layout,
    extract_padded_block,
    reconstruct_from_blocks,
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


def test_calculate_block_layout_invalid():
    with pytest.raises(ValueError, match="n_blocks must be >= 1"):
        calculate_block_layout((10, 10), 0)


@pytest.mark.parametrize("n_blocks", [1, 2, 3])
def test_block_layout_tiles_image(n_blocks):
    shape = (50, 40)
    blocks = calculate_block_layout(shape, n_blocks, pad=0)

    coverage = np.zeros(shape, dtype=int)
    for block in blocks:
        coverage[block.inner_slice] += 1

    assert (coverage == 1).all(), f"{n_blocks} blocks didn't tile {shape} exactly once"


@pytest.mark.parametrize("n_blocks", [2, 3])
def test_block_layout_inner_regions_dont_overlap(n_blocks):
    shape = (101, 81)
    blocks = calculate_block_layout(shape, n_blocks, pad=0)

    coverage = np.zeros(shape, dtype=int)
    for i, block in enumerate(blocks):
        coverage[block.inner_slice] += 1

    assert (coverage <= 1).all()


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


def test_reconstruct_from_blocks():
    shape = (20, 20)
    blocks = calculate_block_layout(shape, 2, pad=0)

    block_results = [
        np.full(block.size.tolist(), i, dtype=np.float64)
        for i, block in enumerate(blocks)
    ]

    result = reconstruct_from_blocks(blocks, block_results, shape)

    assert result.shape == shape
    npt.assert_array_equal(result[:10, :10], 0)
    npt.assert_array_equal(result[:10, 10:], 1)
    npt.assert_array_equal(result[10:, :10], 2)
    npt.assert_array_equal(result[10:, 10:], 3)


def test_reconstruct_roundtrip():
    rng = np.random.default_rng(42)
    original = rng.normal(size=(32, 32)).astype(np.float64)

    for n_blocks in (1, 2, 4):
        blocks = calculate_block_layout((32, 32), n_blocks, pad=0)
        block_results = [extract_padded_block(original, block) for block in blocks]
        result = reconstruct_from_blocks(blocks, block_results, (32, 32))
        npt.assert_array_equal(
            result,
            original,
            err_msg=f"Roundtrip failed for n_blocks={n_blocks}",
        )


def test_reconstruct_roundtrip_with_padding():
    rng = np.random.default_rng(43)
    original = rng.normal(size=(40, 40)).astype(np.float64)

    blocks = calculate_block_layout((40, 40), 2, pad=4)
    block_results = [extract_padded_block(original, block) for block in blocks]
    result = reconstruct_from_blocks(blocks, block_results, (40, 40))
    npt.assert_array_equal(result, original)
