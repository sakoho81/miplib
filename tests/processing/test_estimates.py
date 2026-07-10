from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
from miplib.processing.deconvolution.blocks import BlockSpec
from miplib.processing.deconvolution.estimates import FirstEstimate, create_estimate


def _make_source(n_views=1, shape=(32, 32), seed=42):
    rng = np.random.default_rng(seed)
    images = [
        Image(
            rng.normal(loc=10, scale=2, size=shape).astype(np.float64),
            spacing=(1.0, 1.0),
        )
        for _ in range(n_views)
    ]
    return ArrayDataSource(images)


def _full_block(shape):
    return BlockSpec(
        start=np.zeros(len(shape), dtype=int), size=np.array(shape, dtype=int), pad=0
    )


def test_constant():
    source = _make_source()
    est = create_estimate(source, FirstEstimate.CONSTANT, constant=5.0)
    npt.assert_array_equal(est, 5.0)


def test_constant_default_value():
    source = _make_source()
    est = create_estimate(source, FirstEstimate.CONSTANT)
    npt.assert_array_equal(est, 1.0)


def test_image():
    source = _make_source()
    est = create_estimate(source, FirstEstimate.IMAGE)
    fwd = source.get_image_block(0, _full_block(source.shape))
    npt.assert_allclose(est, fwd, rtol=1e-6)  # float32→float64 conversion


def test_image_mean():
    source = _make_source()
    est = create_estimate(source, FirstEstimate.IMAGE_MEAN)
    fwd = source.get_image_block(0, _full_block(source.shape))
    npt.assert_allclose(est, fwd.mean(), rtol=1e-6)


def test_average():
    source = _make_source(n_views=3, seed=1)
    est = create_estimate(source, FirstEstimate.AVERAGE)
    fwd0 = source.get_image_block(0, _full_block(source.shape))
    fwd1 = source.get_image_block(1, _full_block(source.shape))
    fwd2 = source.get_image_block(2, _full_block(source.shape))
    expected = (fwd0 + fwd1 + fwd2) / 3
    npt.assert_allclose(est, expected, rtol=1e-6)


def test_sum():
    source = _make_source(n_views=3, seed=2)
    est = create_estimate(source, FirstEstimate.SUM)
    fwd0 = source.get_image_block(0, _full_block(source.shape))
    fwd1 = source.get_image_block(1, _full_block(source.shape))
    fwd2 = source.get_image_block(2, _full_block(source.shape))
    expected = fwd0 + fwd1 + fwd2
    npt.assert_allclose(est, expected, rtol=1e-6)


def test_out_parameter():
    source = _make_source()
    out = np.zeros(source.shape, dtype=np.float64)
    result = create_estimate(source, FirstEstimate.CONSTANT, constant=3.0, out=out)
    assert result is out
    npt.assert_array_equal(out, 3.0)


def test_out_shape_mismatch():
    source = _make_source()
    out = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        create_estimate(source, FirstEstimate.CONSTANT, out=out)


def test_default_strategy():
    source = _make_source()
    est = create_estimate(source)
    assert est.shape == source.shape
    assert est.min() > 0
