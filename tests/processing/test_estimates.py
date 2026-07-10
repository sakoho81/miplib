from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from miplib.data.adapters.image_data import ArrayDataSource
from miplib.data.containers.image import Image
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
    npt.assert_allclose(est, source.get_full_image(0), rtol=1e-6)


def test_image_mean():
    source = _make_source()
    est = create_estimate(source, FirstEstimate.IMAGE_MEAN)
    npt.assert_allclose(est, source.get_full_image(0).mean())


def test_average():
    source = _make_source(n_views=3, seed=1)
    est = create_estimate(source, FirstEstimate.AVERAGE)
    expected = (
        source.get_full_image(0) + source.get_full_image(1) + source.get_full_image(2)
    ) / 3
    npt.assert_allclose(est, expected, rtol=1e-6)


def test_sum():
    source = _make_source(n_views=3, seed=2)
    est = create_estimate(source, FirstEstimate.SUM)
    expected = (
        source.get_full_image(0) + source.get_full_image(1) + source.get_full_image(2)
    )
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
