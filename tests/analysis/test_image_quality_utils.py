import numpy as np
import pytest

from miplib.analysis.image_quality.utils import (
    analyze_accumulation,
    calculate_entropy,
)

# -- analyze_accumulation ------------------------------------------------------


def test_accumulation_uniform():
    """For uniform data, each element contributes equally — the index
    should be ceil(fraction * len(data))."""
    x = np.ones(10)
    assert analyze_accumulation(x, 0.3) == 3
    assert analyze_accumulation(x, 0.5) == 5


def test_accumulation_ascending():
    """Largest values at tail dominate — fewer elements needed."""
    x = np.array([1, 2, 3, 4, 5])
    assert analyze_accumulation(x, 0.35) == 2  # need 4+5=9/15 to reach 0.35 threshold
    assert analyze_accumulation(x, 0.9) == 4  # need 2+3+4+5=14/15


def test_accumulation_descending():
    """Smallest values at tail — many elements needed."""
    x = np.array([5, 4, 3, 2, 1])
    assert analyze_accumulation(x, 0.35) == 3  # need 3 tail elements


def test_accumulation_fraction_one():
    x = np.array([1.0, 2.0])
    assert analyze_accumulation(x, 1.0) == 2


def test_accumulation_single_element():
    assert analyze_accumulation(np.array([42.0]), 0.5) == 1
    assert analyze_accumulation(np.array([42.0]), 1.0) == 1


def test_accumulation_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="fraction must be"):
        analyze_accumulation(np.array([1, 2]), 0.0)
    with pytest.raises(ValueError, match="fraction must be"):
        analyze_accumulation(np.array([1, 2]), 1.5)
    with pytest.raises(ValueError, match="fraction must be"):
        analyze_accumulation(np.array([1, 2]), -0.1)


# -- calculate_entropy ---------------------------------------------------------


def test_entropy_constant_signal_is_zero():
    """A constant signal has only one histogram bin → entropy = 0."""
    data = np.ones(100) * 5.0
    assert calculate_entropy(data) == pytest.approx(0.0, abs=1e-10)


def test_entropy_two_bins_maximum():
    """Equal counts of two values → entropy = 1 bit."""
    data = np.array([0.0] * 50 + [1.0] * 50)
    assert calculate_entropy(data) == pytest.approx(1.0, abs=1e-6, rel=1e-6)


def test_entropy_nonnegative():
    rng = np.random.default_rng(42)
    data = rng.normal(size=200)
    assert calculate_entropy(data) >= 0.0


def test_entropy_deterministic():
    """Same data produces the same entropy value."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=500)
    assert calculate_entropy(data) == pytest.approx(calculate_entropy(data))
