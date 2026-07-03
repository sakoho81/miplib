import numpy as np
import pytest

from miplib.utils.numeric import find_next_power_of_2


def test_find_next_power_of_2_exact():
    assert find_next_power_of_2(1) == 1.0
    assert find_next_power_of_2(2) == 2.0
    assert find_next_power_of_2(4) == 4.0
    assert find_next_power_of_2(8) == 8.0
    assert find_next_power_of_2(16) == 16.0
    assert find_next_power_of_2(1024) == 1024.0


def test_find_next_power_of_2_rounds_up():
    assert find_next_power_of_2(3) == 4.0
    assert find_next_power_of_2(5) == 8.0
    assert find_next_power_of_2(9) == 16.0
    assert find_next_power_of_2(17) == 32.0
    assert find_next_power_of_2(1023) == 1024.0


def test_find_next_power_of_2_fractional():
    assert find_next_power_of_2(0.5) == 0.5
    assert find_next_power_of_2(0.25) == 0.25
    assert find_next_power_of_2(0.75) == 1.0
    assert find_next_power_of_2(0.1) == 0.125


def test_find_next_power_of_2_zero():
    assert find_next_power_of_2(0.0) == 0.0


@pytest.mark.parametrize("value", [-1.0, -100.0])
def test_find_next_power_of_2_negative(value):
    assert np.isnan(find_next_power_of_2(value))
