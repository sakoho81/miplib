import numpy as np


def find_next_power_of_2(number: int | float) -> float:
    """Return the smallest power of two that is >= *number*."""
    power = np.ceil(np.log2(number))
    return 2**power
