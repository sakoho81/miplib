from collections.abc import Callable

import numpy as np
from scipy.signal.windows import tukey


def _nd_window(
    data: np.ndarray, filter_function: Callable, **kwargs: float | bool
) -> np.ndarray:
    """Apply an N-dimensional window by multiplying 1D windows along each axis."""
    result = data.copy().astype(np.float64)
    for axis, axis_size in enumerate(data.shape):
        filter_shape = [
            1,
        ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        np.power(window, (1.0 / data.ndim), out=window)
        result *= window
    return result


def apply_hamming_window(data: np.ndarray) -> np.ndarray:
    """Apply a Hamming window to N-dimensional data."""
    return _nd_window(data, np.hamming)


def apply_tukey_window(
    data: np.ndarray, alpha: float = 0.25, sym: bool = True
) -> np.ndarray:
    """Apply a Tukey window to N-dimensional data."""
    return _nd_window(data, tukey, alpha=alpha, sym=sym)
