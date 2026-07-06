import logging
from collections.abc import Sequence
from functools import reduce

import numpy as np

logger = logging.getLogger(__name__)


def nroot(array: np.ndarray | float, n: float) -> np.ndarray | float:
    """Return the n-th root of each element in array."""
    return array ** (1.0 / n)


def normalize(array: np.ndarray) -> np.ndarray:
    """Normalize by dividing each element by the array sum."""
    return safe_divide(array, array.sum())


def float2dtype(float_type: str | None) -> type[np.float32] | type[np.float64]:
    """Return numpy float dtype object from float type label."""
    if float_type == "single" or float_type is None:
        return np.float32
    if float_type == "double":
        return np.float64
    raise NotImplementedError(repr(float_type))


def contract_to_shape(data: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Remove symmetric padding to fit data to given shape."""
    if not all(a <= b for a, b in zip(shape, data.shape)):
        raise ValueError("All target shape dimensions must be <= data dimensions")

    if any(x != y for x, y in zip(shape, data.shape, strict=False)):
        slices = []
        for s1, s2 in zip(data.shape, shape, strict=False):
            slices.append(slice((s1 - s2) // 2, (s1 + s2) // 2))

        image = data[tuple(slices)]
    else:
        image = data

    return image


def expand_to_shape(
    data: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype | None = None,
    background: float | None = None,
) -> np.ndarray:
    """Expand data to given shape by zero-padding."""
    if dtype is None:
        dtype = data.dtype

    start_index = np.array(shape) - data.shape
    data_start = np.negative(start_index.clip(max=0))
    data = cast_to_dtype(data, dtype, rescale=False)
    if data.ndim == 3:
        data = data[data_start[0] :, data_start[1] :, data_start[2] :]
    else:
        data = data[data_start[0] :, data_start[1] :]

    if background is None:
        background = 0

    if tuple(shape) != data.shape:
        expanded_data = np.zeros(shape, dtype=dtype) + background
        slices = []
        rhs_slices = []
        for s1, s2 in zip(shape, data.shape, strict=False):
            a, b = (s1 - s2 + 1) // 2, (s1 + s2 + 1) // 2
            c, d = 0, s2
            while a < 0:
                a += 1
                b -= 1
                c += 1
                d -= 1
            slices.append(slice(a, b))
            rhs_slices.append(slice(c, d))
        try:
            expanded_data[tuple(slices)] = data[tuple(rhs_slices)]
        except ValueError:
            logger.debug("data.shape: %s, shape: %s", data.shape, shape)
            raise
        return expanded_data
    else:
        return data


def mul_seq(seq: Sequence[float]) -> float:
    """Return the product of a sequence of numbers."""
    return reduce(lambda x, y: x * y, seq, 1)


def cast_to_dtype(
    data: np.ndarray,
    dtype: np.dtype,
    rescale: bool = True,
    remove_outliers: bool = False,
) -> np.ndarray:
    """Cast an array to a new dtype with optional rescaling.

    Handles dynamic range differences better than .astype() alone.
    """
    if data.dtype == dtype:
        return data

    if "int" in str(dtype):
        data_info = np.iinfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    elif "float" in str(dtype):
        data_info = np.finfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    else:
        data_max = data.max()
        data_min = data.min()
        logger.warning("Casting into unknown data type; detail clipping may occur")

    # In case of unsigned integers, numbers below zero need to be clipped
    if "uint" in str(dtype):
        data_max = 255
        data_min = 0

    if remove_outliers:
        data = data.clip(0, np.percentile(data, 99.99))

    if rescale is True:
        return rescale_to_min_max(data, data_min, data_max).astype(dtype)
    else:
        return data.clip(data_min, data_max).astype(dtype)


def rescale_to_min_max(
    data: np.ndarray, data_min: float, data_max: float
) -> np.ndarray:
    """Rescale data intensities to the range [data_min, data_max]."""
    # Return array with max value in the original data scaled to correct
    # range
    if abs(data.max()) > abs(data.min()) or data_min == 0:
        return data_max / data.max() * data
    else:
        return data_min / data.min() * data


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide arrays, coercing division-by-zero results to zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
        result[result == np.inf] = 0.0
        return np.nan_to_num(result)


def start_to_stop_idx(start: np.ndarray, stop: np.ndarray) -> tuple[slice, ...]:
    """Generate an n-dimensional start-to-stop slice for each axis."""
    return tuple(slice(a, b) for a, b in zip(start, stop, strict=False))


def start_to_offset_idx(start: np.ndarray, offset: np.ndarray) -> tuple[slice, ...]:
    """Generate an n-dimensional slicing structure from start indices and offsets."""
    stop = start + offset
    return tuple(slice(a, b) for a, b in zip(start, stop, strict=False))


def reverse_array(array: np.ndarray) -> np.ndarray:
    """Reverse an array by flipping along every axis."""
    temp = array.copy()
    for i in range(temp.ndim):
        temp = np.flip(temp, i)

    return temp


def first_order_derivative_2d(array: np.ndarray) -> np.ndarray:
    """Calculate the first order derivative magnitude of a 2D array."""
    d1 = np.vstack([np.zeros((1, array.shape[1])), np.diff(array, axis=0)])
    d2 = np.hstack([np.zeros((array.shape[0], 1)), np.diff(array, axis=1)])
    return d1**2 + d2**2


def get_rounded_kernel(diameter: int) -> np.ndarray:
    """Make a rounded kernel of the given diameter for filtering."""
    dd = np.linspace(-1, 1, diameter)
    xx1, yy1 = np.meshgrid(dd, dd)
    rr = np.sqrt(xx1**2 + yy1**2)

    kernel = np.zeros((diameter,) * 2)
    kernel[rr < 1] = 1

    return kernel


def center_of_mass(
    xx: np.ndarray,
    yy: np.ndarray,
    array: np.ndarray,
    threshold: float = 0.0,
) -> tuple[float, float]:
    """Calculate the center of mass on a meshgrid."""

    if threshold > 0.0:
        array = array.copy()
        array[array < threshold] = 0

    xsum = (xx * array).sum()
    ysum = (yy * array).sum()
    mass = array.sum()

    return xsum / mass, ysum / mass
