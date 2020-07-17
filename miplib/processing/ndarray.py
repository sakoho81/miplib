import numpy as np
from functools import reduce

def nroot(array, n):
    """

    :param array:   A n dimensional numpy array by default. Of course this works
                    with single numbers and whatever the interpreter can understand
    :param n:       The root - a number
    :return:
    """
    return array ** (1.0 / n)


def normalize(array):
    """
    Normalizes a numpy array by dividing each element with the array.sum()

    :param array: a numpy.array
    :return:
    """
    return array / array.sum()


def float2dtype(float_type):
    """Return numpy float dtype object from float type label.
    """
    if float_type == 'single' or float_type is None:
        return np.float32
    if float_type == 'double':
        return np.float64
    raise NotImplementedError (repr(float_type))


def contract_to_shape(data, shape):
    """
    Remove padding from input data array. The function
    expects the padding to be symmetric on all sides
    """
    assert all(x <= y for x,y in zip(shape, data.shape))

    if any(x != y for x,y in zip(shape, data.shape)):

        slices = []
        for s1, s2 in zip(data.shape, shape):
            slices.append(slice((s1 - s2) // 2, (s1 + s2) // 2))

        image = data[tuple(slices)]
    else:
        image = data

    return image


def expand_to_shape(data, shape, dtype=None, background=None):
    """
    Expand data to given shape by zero-padding.
    """
    if dtype is None:
        dtype = data.dtype

    start_index = np.array(shape) - data.shape
    data_start = np.negative(start_index.clip(max=0))
    data = cast_to_dtype(data, dtype, rescale=False)
    if data.ndim == 3:
        data = data[data_start[0]:, data_start[1]:, data_start[2]:]
    else:
        data = data[data_start[0]:, data_start[1]:]

    if background is None:
        background = 0

    if tuple(shape) != data.shape:
        expanded_data = np.zeros(shape, dtype=dtype) + background
        slices = []
        rhs_slices = []
        for s1, s2 in zip(shape, data.shape):
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
            print(data.shape, shape)
            raise
        return expanded_data
    else:
        return data


def mul_seq(seq):
    return reduce(lambda x, y: x * y, seq, 1)


def float2dtype(float_type):
    """Return numpy float dtype object from float type label.
    """
    if float_type == 'single' or float_type is None:
        return np.float32
    if float_type == 'double':
        return np.float64
    raise NotImplementedError(repr(float_type))


def cast_to_dtype(data, dtype, rescale=True, remove_outliers=False):
    """
     A function for casting a numpy array into a new data type.
    The .astype() property of Numpy sometimes produces satisfactory
    results, but if the data type to cast into has a more limited
    dynamic range than the original data type, problems may occur.

    :param data:            a np.array object
    :param dtype:           data type string, as in Python
    :param rescale:         switch to enable rescaling pixel
                            values to the new dynamic range.
                            This should always be enabled when
                            scaling to a more limited range,
                            e.g. from float to int
    :param remove_outliers: sometimes deconvolution/fusion generates
                            bright artifacts, which interfere with
                            the rescaling calculation. You can remove them
                            with this switch
    :return:                Returns the input data, cast into the new datatype
    """
    if data.dtype == dtype:
        return data

    if 'int' in str(dtype):
        data_info = np.iinfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    elif 'float' in str(dtype):
        data_info = np.finfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    else:
        data_max = data.max()
        data_min = data.min()
        print("Warning casting into unknown data type. Detail clipping" \
              "may occur")

    # In case of unsigned integers, numbers below zero need to be clipped
    if 'uint' in str(dtype):
        data_max = 255
        data_min = 0

    if remove_outliers:
        data = data.clip(0, np.percentile(data, 99.99))

    if rescale is True:
        return rescale_to_min_max(data, data_min, data_max).astype(dtype)
    else:
        return data.clip(data_min, data_max).astype(dtype)


def rescale_to_min_max(data, data_min, data_max):
    """
    A function to rescale data intensities to range, define by
    data_min and data_max input parameters.

    :param data:        Input data (Numpy array)
    :param data_min:    Minimum pixel value. Can be any type of a number
                        (preferably of the same type with the data.dtype)
    :param data_max:    Maximum pixel value
    :return:            Return the rescaled array
    """
    # Return array with max value in the original data scaled to correct
    # range
    if abs(data.max()) > abs(data.min()) or data_min == 0:
        return data_max / data.max() * data
    else:
        return data_min / data.min() * data

def safe_divide(numerator, denominator):
    """
    Division of numpy arrays that can handle division by zero. NaN results are
    coerced to zero. Also suppresses the division by zero warning.
    :param numerator:
    :param denominator:
    :return:
    """
    with np.errstate(divide="ignore"):
        result = numerator / denominator
        result[result == np.inf] = 0.0
        return np.nan_to_num(result)


def start_to_stop_idx(start, stop):
    """
    Generate n-dimensional indexing strucure for a numpy array,
    consisting of a start-to-stop slice in each dimension
    :param start: start indexes
    :param stop: stop indexes
    :return:
    """
    return tuple(slice(a, b) for a, b in zip(start, stop))


def start_to_offset_idx(start, offset):
    """
    Generate n-dimensional indexing structure for a numpy array,
    based on start indexes and offsets
    :param start: list of indexes to start the slicing from
    :param offset: list of slice lengths
    :return:
    """
    stop = start + offset
    return tuple(slice(a, b) for a, b in zip(start, stop))


def reverse_array(array):

    assert isinstance(array, np.ndarray)

    temp = array.copy()
    for i in range(temp.ndim):
        temp = np.flip(temp, i)

    return temp


def first_order_derivative_2d(array):
    """
    Calculates the first order (a[i]-a[i+1]) derivative of a 2D array
    :param array: a 2D numeric array
    :type array: np.ndarray
    """
    d1 = np.vstack([np.zeros((1, array.shape[1])), np.diff(array, axis=0)])
    d2 = np.hstack([np.zeros((array.shape[0], 1)), np.diff(array, axis=1)])
    return d1 ** 2 + d2 ** 2


def get_rounded_kernel(diameter):
    """
    Makes a rounded kernel of a desired size for filtering operations
    :param size:
    :return:
    """
    dd = np.linspace(-1, 1, diameter)
    xx1, yy1 = np.meshgrid(dd, dd)
    rr = np.sqrt(xx1 ** 2 + yy1 ** 2)

    kernel = np.zeros((diameter,)*2)
    kernel[rr < 1] = 1

    return kernel


def center_of_mass(xx, yy, array, threshold=0.0):
    """
    A small utility calculate the center of mass on a meshgrid
    :param xx: the x coordinates of the meshgrid
    :param yy: the y coordinates of the meshgrid
    :param array: an array with numeric values
    :param threshold: a threshold value  that can be used to exclude certain
    array elements from the calculation.
    :return: the x,y coordinates of the center of mass
    """

    if threshold > 0.0:
        array = array.copy()
        array[array < threshold] = 0

    xsum = (xx * array).sum()
    ysum = (yy * array).sum()
    mass = array.sum()

    return xsum / mass, ysum / mass
