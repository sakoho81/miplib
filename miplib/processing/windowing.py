
import numpy as np
from scipy.signal import tukey


def _nd_window(data, filter_function, **kwargs):
    """
    Performs on N-dimensional spatial-domain data.
    This is done to mitigate boundary effects in the FFT.

    Parameters
    ----------
    data : ndarray
           Input data to be windowed, modified in place.
    filter_function : 1D window generation function
           Function should accept one argument: the window length.
           Example: scipy.signal.hamming
    """
    result = data.copy().astype(np.float64)
    for axis, axis_size in enumerate(data.shape):
        # set up shape for numpy broadcasting
        filter_shape = [1, ] * data.ndim
        filter_shape[axis] = axis_size
        window = filter_function(axis_size, **kwargs).reshape(filter_shape)
        # scale the window intensities to maintain array intensity
        np.power(window, (1.0/data.ndim), out=window)
        result *= window
    return result


def apply_hamming_window(data):
    """
    Apply Hamming window to data

    :param data (np.ndarray): An N-dimensional Numpy array to be used in Windowing
    :return:
    """
    assert issubclass(data.__class__, np.ndarray)

    return _nd_window(data, np.hamming)


def apply_tukey_window(data, alpha=0.25, sym=True):
    """
    Apply Tukey window to data

    :param data (np.ndarray): An N-dimensional Numpy array to be used in Windowing
    :return:
    """
    assert issubclass(data.__class__, np.ndarray)

    return _nd_window(data, tukey, alpha=alpha, sym=sym)
