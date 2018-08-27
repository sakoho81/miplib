from numba import vectorize


@vectorize(['complex64(complex64, complex64)'], target='cuda')
def cuda_complex_div(a, b):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64
    :param  b

    Returns
    -------

    a/b

    """

    return a / b


@vectorize(['complex64(complex64, complex64)'], target='parallel')
def complex_div(a, b):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64
    :param  b

    Returns
    -------

    a/b

    """

    return a / b


@vectorize(['complex64(complex64)'], target='parallel')
def complex_squared(a):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64

    Returns
    -------

    a**2

    """

    return a**2

@vectorize(['complex64(complex64)'], target='cuda')
def complex_squared_cuda(a):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64

    Returns
    -------

    a**2

    """

    return a**2


@vectorize(['complex64(complex64)'], target='parallel')
def complex_mul(a,b):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64

    Returns
    -------

    a**2

    """

    return a*b

@vectorize(['complex64(complex64)'], target='cuda')
def complex_mul_cuda(a,b):
    """
    Implements array division on GPU

    Parameters
    ----------
    :param  a  Two Numpy arrays of the same shape, dtype=numpy.complex64

    Returns
    -------

    a**2

    """

    return a*b