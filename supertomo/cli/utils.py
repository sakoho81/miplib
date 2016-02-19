"""
Various utilities that are used to convert command line parameters into
data types that the progrma understands.
"""

import numpy

def float2dtype(float_type):
    """Return numpy float dtype object from float type label.
    """
    if float_type == 'single' or float_type is None:
        return numpy.float32
    if float_type == 'double':
        return numpy.float64
    raise NotImplementedError (`float_type`)
