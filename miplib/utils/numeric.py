import numpy as np


def find_next_power_of_2(number):
    """ A simple utility to find the closest power of two
    :arg number: a non-zero numeric value
    """
    power = np.ceil(np.log2(number))
    return 2**power
