import os
import argparse

from itertools import chain

def parse_range_list(rngs):
    """ This parser type was created to enable the input of numeric ranges, 
    such as "2, 5, 7-11, 26". It returns a sorted list of integers.
    
    Arguments:
        rngs {string} -- A string containing comma delimited list of ranges, 
        e.g. "2, 5, 7-11, 26". A range should consist of a start and end (3-6)
        or a single integer number. 
    
    Raises:
        ValueError: Raised if a bad range, e.g. 7-11-3 is given. 
    
    Returns:
        tuple -- A tuple with the selected indices. The above range for example,
        "2, 5, 7-11, 26"  will generate a tuple(2, 5, 7, 8, 9, 10, 11, 26)
    """
    def parse_range(rng):
        parts = rng.split('-')
        if 1 > len(parts) > 2:
            raise ValueError("Bad range: '%s'" % (rng,))
        parts = [int(i) for i in parts]
        start = parts[0]
        end = start if len(parts) == 1 else parts[1]
        if start > end:
            end, start = start, end
        return list(range(start, end + 1))

    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))


def parseFromToString(string):
    return list(int(i) for i in string.split("to"))


def ensure_positive(number):
    """ Check that a positive number is inserted
    
    Arguments:
        number {string} -- a string of a number, may be float or an int
    
    Raises:
        argparse.ArgumentTypeError: raises an error if argument is not a number
        or if the number is negative
    
    Returns:
        float -- returns the number as as a float
    """
    try:
        number = float(number)
    except ValueError:
        msg = "You must enter a number"
        raise argparse.ArgumentTypeError(msg)
    if number <= 0:
        raise argparse.ArgumentTypeError("The value should be greater than zero")

    return number

    import os
import argparse

def parse_int_tuple(string):
    """ Converts a string of comma separated integer digits into a tuple of ints

    Arguments:
    string {string} -- The input string

    Returns:
    tuple -- A tuple of integers
    """
    
    return tuple(int(i) for i in string.split(','))


def parse_float_tuple(string):
    """ Converts a string of comma separated floating point numbers 
    (. for decimal) into a tuple of floating point numbers.
    
    Arguments:
        string {string} -- The input string
    
    Returns:
        tuple -- The tuple of floats.
    """
    return tuple(float(i) for i in string.split(','))

def parse_is_dir(dirname):
    """ Checks if a path is an actual directory

    
    Arguments:
        dirname {string} -- A path to a directory
    
    Raises:
        argparse.ArgumentTypeError: Raises a parse error if the directory does
        not exist.
    
    Returns:
        string -- Returns the directory, if it exists.
    """
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname
