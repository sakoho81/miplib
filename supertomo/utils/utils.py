"""
Various utilities.
"""

__autodoc__ = ['expand_to_shape', 'contract_to_shape', 'ProgressBar', 'Options', 'encode',
               'tostr', 'get_path_dir', 'float2dtype', 'time_to_str', 'time_it',
               'time2str', 'bytes2str', 'sround', 'mfloat']

import os
import sys
import time
import hashlib

import numpy
import optparse
import numpy.testing.utils as numpy_utils

file_extensions = ['.tif', '.lsm', 'tiff', '.raw', '.data']


VERBOSE = False

def argument_string(obj):
    if isinstance(obj, (str, )):
        return repr(obj)
    if isinstance(obj, (int, float, complex)):
        return str(obj)
    if isinstance(obj, tuple):
        if len(obj)<2: return '(%s,)' % (', '.join(map(argument_string, obj)))
        if len(obj)<5: return '(%s)' % (', '.join(map(argument_string, obj)))
        return '<%s-tuple>' % (len(obj))
    if isinstance(obj, list):
        if len(obj)<5: return '[%s]' % (', '.join(map(argument_string, obj)))
        return '<%s-list>' % (len(obj))
    if isinstance(obj, numpy.ndarray):
        return '<%s %s-array>' % (obj.dtype, obj.shape)
    if obj is None:
        return str(obj)
    return '<'+str(type(obj))[8:-2]+'>'

def time_it(func):
    """Decorator: print how long calling given function took.

    Notes
    ----- 
    ``iocbio.utils.VERBOSE`` must be True for this decorator to be
    effective.
    """
    if not VERBOSE:
        return func
    def new_func(*args, **kws):
        t = time.time()
        r = func (*args, **kws)
        dt = time.time() - t
        print 'Calling %s(%s) -> %s took %s seconds' % \
            (func.__name__, ', '.join(map(argument_string, args)), argument_string(r), dt)
        return r
    return new_func

class ProgressBar:
    """Creates a text-based progress bar. 

    Call the object with the ``print`` command to see the progress bar,
    which looks something like this::

        [=======>        22%                  ]

    You may specify the progress bar's width, min and max values on
    init. For example::
    
      bar = ProgressBar(N)
      for i in range(N):
        print bar(i)
      print bar(N)
        
    References
    ----------
    http://code.activestate.com/recipes/168639/

    See also
    --------
    __init__, updateComment
    """

    def __init__(self, minValue = 0, maxValue = 100, totalWidth=80, prefix='',
                 show_percentage = True):
        self.show_percentage = show_percentage
        self.progBar = self.progBar_last = "[]"   # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue or 1
        self.width = totalWidth
        self.amount = 0       # When amount == max, we are 100% done
        self.start_time = self.current_time = self.prev_time = time.time()
        self.starting_amount = None
        self.updateAmount(0)  # Build progress bar string
        self.prefix = prefix
        self.comment = self.comment_last = ''


    def updateComment(self, comment):
        self.comment = comment

    def updateAmount(self, newAmount = 0):
        """ Update the progress bar with the new amount (with min and max
            values set at initialization; if it is over or under, it takes the
            min or max value as a default. """
        if newAmount and self.starting_amount is None:
            self.starting_amount = newAmount
            self.starting_time = time.time()
        if newAmount < self.min: newAmount = self.min
        if newAmount > self.max: newAmount = self.max
        self.prev_amount = self.amount
        self.amount = newAmount

        # Figure out the new percent done, round to an integer
        diffFromMin = float(self.amount - self.min)
        percentDone = (diffFromMin / float(self.span)) * 100.0
        percentDone = int(round(percentDone))

        # Figure out how many hash bars the percentage should be
        allFull = self.width - 2
        numHashes = (percentDone / 100.0) * allFull
        numHashes = int(round(numHashes))

        # Build a progress bar with an arrow of equal signs; special cases for
        # empty and full

        if numHashes == 0:
            self.progBar = "[>%s]" % (' '*(allFull-1))
        elif numHashes == allFull:
            self.progBar = "[%s]" % ('='*allFull)
        else:
            self.progBar = "[%s>%s]" % ('='*(numHashes-1),
                                        ' '*(allFull-numHashes))
        
        if self.show_percentage:
            # figure out where to put the percentage, roughly centered
            percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
            percentString = str(percentDone) + "%"
        else:
            percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
            percentString = '%s/%s' % (self.amount, self.span)
        # slice the percentage into the bar
        self.progBar = ''.join([self.progBar[0:percentPlace], percentString,
                                self.progBar[percentPlace+len(percentString):]
                                ])
        if self.starting_amount is not None:
            amount_diff = self.amount - self.starting_amount
            if amount_diff:
                self.prev_time = self.current_time
                self.current_time = time.time()
                elapsed = self.current_time - self.starting_time
                eta = elapsed * (self.max - self.amount)/float(amount_diff)
                self.progBar += ' ETA:'+time_to_str(eta)

    def __str__(self):
        return str(self.progBar)

    def __call__(self, value):
        """ Updates the amount, and writes to stdout. Prints a carriage return
            first, so it will overwrite the current line in stdout."""
        self.updateAmount(value)
        if self.progBar_last == self.progBar and self.comment==self.comment_last:
            return
        print '\r',
        sys.stdout.write(self.prefix + str(self) + str(self.comment) + ' ')
        sys.stdout.flush()
        self.progBar_last = self.progBar
        self.comment_last = self.comment

class Holder:
    """ Holds pairs ``(name, value)`` as instance attributes.
    
    The set of Holder pairs is extendable by

    ::
    
      <Holder instance>.<name> = <value>

    and the values are accessible as

    ::

      value = <Holder instance>.<name>
    """
    def __init__(self, descr):
        self._descr = descr
        self._counter = 0

    def __str__(self):
        return self._descr % (self.__dict__)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, str(self))

    def __getattr__(self, name):
        raise AttributeError('%r instance has no attribute %r' % (self, name))
    
    def __setattr__(self, name, obj):
        if not self.__dict__.has_key(name) and self.__dict__.has_key('_counter'):
            self._counter += 1
        self.__dict__[name] = obj

    def iterNameValue(self):
        for k,v in self.__dict__.iteritems():
            if k.startswith('_'):
                continue
            yield k,v

    def copy(self, **kws):
        r = self.__class__(self._descr+' - a copy')
        for name, value in self.iterNameValue():
            setattr(r, name, value)
        for name, value in kws.items():
            setattr(r, name, value)
        return r

options = Holder('Options')

alphabet='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def getalpha(r):
    if r>=len(alphabet):
        return '_'+nary(r-len(alphabet),len(alphabet))
    return alphabet[r]

def nary(number, base=64):
    if isinstance(number, str):
        number = eval(number)
    n = number
    s = ''
    while n:
        n1 = n // base
        r = n - n1*base
        n = n1
        s = getalpha(r) + s
    return s

def encode(string):
    """ Return encoded string.
    """
    return nary('0x'+hashlib.md5(string).hexdigest())


def fix_exp_str(s):
    return s.replace ('e+00','').replace('e+0','E').replace ('e+','E').replace ('e-0','E-').replace ('e-','E-')

def float_to_str(x):
    if abs(x)>=1000: return fix_exp_str('%.1e' % x)
    if abs(x)>=100: return '%.0f' % x
    if abs(x)>=10: return '%.1f' % x
    if abs(x)>=1: return '%.2f' % x
    if abs(x)>=.1: return '%.3f' % x
    if abs(x)<=1e-6: return fix_exp_str ('%.1e' % x)
    if not x: return '0'
    return fix_exp_str('%.2e' % x)

def tostr (x):
    """ Return pretty string representation of x.

    Parameters
    ----------
    x : {tuple, float, :numpy:.float32, :numpy:.float64}
    """
    if isinstance (x, tuple):
        return tuple ( map (tostr, x))
    if isinstance(x, (float, numpy.float32,numpy.float64)):
        return float_to_str(x)
    return str(x)

def time_to_str(s):
    """ Return human readable time string from seconds.

    Examples
    --------
    >>> from iocbio.utils import time_to_str
    >>> print time_to_str(123000000)
    3Y10M24d10h40m
    >>> print time_to_str(1230000)
    14d5h40m
    >>> print time_to_str(1230)
    20m30.0s
    >>> print time_to_str(0.123)
    123ms
    >>> print time_to_str(0.000123)
    123us
    >>> print time_to_str(0.000000123)
    123ns

    """
    seconds_in_year = 31556925.9747 # a standard SI year
    orig_s = s
    years = int(s / (seconds_in_year))
    r = []
    if years:
        r.append ('%sY' % (years))
        s -= years * (seconds_in_year)
    months = int(s / (seconds_in_year/12.0))
    if months:
        r.append ('%sM' % (months))
        s -= months * (seconds_in_year/12.0)
    days = int(s / (60*60*24))
    if days:
        r.append ('%sd' % (days))
        s -= days * 60*60*24
    hours = int(s / (60*60))
    if hours:
        r.append ('%sh' % (hours))
        s -= hours * 60*60
    minutes = int(s / 60)
    if minutes:
        r.append ('%sm' % (minutes))
        s -= minutes * 60
    seconds = int(s)
    if seconds:
        r.append ('%.1fs' % (s))
        s -= seconds
    elif not r:
        mseconds = int(s*1000)
        if mseconds:
            r.append ('%sms' % (mseconds))
            s -= mseconds / 1000
        elif not r:
            useconds = int(s*1000000)
            if useconds:
                r.append ('%sus' % (useconds))
                s -= useconds / 1000000
            elif not r:
                nseconds = int(s*1000000000)
                if nseconds:
                    r.append ('%sns' % (nseconds))
                    s -= nseconds / 1000000000
    if not r:
        return '0'
    return ''.join(r)

time2str = time_to_str


def cast_to_dtype(data, dtype, rescale=True, remove_outliers=False):
    """
     A function for casting a numpy array into a new data type.
    The .astype() property of Numpy sometimes produces satisfactory
    results, but if the data type to cast into has a more limited
    dynamic range than the original data type, problems may occur.

    :param data:            a numpy.array object
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
        data_info = numpy.iinfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    elif 'float' in str(dtype):
        data_info = numpy.finfo(dtype)
        data_max = data_info.max
        data_min = data_info.min
    else:
        data_max = data.max()
        data_min = data.min()
        print "Warning casting into unknown data type. Detail clipping" \
              "may occur"

    # In case of unsigned integers, numbers below zero need to be clipped
    if 'uint' in str(dtype):
        data_max = 255
        data_min = 0

    if remove_outliers:
        data = data.clip(0, numpy.percentile(data, 99.99))

    if rescale is True:
        return rescale_to_min_max(data, data_min, data_max).astype(dtype)
    else:
        return data.clip(data_min, data_max).astype(dtype)


def rescale_to_min_max(data, data_min, data_max):
    """
    A function to rescale image intensities to range, define by
    data_min and data_max input parameters.

    :param data:        Input image (Numpy array)
    :param data_min:    Minimum pixel value. Can be any type of a number
                        (preferably of the same type with the data.dtype)
    :param data_max:    Maximum pixel value
    :return:            Return the rescaled array
    """
    # Return array with max value in the original image scaled to correct
    # range
    if abs(data.max()) > abs(data.min()) or data_min == 0:
        return data_max / data.max()*data
    else:
        return data_min / data.min()*data

def expand_to_shape(data, shape, dtype=None, background=None):
    """
    Expand data to given shape by zero-padding.
    """
    if dtype is None:
        dtype = data.dtype

    data = cast_to_dtype(data, dtype, rescale=False)

    if background is None:
        background = data.min()

    if shape != data.shape:
        expanded_data = numpy.zeros(shape, dtype=dtype) + background
        slices = []
        rhs_slices = []
        for s1, s2 in zip (shape, data.shape):
            a, b = (s1-s2+1)//2, (s1+s2+1)//2
            c, d = 0, s2
            while a<0:
                a += 1
                b -= 1
                c += 1
                d -= 1
            slices.append(slice(a, b))
            rhs_slices.append(slice(c, d))
        try:
            expanded_data[tuple(slices)] = data[tuple(rhs_slices)]
        except ValueError:
            print data.shape, shape
            raise
        return expanded_data
    else:
        return data


def contract_to_shape(data, shape):
    """
    Remove padding from input data array. The function
    expects the padding to be symmetric on all sides
    """

    if shape != data.shape:
        slices = []
        for s1, s2 in zip(data.shape, shape):
            slices.append(slice((s1-s2)//2, (s1+s2)//2))

        image = data[tuple(slices)]
    else:
        image = data

    return image


def mul_seq(seq):
    return reduce(lambda x, y: x*y, seq, 1)

def float2dtype(float_type):
    """Return numpy float dtype object from float type label.
    """
    if float_type == 'single' or float_type is None:
        return numpy.float32
    if float_type == 'double':
        return numpy.float64
    raise NotImplementedError (`float_type`)


def check_path(path, prefix):
    """
    :param path:    Path to a file (string)
    :param prefix:  Path prefix, if applicable. Used in cases in
                    which the path argument is not an absolute
                    path
    :return:        Returns the absolute path, if the file is found,
                    None otherwise
    """
    if not os.path.isfile(path):
        path = os.path.join(prefix, path)
        if not os.path.isfile(path):
                print 'Not a valid file %s' % path
                return None
        return path


def get_path_dir(path, suffix):
    """ Return a directory name with suffix that will be used to save data
    related to given path.
    """
    if os.path.isfile(path):
        path_dir = path+'.'+suffix
    elif os.path.isdir(path):
        path_dir = os.path.join(path, suffix)
    elif os.path.exists(path):
        raise ValueError ('Not a file or directory: %r' % path)
    else:
        base, ext = os.path.splitext(path)
        if ext in file_extensions:
            path_dir = path +'.'+suffix
        else:
            path_dir = os.path.join(path, suffix)
    return path_dir


def nroot(array, n):
    """

    :param array:   A n dimensional numpy array by default. Of course this works
                    with single numbers and whatever the interpreter can understand
    :param n:       The root - a number
    :return:
    """
    return array**(1.0/n)

def normalize(array):
    return array / array.sum()
