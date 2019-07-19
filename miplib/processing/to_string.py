import hashlib
import sys
import time

import numpy

VERBOSE = False


def concatenate_to_csv(values):
    assert isinstance(values, tuple) or \
           isinstance(values, list)

    return ",".join('%.6f' % s for s in values)


def argument_string(obj):
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, (int, float, complex)):
        return str(obj)
    if isinstance(obj, tuple):
        if len(obj) < 2: return '(%s,)' % (', '.join(map(argument_string, obj)))
        if len(obj) < 5: return '(%s)' % (', '.join(map(argument_string, obj)))
        return '<%s-tuple>' % (len(obj))
    if isinstance(obj, list):
        if len(obj) < 5: return '[%s]' % (', '.join(map(argument_string, obj)))
        return '<%s-list>' % (len(obj))
    if isinstance(obj, numpy.ndarray):
        return '<%s %s-array>' % (obj.dtype, obj.shape)
    if obj is None:
        return str(obj)
    return '<' + str(type(obj))[8:-2] + '>'


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
        r = func(*args, **kws)
        dt = time.time() - t
        print('Calling %s(%s) -> %s took %s seconds' % \
              (func.__name__, ', '.join(map(argument_string, args)), argument_string(r), dt))
        return r

    return new_func


def format_time_string(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


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

    def __init__(self, minValue=0, maxValue=100, totalWidth=80, prefix='',
                 show_percentage=True):
        self.show_percentage = show_percentage
        self.progBar = self.progBar_last = "[]"  # This holds the progress bar string
        self.min = minValue
        self.max = maxValue
        self.span = maxValue - minValue or 1
        self.width = totalWidth
        self.amount = 0  # When amount == max, we are 100% done
        self.start_time = self.current_time = self.prev_time = time.time()
        self.starting_amount = None
        self.updateAmount(0)  # Build progress bar string
        self.prefix = prefix
        self.comment = self.comment_last = ''

    def updateComment(self, comment):
        self.comment = comment

    def updateAmount(self, newAmount=0):
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
            self.progBar = "[>%s]" % (' ' * (allFull - 1))
        elif numHashes == allFull:
            self.progBar = "[%s]" % ('=' * allFull)
        else:
            self.progBar = "[%s>%s]" % ('=' * (numHashes - 1),
                                        ' ' * (allFull - numHashes))

        if self.show_percentage:
            # figure out where to put the percentage, roughly centered
            percentPlace = (len(self.progBar) / 2) - len(str(percentDone))
            percentString = str(percentDone) + "%"
        else:
            percentPlace = int((len(self.progBar) / 2) - len(str(percentDone)))
            percentString = '%s/%s' % (self.amount, self.span)
        # slice the percentage into the bar
        self.progBar = ''.join([self.progBar[0:percentPlace], percentString,
                                self.progBar[percentPlace + len(percentString):]])
        if self.starting_amount is not None:
            amount_diff = self.amount - self.starting_amount
            if amount_diff:
                self.prev_time = self.current_time
                self.current_time = time.time()
                elapsed = self.current_time - self.starting_time
                eta = elapsed * (self.max - self.amount) / float(amount_diff)
                self.progBar += ' ETA:' + time_to_str(eta)

    def __str__(self):
        return str(self.progBar)

    def __call__(self, value):
        """ Updates the amount, and writes to stdout. Prints a carriage return
            first, so it will overwrite the current line in stdout."""
        self.updateAmount(value)
        if self.progBar_last == self.progBar and self.comment == self.comment_last:
            return
        print('\r', end=' ')
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
        if name not in self.__dict__ and '_counter' in self.__dict__:
            self._counter += 1
        self.__dict__[name] = obj

    def iterNameValue(self):
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            yield k, v

    def copy(self, **kws):
        r = self.__class__(self._descr + ' - a copy')
        for name, value in self.iterNameValue():
            setattr(r, name, value)
        for name, value in list(kws.items()):
            setattr(r, name, value)
        return r


options = Holder('Options')

alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def getalpha(r):
    if r >= len(alphabet):
        return '_' + nary(r - len(alphabet), len(alphabet))
    return alphabet[r]


def nary(number, base=64):
    if isinstance(number, str):
        number = eval(number)
    n = number
    s = ''
    while n:
        n1 = n // base
        r = n - n1 * base
        n = n1
        s = getalpha(r) + s
    return s


def encode(string):
    """ Return encoded string.
    """
    return nary('0x' + hashlib.md5(string).hexdigest())


def fix_exp_str(s):
    return s.replace('e+00', '').replace('e+0', 'E').replace('e+', 'E').replace('e-0', 'E-').replace('e-', 'E-')


def float_to_str(x):
    if abs(x) >= 1000: return fix_exp_str('%.1e' % x)
    if abs(x) >= 100: return '%.0f' % x
    if abs(x) >= 10: return '%.1f' % x
    if abs(x) >= 1: return '%.2f' % x
    if abs(x) >= .1: return '%.3f' % x
    if abs(x) <= 1e-6: return fix_exp_str('%.1e' % x)
    if not x: return '0'
    return fix_exp_str('%.2e' % x)


def tostr(x):
    """ Return pretty string representation of x.

    Parameters
    ----------
    x : {tuple, float, :numpy:.float32, :numpy:.float64}
    """
    if isinstance(x, tuple):
        return tuple(map(tostr, x))
    if isinstance(x, (float, numpy.float32, numpy.float64)):
        return float_to_str(x)
    return str(x)


def time_to_str(s):
    """ Return human readable time string from seconds.

    Examples
    --------
    >>> from miplib.processing.to_string import time_to_str
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
    seconds_in_year = 31556925.9747  # a standard SI year
    orig_s = s
    years = int(s / (seconds_in_year))
    r = []
    if years:
        r.append('%sY' % (years))
        s -= years * (seconds_in_year)
    months = int(s / (seconds_in_year / 12.0))
    if months:
        r.append('%sM' % (months))
        s -= months * (seconds_in_year / 12.0)
    days = int(s / (60 * 60 * 24))
    if days:
        r.append('%sd' % (days))
        s -= days * 60 * 60 * 24
    hours = int(s / (60 * 60))
    if hours:
        r.append('%sh' % (hours))
        s -= hours * 60 * 60
    minutes = int(s / 60)
    if minutes:
        r.append('%sm' % (minutes))
        s -= minutes * 60
    seconds = int(s)
    if seconds:
        r.append('%.1fs' % (s))
        s -= seconds
    elif not r:
        mseconds = int(s * 1000)
        if mseconds:
            r.append('%sms' % (mseconds))
            s -= mseconds / 1000
        elif not r:
            useconds = int(s * 1000000)
            if useconds:
                r.append('%sus' % (useconds))
                s -= useconds / 1000000
            elif not r:
                nseconds = int(s * 1000000000)
                if nseconds:
                    r.append('%sns' % (nseconds))
                    s -= nseconds / 1000000000
    if not r:
        return '0'
    return ''.join(r)


time2str = time_to_str