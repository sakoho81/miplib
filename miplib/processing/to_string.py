import hashlib
import time

import numpy

VERBOSE = False


def concatenate_to_csv(values: tuple[float, ...] | list[float]) -> str:
    return ",".join(f"{s:.6f}" for s in values)


def argument_string(obj: object) -> str:
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, int | float | complex):
        return str(obj)
    if isinstance(obj, tuple):
        if len(obj) < 2:
            return "({},)".format(", ".join(map(argument_string, obj)))
        if len(obj) < 5:
            return "({})".format(", ".join(map(argument_string, obj)))
        return f"<{len(obj)}-tuple>"
    if isinstance(obj, list):
        if len(obj) < 5:
            return "[{}]".format(", ".join(map(argument_string, obj)))
        return f"<{len(obj)}-list>"
    if isinstance(obj, numpy.ndarray):
        return f"<{obj.dtype} {obj.shape}-array>"
    if obj is None:
        return str(obj)
    return "<" + str(type(obj))[8:-2] + ">"


def time_it(func):
    """Decorator: print how long calling given function took.

    Only active when ``VERBOSE`` is True.
    """
    if not VERBOSE:
        return func

    def new_func(*args, **kws):
        t = time.time()
        r = func(*args, **kws)
        dt = time.time() - t
        print(
            "Calling {}({}) -> {} took {} seconds".format(
                func.__name__,
                ", ".join(map(argument_string, args)),
                argument_string(r),
                dt,
            )
        )
        return r

    return new_func


def format_time_string(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def getalpha(r: int) -> str:
    if r >= len(alphabet):
        return "_" + nary(r - len(alphabet), len(alphabet))
    return alphabet[r]


def nary(number: int | str, base: int = 64) -> str:
    if isinstance(number, str):
        number = eval(number)
    n = number
    s = ""
    while n:
        n1 = n // base
        r = n - n1 * base
        n = n1
        s = getalpha(r) + s
    return s


def encode(string: str | bytes) -> str:
    """Return encoded string."""
    if isinstance(string, str):
        string = string.encode()
    return nary("0x" + hashlib.md5(string).hexdigest())


def fix_exp_str(s: str) -> str:
    return (
        s.replace("e+00", "")
        .replace("e+0", "E")
        .replace("e+", "E")
        .replace("e-0", "E-")
        .replace("e-", "E-")
    )


def float_to_str(x: float) -> str:
    if abs(x) >= 1000:
        return fix_exp_str(f"{x:.1e}")
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    if abs(x) >= 0.1:
        return f"{x:.3f}"
    if abs(x) <= 1e-6:
        return fix_exp_str(f"{x:.1e}")
    if not x:
        return "0"
    return fix_exp_str(f"{x:.2e}")


def tostr(x: object) -> str | tuple[str, ...]:
    """Return pretty string representation of x."""
    if isinstance(x, tuple):
        return tuple(map(tostr, x))
    if isinstance(x, float | numpy.float32 | numpy.float64):
        return float_to_str(x)
    return str(x)


def time_to_str(s: float) -> str:
    """Return human readable time string from seconds.

    Examples
    --------
    >>> print(time_to_str(123000000))
    3Y10M24d10h40m
    >>> print(time_to_str(1230000))
    14d5h40m
    >>> print(time_to_str(1230))
    20m30.0s
    >>> print(time_to_str(0.123))
    123ms
    >>> print(time_to_str(0.000123))
    123us
    >>> print(time_to_str(0.000000123))
    123ns

    """
    seconds_in_year = 31556925.9747  # a standard SI year
    years = int(s / (seconds_in_year))
    r: list[str] = []
    if years:
        r.append(f"{years}Y")
        s -= years * (seconds_in_year)
    months = int(s / (seconds_in_year / 12.0))
    if months:
        r.append(f"{months}M")
        s -= months * (seconds_in_year / 12.0)
    days = int(s / (60 * 60 * 24))
    if days:
        r.append(f"{days}d")
        s -= days * 60 * 60 * 24
    hours = int(s / (60 * 60))
    if hours:
        r.append(f"{hours}h")
        s -= hours * 60 * 60
    minutes = int(s / 60)
    if minutes:
        r.append(f"{minutes}m")
        s -= minutes * 60
    seconds = int(s)
    if seconds:
        r.append(f"{s:.1f}s")
    elif not r:
        mseconds = int(s * 1000)
        if mseconds:
            r.append(f"{mseconds}ms")
        else:
            useconds = int(s * 1000000)
            if useconds:
                r.append(f"{useconds}us")
            else:
                nseconds = int(s * 1000000000)
                if nseconds:
                    r.append(f"{nseconds}ns")
    if not r:
        return "0"
    return "".join(r)


time2str = time_to_str
