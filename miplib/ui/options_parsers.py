from itertools import chain


def parseRangeList(rngs):
    """
    This parser type was created to enable the input of numeric ranges, such
    as "2, 5, 7-11, 26". It returns a sorted list of integers.
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
        return range(start, end + 1)

    return sorted(set(chain(*[parse_range(rng) for rng in rngs.split(',')])))


def parseFromToString(string):
    return list(int(i) for i in string.split("to"))


def parseCommaSeparatedList(string):
    return sorted(int(i)for i in string.split(','))


def ensure_positive(number):
    try:
        number = float(number)
    except ValueError:
        print("You must enter a number")
    if number <= 0:
        raise ValueError("The value should be greater than zero")

    return number