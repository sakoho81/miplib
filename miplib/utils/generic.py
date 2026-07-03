from typing import Any


def isiterable(something: Any) -> bool:
    """Return True if the object is iterable, False otherwise."""
    try:
        iter(something)
        return True
    except TypeError:
        return False
