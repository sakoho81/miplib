import pytest

from miplib.utils.generic import isiterable


def test_isiterable_list_tuple_dict_set():
    assert isiterable([1, 2, 3])
    assert isiterable((1, 2))
    assert isiterable({"a": 1})
    assert isiterable({1, 2, 3})


def test_isiterable_string():
    assert isiterable("hello")


def test_isiterable_generator_and_range():
    assert isiterable(x for x in range(3))
    assert isiterable(range(10))


def test_isiterable_custom_iter():
    class WithIter:
        def __iter__(self):
            return iter([1, 2])

    assert isiterable(WithIter())


@pytest.mark.parametrize("value", [42, 3.14, None, True, False])
def test_isiterable_non_iterable(value):
    assert not isiterable(value)


def test_isiterable_plain_object():
    class Plain:
        pass

    assert not isiterable(Plain())
