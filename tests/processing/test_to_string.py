import numpy as np

from miplib.processing.to_string import (
    argument_string,
    concatenate_to_csv,
    fix_exp_str,
    float_to_str,
    format_time_string,
    time_to_str,
    tostr,
)


def test_concatenate_to_csv_empty():
    assert concatenate_to_csv(()) == ""
    assert concatenate_to_csv([]) == ""


def test_concatenate_to_csv_single():
    assert concatenate_to_csv([3.0]) == "3.000000"


def test_concatenate_to_csv_multiple():
    assert concatenate_to_csv([1.0, 2.5, 3.14159]) == "1.000000,2.500000,3.141590"


def test_argument_string_str():
    assert argument_string("hello") == "'hello'"


def test_argument_string_int():
    assert argument_string(42) == "42"


def test_argument_string_float():
    assert argument_string(3.14) == "3.14"


def test_argument_string_complex():
    assert argument_string(1 + 2j) == "(1+2j)"


def test_argument_string_tuple_small():
    assert argument_string((1, 2, 3)) == "(1, 2, 3)"


def test_argument_string_tuple_single():
    assert argument_string((42,)) == "(42,)"


def test_argument_string_tuple_large():
    assert argument_string(tuple(range(10))) == "<10-tuple>"


def test_argument_string_list_small():
    assert argument_string([1, 2]) == "[1, 2]"


def test_argument_string_list_large():
    assert argument_string(list(range(10))) == "<10-list>"


def test_argument_string_ndarray():
    arr = np.ones((3, 4))
    assert argument_string(arr) == "<float64 (3, 4)-array>"


def test_argument_string_none():
    assert argument_string(None) == "None"


def test_argument_string_unknown_type():
    result = argument_string(object())
    assert result == "<object>"


def test_format_time_string_zero():
    assert format_time_string(0) == "0:00:00"


def test_format_time_string_one_hour():
    assert format_time_string(3600) == "1:00:00"


def test_format_time_string_one_minute():
    assert format_time_string(61) == "0:01:01"


def test_format_time_string_full():
    assert format_time_string(3661) == "1:01:01"


def test_fix_exp_str_e_plus_00():
    assert fix_exp_str("1.0e+00") == "1.0"


def test_fix_exp_str_e_plus_0():
    assert fix_exp_str("1.0e+03") == "1.0E3"


def test_fix_exp_str_e_minus():
    assert fix_exp_str("1.0e-02") == "1.0E-2"


def test_fix_exp_str_no_exp():
    assert fix_exp_str("0.5") == "0.5"


def test_float_to_str_large():
    assert float_to_str(1234.0) == "1.2E3"


def test_float_to_str_hundreds():
    assert float_to_str(456.0) == "456"


def test_float_to_str_tens():
    assert float_to_str(45.6) == "45.6"


def test_float_to_str_ones():
    assert float_to_str(3.14) == "3.14"


def test_float_to_str_tenths():
    assert float_to_str(0.5) == "0.500"


def test_float_to_str_very_small():
    assert float_to_str(1e-7) == "1.0E-7"


def test_float_to_str_zero():
    assert float_to_str(0.0) == "0.0"


def test_float_to_str_small_magnitude():
    assert float_to_str(0.005) == "5.00E-3"


def test_tostr_float():
    assert tostr(3.14) == "3.14"


def test_tostr_tuple():
    assert tostr((1.0, 2.0)) == ("1.00", "2.00")


def test_tostr_non_float():
    assert tostr("hello") == "hello"


def test_time_to_str_years():
    assert time_to_str(63113851.0) == "1Y11M30d10h29m2.9s"


def test_time_to_str_months():
    assert time_to_str(26300000.0) == "10M42m41.7s"


def test_time_to_str_days():
    assert time_to_str(86400 * 3) == "3d"


def test_time_to_str_hours():
    assert time_to_str(3600 * 5) == "5h"


def test_time_to_str_minutes():
    assert time_to_str(120) == "2m"


def test_time_to_str_seconds():
    assert time_to_str(30.5) == "30.5s"


def test_time_to_str_milliseconds():
    assert time_to_str(0.5) == "500ms"


def test_time_to_str_microseconds():
    assert time_to_str(0.0005) == "500us"


def test_time_to_str_nanoseconds():
    assert time_to_str(0.0000005) == "500ns"


def test_time_to_str_zero():
    assert time_to_str(0.0) == "0"
