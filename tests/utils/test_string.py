from miplib.utils.string import common_start, common_string


def test_common_start_identical():
    assert common_start("hello", "hello") == "hello"


def test_common_start_partial_prefix():
    assert common_start("hello_world", "hello_there") == "hello_"


def test_common_start_no_common():
    assert common_start("abc", "xyz") == ""


def test_common_start_one_empty():
    assert common_start("", "abc") == ""
    assert common_start("abc", "") == ""


def test_common_start_both_empty():
    assert common_start("", "") == ""


def test_common_start_single_char_match():
    assert common_start("a", "a") == "a"


def test_common_start_single_char_mismatch():
    assert common_start("a", "b") == ""


def test_common_string_plain_strings_with_prefix():
    assert common_string(["hello_world", "hello_there"]) == "hello_"


def test_common_string_four_strings_shortest_first_mismatch():
    result = common_string(["abc", "abx", "aby"])
    assert result == "ab"


def test_common_string_identical():
    assert common_string(["abc", "abc", "abc"]) == "abc"


def test_common_string_no_common():
    assert common_string(["foo", "bar"]) == ""


def test_common_string_paths_with_common_basename_prefix():
    assert common_string(["/a/b/cat", "/x/y/car"]) == "ca"


def test_common_string_one_path_one_plain():
    assert common_string(["/a/b/hello", "help"]) == "hel"


def test_common_string_single_string():
    assert common_string(["hello"]) == "hello"


def test_common_string_empty_string_in_list():
    assert common_string(["abc", ""]) == ""


def test_common_string_empty_list():
    assert common_string([]) == ""
