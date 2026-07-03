from miplib.utils.string import common_start, common_string


class TestCommonStart:
    def test_identical(self):
        assert common_start("hello", "hello") == "hello"

    def test_partial_prefix(self):
        assert common_start("hello_world", "hello_there") == "hello_"

    def test_no_common(self):
        assert common_start("abc", "xyz") == ""

    def test_one_empty(self):
        assert common_start("", "abc") == ""
        assert common_start("abc", "") == ""

    def test_both_empty(self):
        assert common_start("", "") == ""

    def test_single_char_match(self):
        assert common_start("a", "a") == "a"

    def test_single_char_mismatch(self):
        assert common_start("a", "b") == ""


class TestCommonString:
    def test_plain_strings_with_prefix(self):
        assert common_string(["hello_world", "hello_there"]) == "hello_"

    def test_three_strings(self):
        result = common_string(["abc_def", "abc_ghi", "abc_jkl"])
        assert result == "abc_"

    def test_four_strings_shortest_first_mismatch(self):
        result = common_string(["abc", "abx", "aby"])
        assert result == "ab"

    def test_identical(self):
        assert common_string(["abc", "abc", "abc"]) == "abc"

    def test_no_common(self):
        assert common_string(["foo", "bar"]) == ""

    def test_paths_with_common_basename_prefix(self):
        assert common_string(["/a/b/cat", "/x/y/car"]) == "ca"

    def test_paths_no_common_basename(self):
        assert common_string(["/a/b/foo", "/x/y/bar"]) == ""

    def test_one_path_one_plain(self):
        assert common_string(["/a/b/hello", "help"]) == "hel"

    def test_single_string(self):
        assert common_string(["hello"]) == "hello"

    def test_empty_string_in_list(self):
        assert common_string(["abc", ""]) == ""

    def test_empty_list(self):
        assert common_string([]) == ""
