import pytest

from miplib.data.core.dictionary import FixedDictionary


def test_set_get_item():
    fd = FixedDictionary(("key1", "key2", "key3"))
    fd["key1"] = 23
    assert fd["key1"] == 23


def test_set_wrong_key_raises_keyerror():
    fd = FixedDictionary(("key1", "key2"))
    with pytest.raises(KeyError, match="not defined"):
        fd["bad"] = 42


def test_get_unset_key_returns_none():
    """dict.fromkeys sets all values to None initially."""
    fd = FixedDictionary(("key1",))
    assert fd["key1"] is None


def test_keys_returns_all_defined_keys():
    fd = FixedDictionary(("a", "b", "c"))
    assert sorted(fd.keys) == ["a", "b", "c"]


def test_contents_returns_keys_and_values():
    fd = FixedDictionary(("a", "b"))
    fd["a"] = 1
    fd["b"] = 2
    keys, values = fd.contents
    assert set(keys) == {"a", "b"}
    assert set(values) == {1, 2}


def test_construction_rejects_string():
    with pytest.raises(TypeError, match="list or tuple"):
        FixedDictionary("not-a-list")


def test_construction_rejects_int():
    with pytest.raises(TypeError, match="list or tuple"):
        FixedDictionary(42)


def test_overwrite_existing_key():
    fd = FixedDictionary(("key1",))
    fd["key1"] = 100
    fd["key1"] = 200
    assert fd["key1"] == 200
