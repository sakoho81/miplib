from unittest import TestCase

from ..dictionary import FixedDictionary


class TestFixedDictionary(TestCase):
    def test_set_get_item(self):
        dictionary = FixedDictionary(("key1", "key2", "key3"))

        dictionary["key1"] = 23

        self.assertEqual(dictionary["key1"], 23)

    def test_set_wrong_item(self):
        dictionary = FixedDictionary(("key1", "key2", "key3"))
        with self.assertRaises(KeyError):
            dictionary["key5"] = 25

    def test_contents(self):
        dictionary = FixedDictionary(("key1", "key2", "key3"))

        dictionary["key2"] = 23
        dictionary["key1"] = "temp"
        dictionary["key3"] = (1, 2, 3)

        keys, values = dictionary.contents

        self.assertListEqual(keys, ['key3', 'key2', 'key1'])
        self.assertListEqual(values, [(1, 2, 3), 23, 'temp'])





