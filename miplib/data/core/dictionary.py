class FixedDictionary(object):
    """
    A dictionary with immutable keys. Is initialized at construction
    with a list of key values.
    """
    def __init__(self, keys):
        assert isinstance(keys, list) or isinstance(keys, tuple)
        self._dictionary = dict.fromkeys(keys)

    def __setitem__(self, key, value):
        if key not in self._dictionary:
            raise KeyError("The key {} is not defined".format(key))
        else:
            self._dictionary[key] = value

    def __getitem__(self, key):
        return self._dictionary[key]

    @property
    def keys(self):
        return list(self._dictionary.keys())

    @property
    def contents(self):
        return list(self._dictionary.keys()), list(self._dictionary.values())