from typing import Any


class FixedDictionary:
    """A dictionary with immutable keys set at construction."""

    def __init__(self, keys: list[str] | tuple[str, ...]) -> None:
        if not isinstance(keys, (list, tuple)):
            raise TypeError(f"keys must be a list or tuple, got {type(keys).__name__}")
        self._dictionary = dict.fromkeys(keys)

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._dictionary:
            raise KeyError(f"The key {key} is not defined")
        self._dictionary[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._dictionary[key]

    def __contains__(self, key: str) -> bool:
        return key in self._dictionary

    def __iter__(self):
        return iter(self._dictionary)

    @property
    def keys(self) -> list[str]:
        return list(self._dictionary.keys())

    @property
    def contents(self) -> tuple[list[str], list[Any]]:
        return list(self._dictionary.keys()), list(self._dictionary.values())
