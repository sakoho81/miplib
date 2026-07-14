from dataclasses import is_dataclass
from typing import TypeVar

from dacite import Config, DaciteError, from_dict

T = TypeVar("T")


def options_from_dict(data_class: type[T], data: dict | None) -> T:
    """Parse a dictionary into a dataclass instance, or return defaults if None.

    Args:
        data_class: The dataclass type to instantiate
        data: Dictionary with field values, or None to use defaults

    Returns:
        Instance of *data_class* with values from *data* merged onto defaults

    Raises:
        ValueError: If the dict contains invalid fields or values for the dataclass
        TypeError: If *data_class* is not a dataclass
    """
    if data is None:
        return data_class()

    if not is_dataclass(data_class):
        raise TypeError(f"{data_class.__name__} is not a dataclass")

    try:
        return from_dict(data_class=data_class, data=data, config=Config(strict=True))
    except DaciteError as e:
        raise ValueError(f"Invalid options for {data_class.__name__}: {e}") from e
