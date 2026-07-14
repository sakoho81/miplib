from dataclasses import dataclass

import pytest

from miplib.ui.cli.miplib_entry_point_options import get_quality_options
from miplib.utils.dataclasses import options_from_dict


@dataclass
class _TestOptions:
    name: str = "default"
    count: int = 42
    enabled: bool = True
    ratio: float = 0.5


# --- options_from_dict ---


def test_options_from_dict_none_returns_defaults():
    opts = options_from_dict(_TestOptions, None)
    assert opts.name == "default"
    assert opts.count == 42
    assert opts.enabled is True
    assert opts.ratio == 0.5


def test_options_from_dict_partial_overrides():
    opts = options_from_dict(_TestOptions, {"name": "custom", "count": 99})
    assert opts.name == "custom"
    assert opts.count == 99
    assert opts.enabled is True
    assert opts.ratio == 0.5


def test_options_from_dict_full_overrides():
    opts = options_from_dict(
        _TestOptions, {"name": "x", "count": 1, "enabled": False, "ratio": 0.9}
    )
    assert opts.name == "x"
    assert opts.count == 1
    assert opts.enabled is False
    assert opts.ratio == 0.9


def test_options_from_dict_unknown_field():
    with pytest.raises(ValueError, match="Invalid options"):
        options_from_dict(_TestOptions, {"unknown": "field"})


def test_options_from_dict_wrong_type():
    with pytest.raises(ValueError, match="Invalid options"):
        options_from_dict(_TestOptions, {"count": "not_a_number"})


def test_options_from_dict_non_dataclass():
    with pytest.raises(TypeError, match="is not a dataclass"):
        options_from_dict(int, {})


# --- CLI --options flag ---


def test_quality_options_options_defaults():
    opts = get_quality_options(["/some/path"])
    assert opts.options is None


def test_quality_options_options_flag():
    opts = get_quality_options(
        ["/some/path", "--options", '{"use_mask": false, "power_threshold": 0.5}']
    )
    assert opts.options == '{"use_mask": false, "power_threshold": 0.5}'
