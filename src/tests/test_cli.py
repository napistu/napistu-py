"""Tests for CLI utilities."""

import pytest

from napistu import _cli


def test_click_str_to_list():
    assert _cli.click_str_to_list("['foo', bar]") == ["foo", "bar"]
    with pytest.raises(ValueError):
        _cli.click_str_to_list("foo")
