"""Tests for display utility functions."""

import pandas as pd

from napistu import utils


def test_show():
    """Test that utils.show() runs without errors."""

    # Create a simple test DataFrame
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Test that show() runs without raising an exception
    # We can't easily test the output since it depends on the environment
    utils.show(df, method="string")
    utils.show(df, method="string", headers=["col1", "col2"])
    utils.show(df, method="string", hide_index=True)

    # Test with auto method (should work in test environment)
    utils.show(df, method="auto")
