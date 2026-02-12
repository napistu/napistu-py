"""Tests for string utility functions."""

import pandas as pd
import pytest

from napistu.utils.string_utils import (
    extract_regex_match,
    extract_regex_search,
    safe_capitalize,
    safe_fill,
    safe_join_set,
    score_nameness,
)


def test_extract_regex():
    assert extract_regex_search("ENS[GT][0-9]+", "ENST0005") == "ENST0005"
    assert extract_regex_search("ENS[GT]([0-9]+)", "ENST0005", 1) == "0005"
    with pytest.raises(ValueError):
        extract_regex_search("ENS[GT][0-9]+", "ENSA0005")

    assert extract_regex_match(".*type=([a-zA-Z]+).*", "Ltype=abcd5") == "abcd"
    # use for formatting identifiers
    assert extract_regex_match("^([a-zA-Z]+)_id$", "sc_id") == "sc"
    with pytest.raises(ValueError):
        extract_regex_match(".*type=[a-zA-Z]+.*", "Ltype=abcd5")


def test_score_nameness():
    assert score_nameness("p53") == 23
    assert score_nameness("ENSG0000001") == 56
    assert score_nameness("pyruvate kinase") == 15


def test_safe_fill():
    safe_fill_test = ["a_very_long stringggg", ""]
    assert [safe_fill(x) for x in safe_fill_test] == [
        "a_very_long\nstringggg",
        "",
    ]


def test_safe_join_set():
    """Test safe_join_set function with various inputs."""
    # Test basic functionality and sorting
    assert safe_join_set([1, 2, 3]) == "1 OR 2 OR 3"
    assert safe_join_set(["c", "a", "b"]) == "a OR b OR c"

    # Test deduplication
    assert safe_join_set([1, 1, 2, 3]) == "1 OR 2 OR 3"

    # Test None handling
    assert safe_join_set([1, None, 3]) == "1 OR 3"
    assert safe_join_set([None, None]) is None

    # Test pandas Series (use object dtype to preserve None)
    series = pd.Series([3, 1, None, 2], dtype=object)
    assert safe_join_set(series) == "1 OR 2 OR 3"

    # Test string as single value
    assert safe_join_set("hello") == "hello"

    # Test empty inputs
    assert safe_join_set([]) is None


def test_safe_capitalize():
    """Test that safe_capitalize preserves acronyms."""
    assert safe_capitalize("regulatory RNAs") == "Regulatory RNAs"
    assert safe_capitalize("proteins") == "Proteins"
    assert safe_capitalize("DNA sequences") == "DNA sequences"
    assert safe_capitalize("") == ""
