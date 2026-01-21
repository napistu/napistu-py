"""Tests for string utility functions."""

import pandas as pd
import pytest

from napistu import utils


def test_extract_regex():
    assert utils.extract_regex_search("ENS[GT][0-9]+", "ENST0005") == "ENST0005"
    assert utils.extract_regex_search("ENS[GT]([0-9]+)", "ENST0005", 1) == "0005"
    with pytest.raises(ValueError):
        utils.extract_regex_search("ENS[GT][0-9]+", "ENSA0005")

    assert utils.extract_regex_match(".*type=([a-zA-Z]+).*", "Ltype=abcd5") == "abcd"
    # use for formatting identifiers
    assert utils.extract_regex_match("^([a-zA-Z]+)_id$", "sc_id") == "sc"
    with pytest.raises(ValueError):
        utils.extract_regex_match(".*type=[a-zA-Z]+.*", "Ltype=abcd5")


def test_score_nameness():
    assert utils.score_nameness("p53") == 23
    assert utils.score_nameness("ENSG0000001") == 56
    assert utils.score_nameness("pyruvate kinase") == 15


def test_safe_fill():
    safe_fill_test = ["a_very_long stringggg", ""]
    assert [utils.safe_fill(x) for x in safe_fill_test] == [
        "a_very_long\nstringggg",
        "",
    ]


def test_safe_join_set():
    """Test safe_join_set function with various inputs."""
    # Test basic functionality and sorting
    assert utils.safe_join_set([1, 2, 3]) == "1 OR 2 OR 3"
    assert utils.safe_join_set(["c", "a", "b"]) == "a OR b OR c"

    # Test deduplication
    assert utils.safe_join_set([1, 1, 2, 3]) == "1 OR 2 OR 3"

    # Test None handling
    assert utils.safe_join_set([1, None, 3]) == "1 OR 3"
    assert utils.safe_join_set([None, None]) is None

    # Test pandas Series (use object dtype to preserve None)
    series = pd.Series([3, 1, None, 2], dtype=object)
    assert utils.safe_join_set(series) == "1 OR 2 OR 3"

    # Test string as single value
    assert utils.safe_join_set("hello") == "hello"

    # Test empty inputs
    assert utils.safe_join_set([]) is None


def test_safe_capitalize():
    """Test that safe_capitalize preserves acronyms."""
    assert utils.safe_capitalize("regulatory RNAs") == "Regulatory RNAs"
    assert utils.safe_capitalize("proteins") == "Proteins"
    assert utils.safe_capitalize("DNA sequences") == "DNA sequences"
    assert utils.safe_capitalize("") == ""
