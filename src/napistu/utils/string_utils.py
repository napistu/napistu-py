"""
Utilities for string operations and text processing.

Public Functions
----------------
extract_regex_match(regex: str, query: str) -> str:
    Extract a matched substring using regex match on the full string.
extract_regex_search(regex: str, query: str, index_value: int = 0) -> str:
    Extract a matched substring using regex search.
match_regex_dict(s: str, regex_dict: Dict[str, any]) -> Optional[any]:
    Apply each regex in regex_dict to the string s and return the first match value.
safe_capitalize(text: str) -> str:
    Capitalize first letter only, preserving case of rest.
safe_fill(x: str, fill_width: int = 15) -> str:
    Safely wrap a string to a specified width.
safe_join_set(values: Any) -> str | None:
    Safely join values, filtering out None values with " OR " separator.
safe_series_tolist(x: str | pd.Series) -> list:
    Convert either a list or str to a list.
score_nameness(string: str) -> int:
    Score how name-like a string is (lower score is more name-like).
"""

from __future__ import annotations

import logging
import re
from textwrap import fill
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def extract_regex_search(regex: str, query: str, index_value: int = 0) -> str:
    """
    Match an identifier substring and otherwise throw an error

    Args:
        regex (str): regular expression to search
        query (str): string to search against
        index_value (int): entry in index to return

    return:
        match (str): a character string match

    """

    if m := re.search(regex, query):
        match = m[index_value]
    else:
        raise ValueError(
            f"{query} does not match the identifier regular expression: {regex}"
        )

    return match


def extract_regex_match(regex: str, query: str) -> str:
    """
    Args:
        regex (str): regular expression to search
        query (str): string to search against

    return:
        match (str): a character string match
    """

    if m := re.match(regex, query):
        if len(m.groups()) > 0:
            match = m.groups()[0]
        else:
            raise ValueError(
                f"{query} does not match a subgroup in the regular expression: {regex}"
            )
    else:
        raise ValueError(f"{query} does not match the regular expression: {regex}")

    return match


def match_regex_dict(s: str, regex_dict: Dict[str, any]) -> Optional[any]:
    """
    Apply each regex in regex_dict to the string s. If a regex matches, return its value.
    If no regex matches, return None.

    Parameters
    ----------
    s : str
        The string to test.
    regex_dict : dict
        Dictionary where keys are regex patterns (str), and values are the values to return.

    Returns
    -------
    The value associated with the first matching regex, or None if no match.
    """
    for pattern, value in regex_dict.items():
        if re.search(pattern, s):
            return value
    return None


def safe_capitalize(text: str) -> str:
    """Capitalize first letter only, preserve case of rest."""
    if not text:
        return text
    return text[0].upper() + text[1:]


def safe_fill(x: str, fill_width: int = 15) -> str:
    """
    Safely wrap a string to a specified width.

    Parameters
    ----------
    x : str
        The string to wrap.
    fill_width : int, optional
        The width to wrap the string to. Default is 15.

    Returns
    -------
    str
        The wrapped string.
    """

    if x == "":
        return ""
    else:
        return fill(x, fill_width)


def safe_join_set(values: Any) -> str | None:
    """
    Safely join values, filtering out None values.

    Converts input to a set (ensuring uniqueness), removes None values,
    and joins remaining values with " OR " separator in sorted order.

    Parameters
    ----------
    values : Any
        Values to join. Can be list, tuple, set, pandas Series, string,
        or other iterable. Strings are treated as single values, not character sequences.

    Returns
    -------
    str or None
        Joined string with " OR " separator in alphabetical order,
        or None if no valid values remain after filtering.

    Examples
    --------
    >>> safe_join_set([1, 2, 3])
    '1 OR 2 OR 3'
    >>> safe_join_set([3, 1, 2, 1])  # Removes duplicates and sorts
    '1 OR 2 OR 3'
    >>> safe_join_set([1, None, 3])
    '1 OR 3'
    >>> safe_join_set([None, None])
    None
    >>> safe_join_set("hello")  # String treated as single value
    'hello'
    """
    # Handle pandas Series
    if hasattr(values, "tolist"):
        unique_values = set(values.tolist()) - {None}
    # Handle regular iterables (but not strings)
    elif hasattr(values, "__iter__") and not isinstance(values, str):
        unique_values = set(values) - {None}
    # Handle single values (including strings)
    else:
        unique_values = set([values]) - {None}

    return " OR ".join(sorted(str(v) for v in unique_values)) if unique_values else None


def safe_series_tolist(x):
    """Convert either a list or str to a list."""

    if isinstance(x, str):
        return [x]
    elif isinstance(x, pd.Series):
        return x.tolist()
    else:
        raise TypeError(f"x was a {type(x)} but only str and pd.Series are supported")


def score_nameness(string: str):
    """
    Score Nameness

    This utility assigns a numeric score to a string reflecting how likely it is to be
    a human readable name. This will help to prioritize readable entries when we are
    trying to pick out a single name to display from a set of values which may also
    include entries like systematic ids.

    Args:
        string (str):
            An alphanumeric string

    Returns:
        score (int):
            An integer score indicating how name-like the string is (low is more name-like)
    """

    return (
        # string length
        string.__len__()
        # no-space penalty
        + (sum(c.isspace() for c in string) == 0) * 10
        # penalty for each number
        + sum(c.isdigit() for c in string) * 5
    )


def _add_nameness_score(df, name_var):
    """Add a nameness_score variable which reflects how name-like each entry is."""

    df.loc[:, "nameness_score"] = df[name_var].apply(score_nameness)
    return df


def _add_nameness_score_wrapper(df, name_var, table_schema):
    """Call _add_nameness_score with default value."""

    if name_var in table_schema.keys():
        return _add_nameness_score(df, table_schema[name_var])
    else:
        logger.debug(
            f"{name_var} is not defined in table_schema; adding a constant (1)"
        )
        return df.assign(nameness_score=1)
