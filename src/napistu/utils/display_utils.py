from __future__ import annotations

import logging

import pandas as pd
from pandas.io.formats.style import Styler

from napistu.utils.pd_utils import style_df

logger = logging.getLogger(__name__)


def show(
    obj,
    method="auto",
    headers="keys",
    hide_index=False,
    left_align_strings=True,
    max_rows=20,
):
    """Show a table using the appropriate method for the environment.

    Parameters
    ----------
    obj : pd.DataFrame or any other object
        The object to show
    method : str
        The method to use to show the object
        - "string" : show the object as a string
        - "jupyter" : show the object in a Jupyter notebook
        - "auto" : show the object in a Jupyter notebook if available, otherwise show as a string
    headers : str, list, or None
        The headers to use for the object
    left_align_strings : bool
        Should strings be left aligned?
    max_rows : int
        The maximum number of rows to show

    Returns
    -------
    None

    Examples
    --------
    >>> show(pd.DataFrame({"a": [1, 2, 3]}), headers="keys", hide_index=True)
    """

    if method == "string":
        _show_as_string(
            obj,
            headers=headers,
            hide_index=hide_index,
            max_rows=max_rows,
            left_align_strings=left_align_strings,
        )

    elif method in ("jupyter", "auto"):
        try:
            from IPython.display import display as jupyter_display

            if method == "jupyter" or _in_jupyter_environment():
                jupyter_display(
                    style_df(obj, headers=headers, hide_index=hide_index)
                    if isinstance(obj, pd.DataFrame)
                    else obj
                )
            else:
                _show_as_string(
                    obj,
                    headers=headers,
                    hide_index=hide_index,
                    max_rows=max_rows,
                    left_align_strings=left_align_strings,
                )
        except ImportError:
            if method == "jupyter":
                raise ImportError("IPython not available but jupyter method requested")
            _show_as_string(
                obj,
                headers=headers,
                hide_index=hide_index,
                max_rows=max_rows,
                left_align_strings=left_align_strings,
            )

    else:
        raise ValueError(f"Unknown method: {method}")


def _in_jupyter_environment():
    """Check if running in Jupyter notebook/lab."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _show_as_string(
    obj, headers="keys", hide_index=False, max_rows=20, left_align_strings=True
):
    """
    Show object using string representation with styling support.

    Parameters
    ----------
    obj : DataFrame or Styler
        Object to display
    headers : str, list, or None
        - "keys": use current column names
        - None: suppress column names
        - list: override column names
    hide_index : bool
        Whether to hide the row index
    max_rows : int
        Maximum number of rows to display
    left_align_strings : bool
        Should strings be left aligned?
    """

    # Extract DataFrame based on actual type
    if isinstance(obj, Styler):
        df = obj.data.copy()
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        print(obj)
        return

    # Handle headers
    if headers is None:
        # Suppress column names by setting them to empty strings
        df.columns = [""] * len(df.columns)
    elif isinstance(headers, list):
        # Override column names
        if len(headers) == len(df.columns):
            df.columns = headers
        else:
            logger.warning(
                f"Warning: headers length ({len(headers)}) doesn't match columns ({len(df.columns)})"
            )
    # If headers == 'keys', keep original column names (default)

    # Print with appropriate index setting
    if df.shape[0] > max_rows:
        logger.info(f"Displaying {max_rows} of {df.shape[0]} rows")

    if left_align_strings:
        formatters = _create_left_align_formatters(df)

        display_string = df.to_string(
            index=not hide_index,
            max_rows=max_rows,
            formatters=formatters,
            justify="left",
        )

    else:
        display_string = df.to_string(
            index=not hide_index, max_rows=max_rows, justify="left"
        )

    print(display_string)


def _create_left_align_formatters(df):
    """Create formatters for left-aligning string columns."""
    formatters = {}
    for col in df.columns:
        # Only apply to object/string columns
        if df[col].dtype == "object":
            # Calculate max width for this column
            if len(df) > 0:
                content_max = df[col].astype(str).str.len().max()
            else:
                content_max = 0
            header_max = len(str(col))
            width = max(content_max, header_max)

            # Create left-align formatter
            formatters[col] = lambda x, w=width: f"{str(x):<{w}}"

    return formatters
