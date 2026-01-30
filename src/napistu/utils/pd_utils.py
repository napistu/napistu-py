"""
Utilities for pandas DataFrame operations.

Classes
-------
match_pd_vars: class
    Match pandas variables - check if required variables are present in a DataFrame or Series.

Public Functions
----------------
check_unique_index(df: pd.DataFrame, label: str = "") -> None:
    Validate that each index value only maps to a single row.
drop_extra_cols(df_in: pd.DataFrame, df_out: pd.DataFrame, always_include: Optional[List[str]] = None) -> pd.DataFrame:
    Remove columns in df_out that are not in df_in, except those specified in always_include.
ensure_pd_df(pd_df_or_series: pd.DataFrame | pd.Series) -> pd.DataFrame:
    Ensure pandas DataFrame by converting a Series to DataFrame if needed.
format_identifiers_as_edgelist(df: pd.DataFrame, defining_vars: list[str], verbose: bool = False) -> pd.DataFrame:
    Format identifiers as edgelist by collapsing multiindex and multiple variables.
infer_entity_type(df: pd.DataFrame) -> str:
    Infer the entity type of a DataFrame based on its structure and schema.
matrix_to_edgelist(matrix: np.ndarray, row_labels: Optional[List] = None, col_labels: Optional[List] = None) -> pd.DataFrame:
    Convert a matrix to an edgelist format.
style_df(df: pd.DataFrame, headers: Union[str, list[str], None] = "keys", hide_index: bool = False) -> Styler:
    Style a pandas DataFrame with simple formatting options.
update_pathological_names(names: pd.Series, prefix: str) -> pd.Series:
    Update pathological names in a pandas Series by adding a prefix if all numeric.
validate_merge(left_df: pd.DataFrame, right_df: pd.DataFrame, left_on: Union[str, List[str]], right_on: Union[str, List[str]], relationship: Optional[str] = None) -> None:
    Validate merge relationship before performing a merge operation.
"""

from __future__ import annotations

import logging
from itertools import starmap
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.io.formats.style import Styler

from napistu.constants import SBML_DFS_SCHEMA, SCHEMA_DEFS
from napistu.utils.constants import (
    MERGE_RELATIONSHIP_TYPES,
    VALID_MERGE_RELATIONSHIP_TYPES,
)

logger = logging.getLogger(__name__)


class match_pd_vars:
    """
    Match Pandas Variables.

    Attributes
    ----------
    req_vars:
        A set of variables which should exist in df
    missing_vars:
        Required variables which are not present in df
    extra_vars:
        Non-required variables which are present in df
    are_present:
        Returns True if req_vars are present and False otherwise

    Methods
    -------
    assert_present()
        Raise an exception of req_vars are absent

    """

    def __init__(
        self, df: pd.DataFrame | pd.Series, req_vars: set, allow_series: bool = True
    ) -> None:
        """
        Connects to an SBML file

        Parameters
        ----------
        df
            A pd.DataFrame or pd.Series
        req_vars
            A set of variables which should exist in df
        allow_series:
            Can a pd.Series be provided as df?

        Returns
        -------
        None.
        """

        if isinstance(df, pd.Series):
            if not allow_series:
                raise TypeError("df was a pd.Series and must be a pd.DataFrame")
            vars_present = set(df.index.tolist())
        elif isinstance(df, pd.DataFrame):
            vars_present = set(df.columns.tolist())
        else:
            raise TypeError(
                f"df was a {type(df).__name__} and must be a pd.DataFrame or pd.Series"
            )

        self.req_vars = req_vars
        self.missing_vars = req_vars.difference(vars_present)
        self.extra_vars = vars_present.difference(req_vars)

        if len(self.missing_vars) == 0:
            self.are_present = True
        else:
            self.are_present = False

    def assert_present(self) -> None:
        """
        Raise an error if required variables are missing
        """

        if not self.are_present:
            raise ValueError(
                f"{len(self.missing_vars)} required variables were "
                "missing from the provided pd.DataFrame or pd.Series: "
                f"{', '.join(self.missing_vars)}"
            )

        return None


def check_unique_index(df, label=""):
    """Validate that each index value only maps to a single row."""

    if len(df.index) != len(df.index.unique()):
        raise ValueError(f"{label} index entries are not unique")

    return None


def drop_extra_cols(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    always_include: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove columns in df_out that are not in df_in, except those specified in always_include.

    Parameters
    ----------
    df_in : pd.DataFrame
        Reference DataFrame whose columns determine what to keep
    df_out : pd.DataFrame
        DataFrame to filter columns from
    always_include : Optional[List[str]], optional
        List of column names to always include in output, even if not in df_in

    Returns
    -------
    pd.DataFrame
        DataFrame with columns filtered to match df_in plus any always_include columns.
        Column order follows df_in, with always_include columns appended at the end.

    Examples
    --------
    >>> df_in = pd.DataFrame({'a': [1], 'b': [2]})
    >>> df_out = pd.DataFrame({'a': [3], 'c': [4], 'd': [5]})
    >>> _drop_extra_cols(df_in, df_out)
    # Returns DataFrame with just column 'a'

    >>> _drop_extra_cols(df_in, df_out, always_include=['d'])
    # Returns DataFrame with columns ['a', 'd']
    """
    # Handle None case for always_include
    if always_include is None:
        always_include = []

    # Get columns to retain: intersection with df_in plus always_include
    retained_cols = df_in.columns.intersection(df_out.columns).union(always_include)

    # Filter to only columns that exist in df_out
    retained_cols = retained_cols.intersection(df_out.columns)

    # Order columns: first those matching df_in's order, then any remaining always_include
    ordered_cols = []
    # Add columns that are in df_in in their original order
    for col in df_in.columns:
        if col in retained_cols:
            ordered_cols.append(col)
    # Add any remaining always_include columns that weren't in df_in
    for col in always_include:
        if col in retained_cols and col not in ordered_cols:
            ordered_cols.append(col)

    return df_out.loc[:, ordered_cols]


def ensure_pd_df(pd_df_or_series: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Ensure Pandas DataFrame

    Convert a pd.Series to a DataFrame if needed.

    Args:
        pd_df_or_series (pd.Series | pd.DataFrame):
            a pandas df or series

    Returns:
        pd_df converted to a pd.DataFrame if needed

    """

    if isinstance(pd_df_or_series, pd.DataFrame):
        return pd_df_or_series
    elif isinstance(pd_df_or_series, pd.Series):
        return pd_df_or_series.to_frame().T
    else:
        raise TypeError(
            "ensure_pd_df expects either a pandas DataFrame or Series but received"
            f" a {type(pd_df_or_series)}"
        )


def format_identifiers_as_edgelist(
    df: pd.DataFrame, defining_vars: list[str], verbose: bool = False
) -> pd.DataFrame:
    """
    Format Identifiers as Edgelist

    Collapse a multiindex to an index (if needed), and similarly collapse multiple variables to a single entry.
    This indexed pd.Sereies of index - ids can be treated as an edgelist for greedy clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Any pd.DataFrame
    defining_vars : list[str]
        A set of attributes which define a distinct entry in df
    verbose : bool, default=False
        If True, then include detailed logs.

    Returns
    -------
    df : pd.DataFrame
        A pd.DataFrame with an "ind" and "id" variable added indicating rolled up
        values of the index and defining_vars
    """

    # requires a named index by convention
    if None in df.index.names:
        raise ValueError(
            "df did not have a named index. A named index or multindex is expected"
        )

    if not isinstance(defining_vars, list):
        raise TypeError("defining_vars must be a list")

    if verbose:
        logger.info(
            f"creating an edgelist linking index levels {', '.join(df.index.names)} and linking it "
            f"to levels defined by {', '.join(defining_vars)}"
        )

    # df is a pd.DataFrame and contains defining_vars
    match_pd_vars(df, req_vars=set(defining_vars), allow_series=False).assert_present()

    # combine all components of a multindex into a single index value
    if df.index.nlevels == 1:
        df.loc[:, "ind"] = ["ind_" + x for x in df.index]
    else:
        # handle a multiindex
        fstr = "ind_" + "_".join(["{}"] * df.index.nlevels)
        df.loc[:, "ind"] = list(starmap(fstr.format, df.index))

    # aggregate defining variables
    df.loc[:, "id"] = df[defining_vars].apply(
        lambda x: "id_" + "_".join(x.dropna().astype(str)), axis=1
    )

    return df


def infer_entity_type(df: pd.DataFrame) -> str:
    """
    Infer the entity type of a DataFrame based on its structure and schema.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to analyze

    Returns
    -------
    str
        The inferred entity type name

    Raises
    ------
    ValueError
        If no entity type can be determined
    """
    schema = SBML_DFS_SCHEMA.SCHEMA

    # Get all primary keys
    primary_keys = [
        entity_schema.get(SCHEMA_DEFS.PK) for entity_schema in schema.values()
    ]
    primary_keys = [pk for pk in primary_keys if pk is not None]

    # Check if index matches a primary key
    if df.index.name in primary_keys:
        for entity_type, entity_schema in schema.items():
            if entity_schema.get(SCHEMA_DEFS.PK) == df.index.name:
                return entity_type

    # Get DataFrame columns that are also primary keys, including index or MultiIndex names
    index_names = []
    if isinstance(df.index, pd.MultiIndex):
        index_names = [name for name in df.index.names if name is not None]
    elif df.index.name is not None:
        index_names = [df.index.name]

    df_columns = set(df.columns).union(index_names).intersection(primary_keys)

    # Check for exact match with primary key + foreign keys
    for entity_type, entity_schema in schema.items():
        expected_keys = set()

        # Add primary key
        pk = entity_schema.get(SCHEMA_DEFS.PK)
        if pk:
            expected_keys.add(pk)

        # Add foreign keys
        fks = entity_schema.get(SCHEMA_DEFS.FK, [])
        expected_keys.update(fks)

        # Check for exact match
        if len(df_columns) == 1 and set(df_columns) == {pk}:
            # only a single key is present and its this entities pk
            return entity_type

        if df_columns == expected_keys:
            # all primary and foreign keys are present
            return entity_type

    # No match found
    raise ValueError(
        f"No entity type matches DataFrame with index: {df.index.names} and columns: {sorted(df_columns)}"
    )


def matrix_to_edgelist(matrix, row_labels=None, col_labels=None):
    rows, cols = np.where(~np.isnan(matrix))

    edgelist = pd.DataFrame(
        {
            "row": rows if row_labels is None else [row_labels[i] for i in rows],
            "column": cols if col_labels is None else [col_labels[i] for i in cols],
            "value": matrix[rows, cols],
        }
    )

    return edgelist


def style_df(
    df: pd.DataFrame,
    headers: Union[str, list[str], None] = "keys",
    hide_index: bool = False,
) -> Styler:
    """
    Style DataFrame

    Provide some simple options for styling a pd.DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        A table to style
    headers: Union[str, list[str], None]
        - "keys" to use the current column names
        - None to suppress column names
        - list[str] to overwrite and show column names
    hide_index: bool
        Should rows be displayed?

    Returns
    -------
    styled_df: Styler
        `df` with styles updated
    """

    if isinstance(headers, list):
        if len(headers) != df.shape[1]:
            raise ValueError(
                f"headers was a list with {len(headers)} entries, but df has {df.shape[1]} "
                "columns. These dimensions should match"
            )

        df.columns = headers  # type: ignore

    styled_df = df.style.format(precision=3).set_table_styles(
        [{"selector": "th", "props": "color: limegreen;"}]
    )

    if hide_index:
        styled_df = styled_df.hide(axis="index")

    if headers is None:
        return styled_df.hide(axis="columns")
    elif isinstance(headers, str):
        if headers == "keys":
            # just plot with the index as headers
            return styled_df
        else:
            raise ValueError(
                f"headers was a string: {headers} but this option is not recognized. "
                'The only defined value is "keys".'
            )
    else:
        assert isinstance(headers, list)
        return styled_df


def update_pathological_names(names: pd.Series, prefix: str) -> pd.Series:
    """
    Update pathological names in a pandas Series.

    Add a prefix to the names if they are all numeric.
    """
    if names.apply(lambda x: x.isdigit()).all():
        names = names.apply(lambda x: f"{prefix}{x}")
    return names


def validate_merge(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_on: Union[str, List[str]],
    right_on: Union[str, List[str]],
    relationship: str,
    _original_relationship: Optional[str] = None,
) -> None:
    """
    Validate merge relationship before performing a merge operation.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left DataFrame for merge
    right_df : pd.DataFrame
        Right DataFrame for merge
    left_on : str or list of str
        Column name(s) in left_df to merge on
    right_on : str or list of str
        Column name(s) in right_df to merge on
    relationship : str
        Expected relationship type to validate:
        - '1:1' (one-to-one): both keys are unique and match exactly
        - '1:m' (one-to-many): left keys are unique and can match to 1 or more right keys
        - 'm:1' (many-to-one): right keys are unique and can match to 1 or more left keys
        - 'm:m' (many-to-many): both keys may have duplicates
        - '1:0' (one-to-zero-or-one): left keys are unique and can match to 0 or more right keys
        - '0:1' (zero-or-one-to-one): right keys are unique and can match to 0 or more left keys

    Raises
    ------
    ValueError
        If relationship validation fails
    """
    valid_relationships = VALID_MERGE_RELATIONSHIP_TYPES
    if relationship not in valid_relationships:
        raise ValueError(
            f"relationship must be one of {valid_relationships}, got '{relationship}'"
        )

    # Handle m:1 and 0:1 by swapping DataFrames
    if relationship == MERGE_RELATIONSHIP_TYPES.MANY_TO_ONE:
        original = (
            _original_relationship
            if _original_relationship is not None
            else relationship
        )
        validate_merge(
            right_df,
            left_df,
            right_on,
            left_on,
            MERGE_RELATIONSHIP_TYPES.ONE_TO_MANY,
            _original_relationship=original,
        )
        return
    elif relationship == MERGE_RELATIONSHIP_TYPES.ZERO_TO_ONE:
        original = (
            _original_relationship
            if _original_relationship is not None
            else relationship
        )
        validate_merge(
            right_df,
            left_df,
            right_on,
            left_on,
            MERGE_RELATIONSHIP_TYPES.ONE_TO_ZERO,
            _original_relationship=original,
        )
        return

    # Normalize column names to lists
    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]

    if len(left_on) != len(right_on):
        raise ValueError(
            f"left_on and right_on must have the same length, got {len(left_on)} and {len(right_on)}"
        )

    # Extract key columns
    left_keys = left_df[left_on]
    right_keys = right_df[right_on]

    # Check uniqueness
    left_is_unique = not left_keys.duplicated().any()
    right_is_unique = not right_keys.duplicated().any()

    # Check if left keys are subset of right keys
    # Convert to tuples for multi-column keys, or use values directly for single column
    if len(left_on) == 1:
        left_key_set = set(left_keys[left_on[0]].unique())
        right_key_set = set(right_keys[right_on[0]].unique())
    else:
        left_key_set = set(left_keys.apply(tuple, axis=1).unique())
        right_key_set = set(right_keys.apply(tuple, axis=1).unique())

    # Use original relationship in error messages if provided.
    # When `_original_relationship` is set, we got here via swapping left/right inputs,
    # so we also swap the *labels* used in messages to match the caller's perspective.
    rel_for_msg = (
        _original_relationship if _original_relationship is not None else relationship
    )
    swapped = _original_relationship is not None
    left_msg = "right" if swapped else "left"
    right_msg = "left" if swapped else "right"

    if relationship != MERGE_RELATIONSHIP_TYPES.ONE_TO_ZERO:
        if left_key_set != right_key_set:
            missing_in_right = left_key_set - right_key_set
            missing_in_left = right_key_set - left_key_set
            msg_parts = []
            if missing_in_right:
                msg_parts.append(
                    f"{left_msg} keys not in {right_msg}: {missing_in_right}"
                )
            if missing_in_left:
                msg_parts.append(
                    f"{right_msg} keys not in {left_msg}: {missing_in_left}"
                )
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but key sets don't match. {', '.join(msg_parts)}"
            )

    if relationship == MERGE_RELATIONSHIP_TYPES.ONE_TO_ONE:
        if not left_is_unique:
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but {left_msg} DataFrame has duplicate keys"
            )
        if not right_is_unique:
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but {right_msg} DataFrame has duplicate keys"
            )

    elif relationship == MERGE_RELATIONSHIP_TYPES.ONE_TO_MANY:
        if not left_is_unique:
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but {left_msg} DataFrame has duplicate keys"
            )
    elif relationship == MERGE_RELATIONSHIP_TYPES.ONE_TO_ZERO:
        if not left_is_unique:
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but {left_msg} DataFrame has duplicate keys"
            )
        if not right_key_set.issubset(left_key_set):
            missing = right_key_set - left_key_set
            raise ValueError(
                f"Expected {rel_for_msg} relationship, but {right_msg} keys are not a subset of {left_msg} keys. "
                f"Missing in {left_msg}: {missing}"
            )


def _merge_and_log_overwrites(
    left_df: pd.DataFrame, right_df: pd.DataFrame, merge_context: str, **merge_kwargs
) -> pd.DataFrame:
    """
    Merge two DataFrames and log any column overwrites.

    Parameters
    ----------
    left_df : pd.DataFrame
        Left DataFrame for merge
    right_df : pd.DataFrame
        Right DataFrame for merge
    merge_context : str
        Description of the merge operation for logging
    **merge_kwargs : dict
        Additional keyword arguments passed to pd.merge

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with overwritten columns removed
    """
    # Track original columns
    original_cols = left_df.columns.tolist()

    # Ensure we're using the correct suffixes
    merge_kwargs["suffixes"] = ("_old", "")

    # Perform merge
    merged_df = pd.merge(left_df, right_df, **merge_kwargs)

    # Check for and log any overwritten columns
    new_cols = merged_df.columns.tolist()
    overwritten_cols = [col for col in original_cols if col + "_old" in new_cols]
    if overwritten_cols:
        logger.warning(
            f"The following columns were overwritten during {merge_context} merge and their original values "
            f"have been suffixed with '_old': {', '.join(overwritten_cols)}"
        )
        # Drop the old columns
        cols_to_drop = [col + "_old" for col in overwritten_cols]
        merged_df = merged_df.drop(columns=cols_to_drop)

    return merged_df
