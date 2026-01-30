"""Tests for pandas utility functions."""

import numpy as np
import pandas as pd
import pytest

from napistu.constants import SBML_DFS
from napistu.utils.pd_utils import (
    _merge_and_log_overwrites,
    drop_extra_cols,
    ensure_pd_df,
    format_identifiers_as_edgelist,
    infer_entity_type,
    match_pd_vars,
    matrix_to_edgelist,
    style_df,
    update_pathological_names,
    validate_merge,
)


def test_match_pd_vars():
    a_series = pd.Series({"foo": 1, "bar": 2})
    a_dataframe = pd.DataFrame({"foo": ["a", "b"], "bar": [1, 2]})

    assert match_pd_vars(a_series, {"foo", "bar"}).are_present
    assert not match_pd_vars(a_series, {"baz"}).are_present
    assert match_pd_vars(a_dataframe, {"foo", "bar"}).are_present
    assert not match_pd_vars(a_dataframe, {"baz"}).are_present


def test_ensure_pd_df():
    source_df = pd.DataFrame({"a": "b"}, index=[0])
    source_series = pd.Series({"a": "b"}).rename(0)

    converted_series = ensure_pd_df(source_series)

    assert isinstance(ensure_pd_df(source_df), pd.DataFrame)
    assert isinstance(converted_series, pd.DataFrame)
    assert all(converted_series.index == source_df.index)
    assert all(converted_series.columns == source_df.columns)
    assert all(converted_series == source_df)


def test_format_identifiers_as_edgelist():
    DEGEN_EDGELIST_DF_1 = pd.DataFrame(
        {
            "ind1": [0, 0, 1, 1, 1, 1],
            "ind2": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index(["ind1", "ind2"])

    DEGEN_EDGELIST_DF_2 = pd.DataFrame(
        {
            "ind": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index("ind")

    edgelist_df = format_identifiers_as_edgelist(DEGEN_EDGELIST_DF_1, ["ont", "val"])
    assert edgelist_df["ind"].iloc[0] == "ind_0_a"
    assert edgelist_df["id"].iloc[0] == "id_X_A"

    edgelist_df = format_identifiers_as_edgelist(DEGEN_EDGELIST_DF_1, ["val"])
    assert edgelist_df["ind"].iloc[0] == "ind_0_a"
    assert edgelist_df["id"].iloc[0] == "id_A"

    edgelist_df = format_identifiers_as_edgelist(DEGEN_EDGELIST_DF_2, ["ont", "val"])
    assert edgelist_df["ind"].iloc[0] == "ind_a"
    assert edgelist_df["id"].iloc[0] == "id_X_A"

    with pytest.raises(ValueError):
        format_identifiers_as_edgelist(
            DEGEN_EDGELIST_DF_2.reset_index(drop=True), ["ont", "val"]
        )


def test_style_df():
    np.random.seed(0)
    simple_df = pd.DataFrame(np.random.randn(20, 4), columns=["A", "B", "C", "D"])
    simple_df.index.name = "foo"

    multiindexed_df = (
        pd.DataFrame(
            {
                "category": ["foo", "foo", "foo", "bar", "bar", "bar"],
                "severity": ["major", "minor", "minor", "major", "major", "minor"],
            }
        )
        .assign(message="stuff")
        .groupby(["category", "severity"])
        .count()
    )

    # style a few pd.DataFrames
    isinstance(style_df(simple_df), pd.io.formats.style.Styler)
    isinstance(
        style_df(simple_df, headers=None, hide_index=True),
        pd.io.formats.style.Styler,
    )
    isinstance(
        style_df(simple_df, headers=["a", "b", "c", "d"], hide_index=True),
        pd.io.formats.style.Styler,
    )
    isinstance(style_df(multiindexed_df), pd.io.formats.style.Styler)


def test_drop_extra_cols():
    """Test the _drop_extra_cols function for removing and reordering columns."""
    # Setup test DataFrames
    df_in = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    df_out = pd.DataFrame(
        {
            "col2": [10, 11, 12],
            "col3": [13, 14, 15],
            "col4": [16, 17, 18],  # Extra column that should be dropped
            "col1": [19, 20, 21],  # Different order than df_in
        }
    )

    # Test basic functionality without always_include
    result = drop_extra_cols(df_in, df_out)

    # Check that extra column was dropped
    assert "col4" not in result.columns

    # Check that columns are in the same order as df_in
    assert list(result.columns) == list(df_in.columns)

    # Check that values are preserved
    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {"col1": [19, 20, 21], "col2": [10, 11, 12], "col3": [13, 14, 15]}
        )[
            list(df_in.columns)
        ],  # Ensure same column order
    )

    # Test with always_include
    result_with_include = drop_extra_cols(df_in, df_out, always_include=["col4"])

    # Check that col4 is retained and appears at the end
    assert list(result_with_include.columns) == list(df_in.columns) + ["col4"]
    assert result_with_include["col4"].equals(df_out["col4"])

    # Test with always_include containing non-existent column
    result_non_existent = drop_extra_cols(
        df_in, df_out, always_include=["col4", "col5"]
    )
    assert list(result_non_existent.columns) == list(df_in.columns) + ["col4"]

    # Test with always_include containing column from df_in
    result_overlap = drop_extra_cols(df_in, df_out, always_include=["col1", "col4"])
    assert list(result_overlap.columns) == list(df_in.columns) + ["col4"]

    # Test with no overlapping columns but some in always_include
    df_out_no_overlap = pd.DataFrame({"col4": [1, 2, 3], "col5": [4, 5, 6]})
    result_no_overlap = drop_extra_cols(df_in, df_out_no_overlap)
    assert result_no_overlap.empty
    assert list(result_no_overlap.columns) == []

    result_no_overlap_with_include = drop_extra_cols(
        df_in, df_out_no_overlap, always_include=["col4"]
    )
    assert list(result_no_overlap_with_include.columns) == ["col4"]
    assert result_no_overlap_with_include["col4"].equals(df_out_no_overlap["col4"])

    # Test with subset of columns
    df_out_subset = pd.DataFrame(
        {"col1": [1, 2, 3], "col3": [7, 8, 9], "col4": [10, 11, 12]}
    )
    result_subset = drop_extra_cols(df_in, df_out_subset)

    assert list(result_subset.columns) == ["col1", "col3"]
    pd.testing.assert_frame_equal(result_subset, df_out_subset[["col1", "col3"]])

    result_subset_with_include = drop_extra_cols(
        df_in, df_out_subset, always_include=["col4"]
    )
    assert list(result_subset_with_include.columns) == ["col1", "col3", "col4"]
    pd.testing.assert_frame_equal(
        result_subset_with_include, df_out_subset[["col1", "col3", "col4"]]
    )


def test_matrix_to_edgelist():
    # Test case 1: Basic functionality with numeric indices
    matrix = np.array([[1, 2, np.nan], [np.nan, 3, 4], [5, np.nan, 6]])
    expected_edgelist = pd.DataFrame(
        {
            "row": [0, 0, 1, 1, 2, 2],
            "column": [0, 1, 1, 2, 0, 2],
            "value": np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        }
    )
    result = matrix_to_edgelist(matrix)
    pd.testing.assert_frame_equal(result, expected_edgelist)

    # Test case 2: With row and column labels
    row_labels = ["A", "B", "C"]
    col_labels = ["X", "Y", "Z"]
    expected_labeled_edgelist = pd.DataFrame(
        {
            "row": ["A", "A", "B", "B", "C", "C"],
            "column": ["X", "Y", "Y", "Z", "X", "Z"],
            "value": np.array([1, 2, 3, 4, 5, 6], dtype=np.float64),
        }
    )
    result_labeled = matrix_to_edgelist(
        matrix, row_labels=row_labels, col_labels=col_labels
    )
    pd.testing.assert_frame_equal(result_labeled, expected_labeled_edgelist)

    # Test case 3: Empty matrix (all NaN)
    empty_matrix = np.full((2, 2), np.nan)
    empty_result = matrix_to_edgelist(empty_matrix)
    assert empty_result.empty

    # Test case 4: Single value matrix
    single_matrix = np.array([[1]])
    expected_single = pd.DataFrame(
        {"row": [0], "column": [0], "value": np.array([1], dtype=np.int64)}
    )
    result_single = matrix_to_edgelist(single_matrix)
    pd.testing.assert_frame_equal(result_single, expected_single)


def test_update_pathological_names():

    # All numeric
    s = pd.Series(["1", "2", "3"])
    out = update_pathological_names(s, "prefix_")
    assert all(x.startswith("prefix_") for x in out)
    assert list(out) == ["prefix_1", "prefix_2", "prefix_3"]

    # Mixed numeric and non-numeric
    s2 = pd.Series(["1", "foo", "3"])
    out2 = update_pathological_names(s2, "prefix_")
    assert list(out2) == ["1", "foo", "3"]

    # All non-numeric
    s3 = pd.Series(["foo", "bar", "baz"])
    out3 = update_pathological_names(s3, "prefix_")
    assert list(out3) == ["foo", "bar", "baz"]


def test_infer_entity_type():
    """Test entity type inference with valid keys"""
    # when index matches primary key.
    # Test compartments with index as primary key
    df = pd.DataFrame(
        {SBML_DFS.C_NAME: ["cytoplasm"], SBML_DFS.C_IDENTIFIERS: ["GO:0005737"]}
    )
    df.index.name = SBML_DFS.C_ID
    result = infer_entity_type(df)
    assert result == SBML_DFS.COMPARTMENTS

    # Test species with index as primary key
    df = pd.DataFrame(
        {SBML_DFS.S_NAME: ["glucose"], SBML_DFS.S_IDENTIFIERS: ["CHEBI:17234"]}
    )
    df.index.name = SBML_DFS.S_ID
    result = infer_entity_type(df)
    assert result == SBML_DFS.SPECIES

    # Test entity type inference by exact column matching.
    # Test compartmentalized_species (has foreign keys)
    df = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["glucose_c"],
            SBML_DFS.S_ID: ["glucose"],
            SBML_DFS.C_ID: ["cytoplasm"],
        }
    )
    result = infer_entity_type(df)
    assert result == "compartmentalized_species"

    # Test reaction_species (has foreign keys)
    df = pd.DataFrame(
        {
            SBML_DFS.RSC_ID: ["rxn1_glc"],
            SBML_DFS.R_ID: ["rxn1"],
            SBML_DFS.SC_ID: ["glucose_c"],
        }
    )
    result = infer_entity_type(df)
    assert result == SBML_DFS.REACTION_SPECIES

    # Test reactions (only primary key)
    df = pd.DataFrame({SBML_DFS.R_ID: ["rxn1"]})
    result = infer_entity_type(df)
    assert result == SBML_DFS.REACTIONS


def test_infer_entity_type_errors():
    """Test error cases for entity type inference."""
    # Test no matching entity type
    df = pd.DataFrame({"random_column": ["value"], "another_col": ["data"]})
    with pytest.raises(ValueError, match="No entity type matches DataFrame"):
        infer_entity_type(df)

    # Test partial match (missing required foreign key)
    df = pd.DataFrame(
        {SBML_DFS.SC_ID: ["glucose_c"], SBML_DFS.S_ID: ["glucose"]}
    )  # Missing c_id
    with pytest.raises(ValueError):
        infer_entity_type(df)

    # Test extra primary keys that shouldn't be there
    df = pd.DataFrame(
        {SBML_DFS.R_ID: ["rxn1"], SBML_DFS.S_ID: ["glucose"]}
    )  # Two primary keys
    with pytest.raises(ValueError):
        infer_entity_type(df)


def test_infer_entity_type_multindex():
    # DataFrame with MultiIndex (r_id, foo), should infer as reactions
    df = pd.DataFrame({"some_col": [1, 2]})
    df.index = pd.MultiIndex.from_tuples(
        [("rxn1", "a"), ("rxn2", "b")], names=[SBML_DFS.R_ID, "foo"]
    )
    result = infer_entity_type(df)
    assert result == SBML_DFS.REACTIONS

    # DataFrame with MultiIndex (sc_id, bar), should infer as compartmentalized_species
    df = pd.DataFrame({"some_col": [1, 2]})
    df.index = pd.MultiIndex.from_tuples(
        [("glucose_c", "a"), ("atp_c", "b")], names=[SBML_DFS.SC_ID, "bar"]
    )
    result = infer_entity_type(df)
    assert result == SBML_DFS.COMPARTMENTALIZED_SPECIES


def test_merge_and_log_overwrites(caplog):
    """Test merge_and_log_overwrites function."""

    # Test basic merge with no conflicts
    df1 = pd.DataFrame({"id": [1, 2], "value1": ["a", "b"]})
    df2 = pd.DataFrame({"id": [1, 2], "value2": ["c", "d"]})
    result = _merge_and_log_overwrites(df1, df2, "test", on="id")
    assert set(result.columns) == {"id", "value1", "value2"}
    assert len(caplog.records) == 0

    # Test merge with column conflict
    df1 = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "value": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "name": ["c", "d"]})
    result = _merge_and_log_overwrites(df1, df2, "test", on="id")

    # Check that the right columns exist
    assert set(result.columns) == {"id", "name", "value"}
    # Check that we got df2's values for the overlapping column
    assert list(result["name"]) == ["c", "d"]
    # Check that we kept df1's non-overlapping column
    assert list(result["value"]) == [10, 20]
    # Check that the warning was logged
    assert len(caplog.records) == 1
    assert "test merge" in caplog.records[0].message
    assert "name" in caplog.records[0].message

    # Test merge with multiple column conflicts
    caplog.clear()
    df1 = pd.DataFrame(
        {"id": [1, 2], "name": ["a", "b"], "value": [10, 20], "status": ["ok", "ok"]}
    )
    df2 = pd.DataFrame(
        {"id": [1, 2], "name": ["c", "d"], "status": ["pending", "done"]}
    )
    result = _merge_and_log_overwrites(df1, df2, "test", on="id")

    # Check that the right columns exist
    assert set(result.columns) == {"id", "name", "value", "status"}
    # Check that we got df2's values for the overlapping columns
    assert list(result["name"]) == ["c", "d"]
    assert list(result["status"]) == ["pending", "done"]
    # Check that we kept df1's non-overlapping column
    assert list(result["value"]) == [10, 20]
    # Check that the warning was logged with both column names
    assert len(caplog.records) == 1
    assert "test merge" in caplog.records[0].message
    assert "name" in caplog.records[0].message
    assert "status" in caplog.records[0].message

    # Test merge with index
    caplog.clear()
    df1 = pd.DataFrame({"name": ["a", "b"], "value": [10, 20]}, index=[1, 2])
    df2 = pd.DataFrame({"name": ["c", "d"]}, index=[1, 2])
    result = _merge_and_log_overwrites(
        df1, df2, "test", left_index=True, right_index=True
    )

    # Check that the right columns exist
    assert set(result.columns) == {"name", "value"}
    # Check that we got df2's values for the overlapping column
    assert list(result["name"]) == ["c", "d"]
    # Check that we kept df1's non-overlapping column
    assert list(result["value"]) == [10, 20]
    # Check that the warning was logged
    assert len(caplog.records) == 1
    assert "test merge" in caplog.records[0].message
    assert "name" in caplog.records[0].message


def test_validate_merge():
    """Test validate_merge function for different relationship types."""
    # Test 1:1 relationship (both unique)
    left_1_1 = pd.DataFrame({"key": [1, 2, 3], "val1": ["a", "b", "c"]})
    right_1_1 = pd.DataFrame({"key": [1, 2, 3], "val2": ["x", "y", "z"]})
    validate_merge(left_1_1, right_1_1, "key", "key", "1:1")

    # Test 1:1 failure - left has duplicates (but key sets still match)
    left_dup = pd.DataFrame({"key": [1, 1, 2, 3], "val1": ["a", "b", "c", "d"]})
    with pytest.raises(
        ValueError, match="Expected 1:1 relationship, but left DataFrame"
    ):
        validate_merge(left_dup, right_1_1, "key", "key", "1:1")

    # Test 1:1 failure - right has duplicates
    right_dup = pd.DataFrame({"key": [1, 2, 2, 3], "val2": ["x", "y", "z", "w"]})
    with pytest.raises(
        ValueError, match="Expected 1:1 relationship, but right DataFrame"
    ):
        validate_merge(left_1_1, right_dup, "key", "key", "1:1")

    # Test 1:1 failure - key sets don't match
    right_mismatch = pd.DataFrame({"key": [1, 2, 4], "val2": ["x", "y", "z"]})
    with pytest.raises(
        ValueError, match="Expected 1:1 relationship, but key sets don't match"
    ):
        validate_merge(left_1_1, right_mismatch, "key", "key", "1:1")

    # Test 1:m relationship (left unique, right can have duplicates, and key sets match)
    left_1_m = pd.DataFrame({"key": [1, 2, 3], "val1": ["a", "b", "c"]})
    right_1_m = pd.DataFrame({"key": [1, 1, 2, 3], "val2": ["x", "y", "z", "w"]})
    validate_merge(left_1_m, right_1_m, "key", "key", "1:m")

    # Test 1:m failure - left has duplicates
    with pytest.raises(
        ValueError, match="Expected 1:m relationship, but left DataFrame"
    ):
        validate_merge(left_dup, right_1_m, "key", "key", "1:m")

    # Test 1:m failure - key sets don't match
    right_1_m_bad = pd.DataFrame({"key": [1, 1, 2, 4], "val2": ["x", "y", "z", "w"]})
    with pytest.raises(
        ValueError, match="Expected 1:m relationship, but key sets don't match"
    ):
        validate_merge(left_1_m, right_1_m_bad, "key", "key", "1:m")

    # Test m:1 relationship (right unique, left can have duplicates, right is subset of left)
    left_m_1 = pd.DataFrame({"key": [1, 1, 2, 3], "val1": ["a", "b", "c", "d"]})
    right_m_1 = pd.DataFrame({"key": [1, 2, 3], "val2": ["x", "y", "z"]})
    validate_merge(left_m_1, right_m_1, "key", "key", "m:1")

    # Test m:1 failure - right has duplicates (key sets still match)
    right_m_1_dup = pd.DataFrame({"key": [1, 2, 2, 3], "val2": ["x", "y", "z", "w"]})
    with pytest.raises(
        ValueError, match="Expected m:1 relationship, but right DataFrame"
    ):
        validate_merge(left_m_1, right_m_1_dup, "key", "key", "m:1")

    # Test m:1 failure - key sets don't match
    right_m_1_mismatch = pd.DataFrame({"key": [1, 2], "val2": ["x", "y"]})
    with pytest.raises(
        ValueError, match="Expected m:1 relationship, but key sets don't match"
    ):
        validate_merge(left_m_1, right_m_1_mismatch, "key", "key", "m:1")

    # Test m:m relationship (both can have duplicates)
    left_m_m = pd.DataFrame({"key": [1, 1, 2], "val1": ["a", "b", "c"]})
    right_m_m = pd.DataFrame({"key": [1, 2, 2], "val2": ["x", "y", "z"]})
    validate_merge(left_m_m, right_m_m, "key", "key", "m:m")

    # Test 1:0 relationship (left unique, right can be duplicated, right is subset of left)
    left_1_0 = pd.DataFrame({"key": [1, 2, 3], "val1": ["a", "b", "c"]})
    right_1_0 = pd.DataFrame({"key": [1, 2], "val2": ["x", "y"]})
    validate_merge(left_1_0, right_1_0, "key", "key", "1:0")

    # Test 1:0 failure - left has duplicates
    with pytest.raises(
        ValueError, match="Expected 1:0 relationship, but left DataFrame"
    ):
        validate_merge(left_dup, right_1_0, "key", "key", "1:0")

    # Test 1:0 failure - right keys not subset of left
    right_1_0_bad = pd.DataFrame({"key": [1, 1, 2, 4], "val2": ["x", "y", "z", "w"]})
    with pytest.raises(
        ValueError, match="Expected 1:0 relationship, but right keys are not a subset"
    ):
        validate_merge(left_1_0, right_1_0_bad, "key", "key", "1:0")

    # Test 0:1 relationship (right unique; left keys must be a subset of right keys)
    left_0_1 = pd.DataFrame({"key": [1, 1, 2, 4], "val1": ["a", "b", "c", "d"]})
    # Include an extra right key (3) to represent a 0-match case on the left.
    right_0_1 = pd.DataFrame({"key": [1, 2, 3, 4], "val2": ["x", "y", "z", "w"]})
    validate_merge(left_0_1, right_0_1, "key", "key", "0:1")

    # Test 0:1 failure - right has duplicate keys
    right_0_1_bad = pd.DataFrame({"key": [1, 2, 2], "val2": ["x", "y", "z"]})
    with pytest.raises(
        ValueError, match="Expected 0:1 relationship, but right DataFrame"
    ):
        validate_merge(left_0_1, right_0_1_bad, "key", "key", "0:1")

    # Test invalid relationship
    with pytest.raises(ValueError, match="relationship must be one of"):
        validate_merge(left_1_1, right_1_1, "key", "key", "invalid")

    # Test multi-column keys
    left_multi = pd.DataFrame({"key1": [1, 2], "key2": ["a", "b"], "val1": [10, 20]})
    right_multi = pd.DataFrame({"key1": [1, 2], "key2": ["a", "b"], "val2": [100, 200]})
    validate_merge(left_multi, right_multi, ["key1", "key2"], ["key1", "key2"], "1:1")

    # Test multi-column keys with 1:m
    right_multi_dup = pd.DataFrame(
        {"key1": [1, 1, 2], "key2": ["a", "a", "b"], "val2": [100, 101, 200]}
    )
    validate_merge(
        left_multi, right_multi_dup, ["key1", "key2"], ["key1", "key2"], "1:m"
    )
