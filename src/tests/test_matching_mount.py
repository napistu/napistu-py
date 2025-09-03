import copy

import numpy as np
import pandas as pd
import pytest

from napistu.constants import IDENTIFIERS, ONTOLOGIES, SBML_DFS
from napistu.matching import mount
from napistu.matching.constants import (
    FEATURE_ID_VAR_DEFAULT,
    RESOLVE_MATCHES_AGGREGATORS,
)


def test_bind_wide_results(sbml_dfs_glucose_metabolism):
    """
    Test that bind_wide_results correctly matches identifiers and adds results to species data.
    """
    # Get species identifiers, excluding reactome
    species_identifiers = (
        sbml_dfs_glucose_metabolism.get_identifiers(SBML_DFS.SPECIES)
        .query("bqb == 'BQB_IS'")
        .query("ontology != 'reactome'")
    )

    # Create example data with identifiers and results
    example_data = species_identifiers.groupby("ontology").head(10)[
        [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]
    ]
    example_data["results_a"] = np.random.randn(len(example_data))
    example_data["results_b"] = np.random.randn(len(example_data))
    example_data[FEATURE_ID_VAR_DEFAULT] = range(0, len(example_data))

    # Create wide format data
    example_data_wide = (
        example_data.pivot(
            columns=IDENTIFIERS.ONTOLOGY,
            values=IDENTIFIERS.IDENTIFIER,
            index=[FEATURE_ID_VAR_DEFAULT, "results_a", "results_b"],
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Test inplace=False (default)
    results_name = "test_results"
    original_sbml_dfs = copy.deepcopy(sbml_dfs_glucose_metabolism)
    sbml_dfs_result = mount.bind_wide_results(
        sbml_dfs=sbml_dfs_glucose_metabolism,
        results_df=example_data_wide,
        results_name=results_name,
        ontologies={ONTOLOGIES.UNIPROT, ONTOLOGIES.CHEBI},
        dogmatic=False,
        species_identifiers=None,
        feature_id_var=FEATURE_ID_VAR_DEFAULT,
        verbose=True,
        inplace=False,
    )

    # Verify original object is unchanged
    assert results_name not in original_sbml_dfs.species_data

    # Verify the results were added correctly to the new object
    assert results_name in sbml_dfs_result.species_data
    bound_results = sbml_dfs_result.species_data[results_name]
    assert set(bound_results.columns) == {
        FEATURE_ID_VAR_DEFAULT,
        "results_a",
        "results_b",
    }
    assert bound_results.shape == (23, 3)
    assert bound_results.loc["S00000056", "feature_id"] == "18,19"
    assert bound_results.loc["S00000057", "feature_id"] == "18"
    assert bound_results.loc["S00000010", "feature_id"] == "9"

    # Test inplace=True
    results_name_2 = "test_results_2"
    sbml_dfs_inplace = copy.deepcopy(sbml_dfs_glucose_metabolism)
    result_inplace = mount.bind_wide_results(
        sbml_dfs=sbml_dfs_inplace,
        results_df=example_data_wide,
        results_name=results_name_2,
        ontologies={ONTOLOGIES.UNIPROT, ONTOLOGIES.CHEBI},
        dogmatic=False,
        species_identifiers=None,
        feature_id_var=FEATURE_ID_VAR_DEFAULT,
        verbose=True,
        inplace=True,
    )

    # Verify the object was modified and function returned None
    assert result_inplace is None
    assert results_name_2 in sbml_dfs_inplace.species_data


def test_resolve_matches_with_example_data():
    """Test resolve_matches function with example data for all aggregation methods."""
    # Setup example data with overlapping 1-to-many and many-to-1 cases
    example_data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "B", "C", "D", "D", "E", "B", "B", "C"],
            SBML_DFS.S_ID: [
                "s_id_1",
                "s_id_1",
                "s_id_1",
                "s_id_4",
                "s_id_5",
                "s_id_6",
                "s_id_2",
                "s_id_3",
                "s_id_3",
            ],
            "results_a": [1, 2, 3, 0.4, 5, 6, 0.7, 0.8, 9],
            "results_b": [
                "foo",
                "foo",
                "bar",
                "bar",
                "baz",
                "baz",
                "not",
                "not",
                "not",
            ],
            # Add boolean column
            "is_active": [True, False, True, False, True, False, True, False, True],
        }
    )

    # Test that missing feature_id raises KeyError
    data_no_id = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s_id_1", "s_id_1", "s_id_2"],
            "results_a": [1, 2, 3],
            "results_b": ["foo", "bar", "baz"],
        }
    )
    with pytest.raises(KeyError, match=FEATURE_ID_VAR_DEFAULT):
        mount.resolve_matches(data_no_id)

    # Test with keep_id_col=True (default)
    result_with_id = mount.resolve_matches(
        example_data, keep_id_col=True, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN
    )

    # Verify feature_id column is present and correctly aggregated
    assert FEATURE_ID_VAR_DEFAULT in result_with_id.columns
    assert result_with_id.loc["s_id_1", FEATURE_ID_VAR_DEFAULT] == "A,B,C"
    assert result_with_id.loc["s_id_3", FEATURE_ID_VAR_DEFAULT] == "B,C"

    # Test with keep_id_col=False
    result_without_id = mount.resolve_matches(
        example_data, keep_id_col=False, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN
    )

    # Verify feature_id column is not in output
    assert FEATURE_ID_VAR_DEFAULT not in result_without_id.columns

    # Verify other columns are still present and correctly aggregated
    assert "results_a" in result_without_id.columns
    assert "results_b" in result_without_id.columns
    assert "feature_id_match_count" in result_without_id.columns

    # Test that boolean columns are handled correctly with first method
    first_result = mount.resolve_matches(
        example_data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.FIRST
    )

    # Verify boolean aggregation (should take first value after sorting by feature_id)
    assert first_result.loc["s_id_1", "is_active"]  # A comes first, has True
    assert not first_result.loc["s_id_3", "is_active"]  # B comes first, has False
    assert isinstance(first_result.loc["s_id_1", "is_active"], (bool, np.bool_))

    # Verify numeric aggregation still works
    actual_mean = result_without_id.loc["s_id_1", "results_a"]
    expected_mean = 2.0  # (1 + 2 + 3) / 3
    assert (
        actual_mean == expected_mean
    ), f"Expected mean {expected_mean}, but got {actual_mean}"

    # Verify string aggregation still works
    assert result_without_id.loc["s_id_1", "results_b"] == "bar,foo"

    # Verify match counts are still present
    assert result_without_id.loc["s_id_1", "feature_id_match_count"] == 3
    assert result_without_id.loc["s_id_3", "feature_id_match_count"] == 2

    # Test maximum aggregation
    max_result = mount.resolve_matches(
        example_data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MAX
    )

    # Verify maximum values are correct
    assert max_result.loc["s_id_1", "results_a"] == 3.0  # max of [1, 2, 3]
    assert max_result.loc["s_id_3", "results_a"] == 9.0  # max of [0.8, 9]
    assert max_result.loc["s_id_4", "results_a"] == 0.4  # single value
    assert max_result.loc["s_id_5", "results_a"] == 5.0  # single value
    assert max_result.loc["s_id_6", "results_a"] == 6.0  # single value

    # Test weighted mean (feature_id is used for weights regardless of keep_id_col)
    weighted_result = mount.resolve_matches(
        example_data,
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
        keep_id_col=True,
    )

    # For s_id_1:
    # A appears once in total (weight = 1/1)
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1 + 1/3 + 1/2 = 1.833
    # Normalized weights:
    # A: (1/1)/1.833 = 0.545
    # B: (1/3)/1.833 = 0.182
    # C: (1/2)/1.833 = 0.273
    # Weighted mean = 1×0.545 + 2×0.182 + 3×0.273 = 1.73
    actual_weighted_mean_1 = weighted_result.loc["s_id_1", "results_a"]
    expected_weighted_mean_1 = 1.73
    assert (
        abs(actual_weighted_mean_1 - expected_weighted_mean_1) < 0.01
    ), f"s_id_1 weighted mean: expected {expected_weighted_mean_1:.3f}, but got {actual_weighted_mean_1:.3f}"

    # For s_id_3:
    # B appears three times in total (weight = 1/3)
    # C appears twice in total (weight = 1/2)
    # Sum of unnormalized weights = 1/3 + 1/2 = 0.833
    # Normalized weights:
    # B: (1/3)/0.833 = 0.4
    # C: (1/2)/0.833 = 0.6
    # Weighted mean = 0.8×0.4 + 9×0.6 = 5.72
    actual_weighted_mean_3 = weighted_result.loc["s_id_3", "results_a"]
    expected_weighted_mean_3 = 5.72
    assert (
        abs(actual_weighted_mean_3 - expected_weighted_mean_3) < 0.01
    ), f"s_id_3 weighted mean: expected {expected_weighted_mean_3:.3f}, but got {actual_weighted_mean_3:.3f}"

    # Test weighted mean with keep_id_col=False (weights still use feature_id)
    weighted_result_no_id = mount.resolve_matches(
        example_data,
        numeric_agg=RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
        keep_id_col=False,
    )

    # Verify weighted means are the same regardless of keep_id_col
    assert (
        abs(weighted_result_no_id.loc["s_id_1", "results_a"] - expected_weighted_mean_1)
        < 0.01
    ), "Weighted mean should be the same regardless of keep_id_col"
    assert (
        abs(weighted_result_no_id.loc["s_id_3", "results_a"] - expected_weighted_mean_3)
        < 0.01
    ), "Weighted mean should be the same regardless of keep_id_col"

    # Test that both versions preserve the same index structure
    expected_index = pd.Index(
        ["s_id_1", "s_id_2", "s_id_3", "s_id_4", "s_id_5", "s_id_6"], name="s_id"
    )
    pd.testing.assert_index_equal(result_with_id.index, expected_index)
    pd.testing.assert_index_equal(result_without_id.index, expected_index)


def test_resolve_matches_first_method():
    """Test resolve_matches with first method."""
    # Setup data with known order
    data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "C", "B", "B", "A"],
            SBML_DFS.S_ID: ["s1", "s1", "s1", "s2", "s2"],
            "value": [1, 2, 3, 4, 5],
        }
    )

    result = mount.resolve_matches(data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.FIRST)

    # Should take first value after sorting by feature_id
    assert result.loc["s1", "value"] == 1  # A comes first
    assert result.loc["s2", "value"] == 5  # A comes first


def test_resolve_matches_deduplicate_feature_id_within_sid():
    """Test that only the first value for each (s_id, feature_id) is used in mean aggregation."""
    data = pd.DataFrame(
        {
            FEATURE_ID_VAR_DEFAULT: ["A", "A", "B"],
            SBML_DFS.S_ID: ["s1", "s1", "s1"],
            "value": [
                1,
                1,
                2,
            ],  # average should be 1.5 because the two A's are redundant
        }
    )

    result = mount.resolve_matches(data, numeric_agg=RESOLVE_MATCHES_AGGREGATORS.MEAN)
    assert result.loc["s1", "value"] == 1.5
