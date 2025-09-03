"""Tests for network utility functions."""

import numpy as np
import pandas as pd
import pytest

from napistu.constants import SBML_DFS
from napistu.network import ng_utils
from napistu.network.constants import DEFAULT_WT_TRANS, WEIGHTING_SPEC


def test_entity_validation():
    # Test basic validation
    entity_attrs = {"table": "reactions", "variable": "foo"}
    assert ng_utils._EntityAttrValidator(**entity_attrs).model_dump() == {
        **entity_attrs,
        **{"trans": DEFAULT_WT_TRANS},
    }

    # Test validation with custom transformations
    custom_transformations = {
        "nlog10": lambda x: -np.log10(x),
        "square": lambda x: x**2,
    }

    # Test valid custom transformation
    entity_attrs_custom = {
        "attr1": {
            WEIGHTING_SPEC.TABLE: "reactions",
            WEIGHTING_SPEC.VARIABLE: "foo",
            WEIGHTING_SPEC.TRANSFORMATION: "nlog10",
        },
        "attr2": {
            WEIGHTING_SPEC.TABLE: "species",
            WEIGHTING_SPEC.VARIABLE: "bar",
            WEIGHTING_SPEC.TRANSFORMATION: "square",
        },
    }
    # Should not raise any errors
    ng_utils._validate_entity_attrs(
        entity_attrs_custom, custom_transformations=custom_transformations
    )

    # Test invalid transformation
    entity_attrs_invalid = {
        "attr1": {
            WEIGHTING_SPEC.TABLE: "reactions",
            WEIGHTING_SPEC.VARIABLE: "foo",
            WEIGHTING_SPEC.TRANSFORMATION: "invalid_trans",
        }
    }
    with pytest.raises(ValueError) as excinfo:
        ng_utils._validate_entity_attrs(
            entity_attrs_invalid, custom_transformations=custom_transformations
        )
    assert "transformation 'invalid_trans' was not defined" in str(excinfo.value)

    # Test with validate_transformations=False
    # Should not raise any errors even with invalid transformation
    ng_utils._validate_entity_attrs(
        entity_attrs_invalid, validate_transformations=False
    )

    # Test with non-dict input
    with pytest.raises(AssertionError) as excinfo:
        ng_utils._validate_entity_attrs(["not", "a", "dict"])
    assert "entity_attrs must be a dictionary" in str(excinfo.value)


def test_pluck_entity_data_species_identity(sbml_dfs):
    # Take first 10 species IDs
    species_ids = sbml_dfs.species.index[:10]
    # Create mock data with explicit dtype to ensure cross-platform consistency
    # Fix for issue-42: Use explicit dtypes to avoid platform-specific dtype differences
    # between Windows (int32) and macOS/Linux (int64)
    mock_df = pd.DataFrame(
        {
            "string_col": [f"str_{i}" for i in range(10)],
            "mixed_col": np.arange(-5, 5, dtype=np.int64),  # Explicitly use int64
            "ones_col": np.ones(10, dtype=np.float64),  # Explicitly use float64
            "squared_col": np.arange(10, dtype=np.int64),  # Explicitly use int64
        },
        index=species_ids,
    )
    # Assign to species_data
    sbml_dfs.species_data["mock_table"] = mock_df

    # Custom transformation: square
    def square(x):
        return x**2

    custom_transformations = {"square": square}
    # Create graph_attrs for species
    graph_attrs = {
        "species": {
            "string_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "string_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "mixed_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "mixed_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "ones_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "ones_col",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            },
            "squared_col": {
                WEIGHTING_SPEC.TABLE: "mock_table",
                WEIGHTING_SPEC.VARIABLE: "squared_col",
                WEIGHTING_SPEC.TRANSFORMATION: "square",
            },
        }
    }
    # Call pluck_entity_data with custom transformation
    result = ng_utils.pluck_entity_data(
        sbml_dfs, graph_attrs, "species", custom_transformations=custom_transformations
    )
    # Check output
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"string_col", "mixed_col", "ones_col", "squared_col"}
    assert list(result.index) == list(species_ids)
    # Check values
    pd.testing.assert_series_equal(result["string_col"], mock_df["string_col"])
    pd.testing.assert_series_equal(result["mixed_col"], mock_df["mixed_col"])
    pd.testing.assert_series_equal(result["ones_col"], mock_df["ones_col"])
    pd.testing.assert_series_equal(
        result["squared_col"], mock_df["squared_col"].apply(square)
    )


def test_pluck_entity_data_missing_species_key(sbml_dfs):
    # graph_attrs does not contain 'species' key
    graph_attrs = {}
    result = ng_utils.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None


def test_pluck_entity_data_empty_species_dict(sbml_dfs):
    # graph_attrs contains 'species' key but value is empty dict
    graph_attrs = {SBML_DFS.SPECIES: {}}
    result = ng_utils.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None


def test_apply_weight_transformations_basic():
    """Test basic weight transformation functionality."""
    # Create test data
    edges_df = pd.DataFrame(
        {"string_wt": [150, 500, 1000, np.nan], "other_attr": [1, 2, 3, 4]}
    )

    reaction_attrs = {
        "string_wt": {
            "table": "string",
            "variable": "combined_score",
            "trans": "string_inv",
        }
    }

    # Apply transformations
    result = ng_utils.apply_weight_transformations(edges_df, reaction_attrs)

    # Check that string_wt was transformed
    expected_values = [1000 / 150, 1000 / 500, 1000 / 1000, np.nan]
    for i, expected in enumerate(expected_values):
        if pd.notna(expected):
            assert abs(result["string_wt"].iloc[i] - expected) < 1e-10
        else:
            assert pd.isna(result["string_wt"].iloc[i])

    # Check that other_attr was not changed
    assert all(result["other_attr"] == edges_df["other_attr"])


def test_apply_weight_transformations_nan_handling():
    """Test that NaN values are handled correctly."""
    edges_df = pd.DataFrame({"string_wt": [150, np.nan, 1000, 500, np.nan]})

    reaction_attrs = {
        "string_wt": {
            "table": "string",
            "variable": "combined_score",
            "trans": "string_inv",
        }
    }

    result = ng_utils.apply_weight_transformations(edges_df, reaction_attrs)

    # Check that NaN values remain NaN
    assert pd.isna(result["string_wt"].iloc[1])
    assert pd.isna(result["string_wt"].iloc[4])

    # Check that non-NaN values are transformed
    expected_values = [1000 / 150, np.nan, 1000 / 1000, 1000 / 500, np.nan]
    for i, expected in enumerate(expected_values):
        if pd.notna(expected):
            assert abs(result["string_wt"].iloc[i] - expected) < 1e-10
        else:
            assert pd.isna(result["string_wt"].iloc[i])
