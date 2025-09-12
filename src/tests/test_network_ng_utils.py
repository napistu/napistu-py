"""Tests for network utility functions."""

import numpy as np
import pandas as pd
import pytest

from napistu.constants import SBML_DFS
from napistu.network import ng_utils
from napistu.network.constants import (
    DEFAULT_WT_TRANS,
    WEIGHTING_SPEC,
)


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


def test_get_sbml_dfs_vertex_summaries_dimensions(sbml_dfs_metabolism):
    """Test that get_sbml_dfs_vertex_summaries returns correct output dimensions."""
    result = ng_utils.get_sbml_dfs_vertex_summaries(sbml_dfs_metabolism)

    # Verify output is a DataFrame
    assert isinstance(result, pd.DataFrame)

    # Verify specific dimensions
    assert result.shape == (245, 13)

    # Verify all values are numeric (should be filled with 0 for missing values)
    assert result.dtypes.apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()

    # Verify no NaN values (should be filled with 0)
    assert not result.isnull().any().any()


def test_separate_entity_attrs_by_source(sbml_dfs_w_data):
    """Test separation of entity attributes by data source (SBML vs side-loaded)."""
    from napistu.network.constants import SBML_DFS, WEIGHTING_SPEC

    # Create side-loaded attributes
    side_loaded_attributes = {
        "external_db": pd.DataFrame({"confidence_score": [0.8, 0.9, 0.7]})
    }

    # Create entity attributes mixing SBML and side-loaded data
    entity_attrs = {
        # SBML attributes - using actual table names from fixture
        "rxn_int_attr": {
            WEIGHTING_SPEC.TABLE: "rxn_data",
            WEIGHTING_SPEC.VARIABLE: "rxn_attr_int",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
        "rxn_float_attr": {
            WEIGHTING_SPEC.TABLE: "rxn_data",
            WEIGHTING_SPEC.VARIABLE: "rxn_attr_float",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
        # Side-loaded attributes
        "external_score": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "confidence_score",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
    }

    # Test the separation
    sbml_attrs, side_loaded_attrs = ng_utils.separate_entity_attrs_by_source(
        entity_attrs,
        entity_type=SBML_DFS.REACTIONS,
        sbml_dfs=sbml_dfs_w_data,
        side_loaded_attributes=side_loaded_attributes,
    )

    # Verify SBML attributes
    assert "rxn_int_attr" in sbml_attrs
    assert "rxn_float_attr" in sbml_attrs
    assert len(sbml_attrs) == 2

    # Verify side-loaded attributes
    assert "external_score" in side_loaded_attrs
    assert len(side_loaded_attrs) == 1

    # Verify structure consistency
    assert WEIGHTING_SPEC.TABLE in sbml_attrs["rxn_int_attr"]
    assert WEIGHTING_SPEC.VARIABLE in sbml_attrs["rxn_int_attr"]
    assert WEIGHTING_SPEC.TABLE in side_loaded_attrs["external_score"]
    assert WEIGHTING_SPEC.VARIABLE in side_loaded_attrs["external_score"]

    # Test single source cases

    # Case 1: Only sbml_dfs provided (no side_loaded_attributes)
    sbml_only_attrs = {
        "rxn_int_attr": {
            WEIGHTING_SPEC.TABLE: "rxn_data",
            WEIGHTING_SPEC.VARIABLE: "rxn_attr_int",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        }
    }

    sbml_attrs_only, side_loaded_attrs_only = ng_utils.separate_entity_attrs_by_source(
        sbml_only_attrs, entity_type=SBML_DFS.REACTIONS, sbml_dfs=sbml_dfs_w_data
    )
    assert "rxn_int_attr" in sbml_attrs_only
    assert len(sbml_attrs_only) == 1
    assert side_loaded_attrs_only == {}

    # Case 2: Only side_loaded_attributes provided (no sbml_dfs)
    side_loaded_only_attrs = {
        "external_score": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "confidence_score",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        }
    }

    sbml_attrs_only, side_loaded_attrs_only = ng_utils.separate_entity_attrs_by_source(
        side_loaded_only_attrs,
        entity_type=SBML_DFS.REACTIONS,
        side_loaded_attributes=side_loaded_attributes,
    )
    assert sbml_attrs_only == {}
    assert "external_score" in side_loaded_attrs_only
    assert len(side_loaded_attrs_only) == 1


def test_separate_entity_attrs_by_source_negative_cases(sbml_dfs_w_data):
    """Test negative cases for separate_entity_attrs_by_source function."""
    from napistu.network.constants import SBML_DFS, WEIGHTING_SPEC

    # Create test data
    side_loaded_attributes = {
        "external_db": pd.DataFrame({"confidence_score": [0.8, 0.9, 0.7]})
    }

    entity_attrs = {
        "rxn_int_attr": {
            WEIGHTING_SPEC.TABLE: "rxn_data",
            WEIGHTING_SPEC.VARIABLE: "rxn_attr_int",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
        "external_score": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "confidence_score",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
    }

    # Case 1: Neither sbml_dfs nor side_loaded_attributes provided
    with pytest.raises(
        ValueError,
        match="At least one of 'sbml_dfs' or 'side_loaded_attributes' must be provided",
    ):
        ng_utils.separate_entity_attrs_by_source(
            entity_attrs, entity_type=SBML_DFS.REACTIONS
        )

    # Case 2: Overlapping table names between sbml_dfs and side_loaded_attributes
    overlapping_side_loaded_attributes = {
        "rxn_data": pd.DataFrame(
            {"overlap_score": [0.1, 0.2, 0.3]}
        )  # overlaps with sbml_dfs table
    }

    with pytest.raises(ValueError, match="Overlapping table names found"):
        ng_utils.separate_entity_attrs_by_source(
            entity_attrs,
            entity_type=SBML_DFS.REACTIONS,
            sbml_dfs=sbml_dfs_w_data,
            side_loaded_attributes=overlapping_side_loaded_attributes,
        )

    # Case 3: Invalid entity_type
    with pytest.raises(ValueError, match="Invalid entity_type"):
        ng_utils.separate_entity_attrs_by_source(
            entity_attrs,
            entity_type="invalid",
            sbml_dfs=sbml_dfs_w_data,
            side_loaded_attributes=side_loaded_attributes,
        )

    # Case 4: Required table not found in either source
    missing_table_attrs = {
        "missing_attr": {
            WEIGHTING_SPEC.TABLE: "nonexistent_table",
            WEIGHTING_SPEC.VARIABLE: "some_var",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        }
    }

    with pytest.raises(ValueError, match="Required table names not found"):
        ng_utils.separate_entity_attrs_by_source(
            missing_table_attrs,
            entity_type=SBML_DFS.REACTIONS,
            sbml_dfs=sbml_dfs_w_data,
            side_loaded_attributes=side_loaded_attributes,
        )
