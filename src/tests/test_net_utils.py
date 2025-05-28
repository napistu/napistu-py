from __future__ import annotations

import pytest

import igraph as ig
import numpy as np
import pandas as pd
from napistu.network import net_utils
from napistu.network import net_create


def test_safe_fill():
    safe_fill_test = ["a_very_long stringggg", ""]
    assert [net_utils.safe_fill(x) for x in safe_fill_test] == [
        "a_very_long\nstringggg",
        "",
    ]


def test_cpr_graph_to_pandas_dfs():
    graph_data = [
        (0, 1),
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 2),
        (2, 5),
        (5, 0),
        (6, 3),
        (5, 6),
    ]

    g = ig.Graph(graph_data, directed=True)
    vs, es = net_utils.cpr_graph_to_pandas_dfs(g)

    assert all(vs["index"] == list(range(0, 7)))
    assert (
        pd.DataFrame(graph_data)
        .rename({0: "source", 1: "target"}, axis=1)
        .sort_values(["source", "target"])
        .equals(es.sort_values(["source", "target"]))
    )


def test_validate_graph_attributes(sbml_dfs):

    cpr_graph = net_create.process_cpr_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    assert (
        net_utils._validate_edge_attributes(cpr_graph, ["weights", "upstream_weights"])
        is None
    )
    assert net_utils._validate_vertex_attributes(cpr_graph, "node_type") is None
    with pytest.raises(ValueError):
        net_utils._validate_vertex_attributes(cpr_graph, "baz")


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
                "table": "mock_table",
                "variable": "string_col",
                "trans": "identity",
            },
            "mixed_col": {
                "table": "mock_table",
                "variable": "mixed_col",
                "trans": "identity",
            },
            "ones_col": {
                "table": "mock_table",
                "variable": "ones_col",
                "trans": "identity",
            },
            "squared_col": {
                "table": "mock_table",
                "variable": "squared_col",
                "trans": "square",
            },
        }
    }
    # Call pluck_entity_data with custom transformation
    result = net_create.pluck_entity_data(
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
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, "species")
    assert result is None


def test_pluck_entity_data_empty_species_dict(sbml_dfs):
    # graph_attrs contains 'species' key but value is empty dict
    graph_attrs = {"species": {}}
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, "species")
    assert result is None


################################################
# __main__
################################################

if __name__ == "__main__":
    test_safe_fill()
    test_cpr_graph_to_pandas_dfs()
    test_validate_graph_attributes()
    test_pluck_entity_data_species_identity()
    test_pluck_entity_data_missing_species_key()
    test_pluck_entity_data_empty_species_dict()
