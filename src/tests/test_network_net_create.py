from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.network import net_create
from napistu.network import net_create_utils
from napistu.network import ng_utils
from napistu.constants import SBML_DFS
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    DEFAULT_WT_TRANS,
    WEIGHTING_SPEC,
    GRAPH_WIRING_APPROACHES,
    NAPISTU_GRAPH_EDGES,
    VALID_GRAPH_WIRING_APPROACHES,
)

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")

sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")
sbml_model = sbml.SBML(sbml_path)
sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)


def test_create_napistu_graph():
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.BIPARTITE
    )
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.REGULATORY
    )
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.SURROGATE
    )


def test_bipartite_regression():
    bipartite_og = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach="bipartite_og"
    )

    bipartite = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.BIPARTITE
    )

    bipartite_og_edges = bipartite_og.get_edge_dataframe()
    bipartite_edges = bipartite.get_edge_dataframe()

    try:
        pdt.assert_frame_equal(
            bipartite_og_edges, bipartite_edges, check_like=True, check_dtype=False
        )
    except AssertionError as e:
        # Print detailed differences
        print("DataFrames are not equal!")
        print(
            "Shape original:",
            bipartite_og_edges.shape,
            "Shape new:",
            bipartite_edges.shape,
        )
        print(
            "Columns original:",
            bipartite_og_edges.columns.tolist(),
            "Columns new:",
            bipartite_edges.columns.tolist(),
        )
        # Show head of both for quick inspection
        print("Original head:\n", bipartite_og_edges.head())
        print("New head:\n", bipartite_edges.head())
        # Optionally, show where values differ
        if bipartite_og_edges.shape == bipartite_edges.shape:
            diff = bipartite_og_edges != bipartite_edges
            print("Differences (first 5 rows):\n", diff.head())
        raise e  # Re-raise to fail the test


def test_create_napistu_graph_edge_reversed():
    """Test that edge_reversed=True properly reverses edges in the graph for all graph types."""
    # Test each graph type
    for wiring_approach in VALID_GRAPH_WIRING_APPROACHES:
        # Create graphs with and without edge reversal
        normal_graph = net_create.create_napistu_graph(
            sbml_dfs,
            wiring_approach=wiring_approach,
            directed=True,
            edge_reversed=False,
        )
        reversed_graph = net_create.create_napistu_graph(
            sbml_dfs, wiring_approach=wiring_approach, directed=True, edge_reversed=True
        )

        # Get edge dataframes for comparison
        normal_edges = normal_graph.get_edge_dataframe()
        reversed_edges = reversed_graph.get_edge_dataframe()

        # Verify we have edges to test
        assert len(normal_edges) > 0, f"No edges found in {wiring_approach} graph"
        assert len(normal_edges) == len(
            reversed_edges
        ), f"Edge count mismatch in {wiring_approach} graph"

        # Test edge reversal
        # Check a few edges to verify from/to are swapped
        for i in range(min(5, len(normal_edges))):
            # Check from/to are swapped
            assert (
                normal_edges.iloc[i][NAPISTU_GRAPH_EDGES.FROM]
                == reversed_edges.iloc[i][NAPISTU_GRAPH_EDGES.TO]
            ), f"From/to not properly swapped in {wiring_approach} graph"
            assert (
                normal_edges.iloc[i][NAPISTU_GRAPH_EDGES.TO]
                == reversed_edges.iloc[i][NAPISTU_GRAPH_EDGES.FROM]
            ), f"From/to not properly swapped in {wiring_approach} graph"

            # Check stoichiometry is negated
            assert (
                normal_edges.iloc[i][SBML_DFS.STOICHIOMETRY]
                == -reversed_edges.iloc[i][SBML_DFS.STOICHIOMETRY]
            ), f"Stoichiometry not properly negated in {wiring_approach} graph"

            # Check direction attributes are properly swapped
            if normal_edges.iloc[i]["direction"] == "forward":
                assert (
                    reversed_edges.iloc[i]["direction"] == "reverse"
                ), f"Direction not properly reversed (forward->reverse) in {wiring_approach} graph"
            elif normal_edges.iloc[i]["direction"] == "reverse":
                assert (
                    reversed_edges.iloc[i]["direction"] == "forward"
                ), f"Direction not properly reversed (reverse->forward) in {wiring_approach} graph"

            # Check parents/children are swapped
            assert (
                normal_edges.iloc[i]["sc_parents"]
                == reversed_edges.iloc[i]["sc_children"]
            ), f"Parents/children not properly swapped in {wiring_approach} graph"
            assert (
                normal_edges.iloc[i]["sc_children"]
                == reversed_edges.iloc[i]["sc_parents"]
            ), f"Parents/children not properly swapped in {wiring_approach} graph"


def test_create_napistu_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.create_napistu_graph(
        sbml_dfs, reaction_graph_attrs=None, wiring_approach="bipartite"
    )


def test_process_napistu_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.process_napistu_graph(sbml_dfs, reaction_graph_attrs=None)


@pytest.mark.skip_on_windows
def test_igraph_loading():
    # test read/write of an igraph network
    directeds = [True, False]
    wiring_approaches = ["bipartite", "regulatory"]

    ng_utils.export_networks(
        sbml_dfs,
        model_prefix="tmp",
        outdir="/tmp",
        directeds=directeds,
        wiring_approaches=wiring_approaches,
    )

    for wiring_approach in wiring_approaches:
        for directed in directeds:
            import_pkl_path = ng_utils._create_network_save_string(
                model_prefix="tmp",
                outdir="/tmp",
                directed=directed,
                wiring_approach=wiring_approach,
            )
            network_graph = ng_utils.read_network_pkl(
                model_prefix="tmp",
                network_dir="/tmp",
                directed=directed,
                wiring_approach=wiring_approach,
            )

            assert network_graph.is_directed() == directed
            # cleanup
            os.unlink(import_pkl_path)


def test_reverse_network_edges(reaction_species_examples):

    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df("regulatory")

    rxn_edges = net_create_utils.format_tiered_reaction_species(
        rxn_species=reaction_species_examples["all_entities"],
        r_id="foo",
        graph_hierarchy_df=graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )
    augmented_network_edges = rxn_edges.assign(r_isreversible=True)
    augmented_network_edges["sc_parents"] = range(0, augmented_network_edges.shape[0])
    augmented_network_edges["sc_children"] = range(
        augmented_network_edges.shape[0], 0, -1
    )

    assert net_create._reverse_network_edges(augmented_network_edges).shape[0] == 2


def test_entity_validation():
    # Test basic validation
    entity_attrs = {"table": "reactions", "variable": "foo"}
    assert net_create._EntityAttrValidator(**entity_attrs).model_dump() == {
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
    net_create._validate_entity_attrs(
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
        net_create._validate_entity_attrs(
            entity_attrs_invalid, custom_transformations=custom_transformations
        )
    assert "transformation 'invalid_trans' was not defined" in str(excinfo.value)

    # Test with validate_transformations=False
    # Should not raise any errors even with invalid transformation
    net_create._validate_entity_attrs(
        entity_attrs_invalid, validate_transformations=False
    )

    # Test with non-dict input
    with pytest.raises(AssertionError) as excinfo:
        net_create._validate_entity_attrs(["not", "a", "dict"])
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
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None


def test_pluck_entity_data_empty_species_dict(sbml_dfs):
    # graph_attrs contains 'species' key but value is empty dict
    graph_attrs = {SBML_DFS.SPECIES: {}}
    result = net_create.pluck_entity_data(sbml_dfs, graph_attrs, SBML_DFS.SPECIES)
    assert result is None
