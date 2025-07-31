from __future__ import annotations

import os

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
