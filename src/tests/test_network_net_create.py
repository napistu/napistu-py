from __future__ import annotations

import pandas.testing as pdt

from napistu.network import net_create, net_create_utils
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    GRAPH_WIRING_APPROACHES,
)


def test_create_napistu_graph(sbml_dfs):
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.BIPARTITE
    )
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.REGULATORY
    )
    _ = net_create.create_napistu_graph(
        sbml_dfs, wiring_approach=GRAPH_WIRING_APPROACHES.SURROGATE
    )


def test_bipartite_regression(sbml_dfs):
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


def test_reverse_network_edges(reaction_species_examples):

    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df(
        GRAPH_WIRING_APPROACHES.REGULATORY
    )

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
