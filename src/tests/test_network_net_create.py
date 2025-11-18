from __future__ import annotations

import pandas.testing as pdt

from napistu.network import net_create, net_create_utils
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    GRAPH_WIRING_APPROACHES,
    NAPISTU_GRAPH_EDGE_DIRECTIONS,
    NAPISTU_GRAPH_EDGES,
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

    # Sort both DataFrames by FROM and TO to ignore row order differences
    # This allows comparison when only the order differs (e.g., due to deduplication)
    sort_cols = [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
    bipartite_og_edges_sorted = bipartite_og_edges.sort_values(sort_cols).reset_index(
        drop=True
    )
    bipartite_edges_sorted = bipartite_edges.sort_values(sort_cols).reset_index(
        drop=True
    )

    pdt.assert_frame_equal(
        bipartite_og_edges_sorted,
        bipartite_edges_sorted,
        check_like=True,
        check_dtype=False,
    )


def test_reverse_network_edges(reaction_species_examples):
    """Test _reverse_network_edges function."""
    graph_hierarchy_df = net_create_utils.create_graph_hierarchy_df(
        GRAPH_WIRING_APPROACHES.REGULATORY
    )

    rxn_edges = net_create_utils.format_tiered_reaction_species(
        rxn_species=reaction_species_examples["all_entities"],
        r_id="foo",
        graph_hierarchy_df=graph_hierarchy_df,
        drop_reactions_when=DROP_REACTIONS_WHEN.SAME_TIER,
    )

    # Create test edges with all required attributes
    augmented_network_edges = rxn_edges.assign(
        r_isreversible=True,
        sc_parents=range(0, rxn_edges.shape[0]),
        sc_children=range(rxn_edges.shape[0], 0, -1),
        weight=[1.0, 2.0, 3.0, 4.0][: rxn_edges.shape[0]],
        upstream_weight=[0.5, 1.5, 2.5, 3.5][: rxn_edges.shape[0]],
    )

    reversed_edges = net_create._reverse_network_edges(augmented_network_edges)

    # Basic checks
    assert reversed_edges.shape[0] == 2  # Should filter out regulator/catalyst edges
    assert NAPISTU_GRAPH_EDGES.DIRECTION in reversed_edges.columns
    assert all(
        reversed_edges[NAPISTU_GRAPH_EDGES.DIRECTION]
        == NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE
    )

    # Verify that edges are actually reversed by checking FROM/TO pairs are swapped
    original_from_to_pairs = set(
        zip(
            augmented_network_edges[NAPISTU_GRAPH_EDGES.FROM],
            augmented_network_edges[NAPISTU_GRAPH_EDGES.TO],
        )
    )
    reversed_from_to_pairs = set(
        zip(
            reversed_edges[NAPISTU_GRAPH_EDGES.FROM],
            reversed_edges[NAPISTU_GRAPH_EDGES.TO],
        )
    )
    # Each reversed edge should have swapped FROM/TO
    assert len(reversed_from_to_pairs) > 0
    assert any(
        (orig_to, orig_from) in reversed_from_to_pairs
        for orig_from, orig_to in original_from_to_pairs
    )
