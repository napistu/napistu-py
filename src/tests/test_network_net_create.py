from __future__ import annotations

import pandas.testing as pdt

from napistu.network import net_create, net_create_utils
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    GRAPH_WIRING_APPROACHES,
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
