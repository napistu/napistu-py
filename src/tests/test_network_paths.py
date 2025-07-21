import pandas as pd

from napistu.network import paths
from napistu.network import ng_utils
from napistu.constants import SBML_DFS
from napistu.network.constants import (
    NAPISTU_GRAPH_EDGES,
    NET_POLARITY,
)


def test_shortest_paths(sbml_dfs, napistu_graph, napistu_graph_undirected):
    species = sbml_dfs.species
    source_species = species[species[SBML_DFS.S_NAME] == "NADH"]
    dest_species = species[species[SBML_DFS.S_NAME] == "NAD+"]
    target_species_paths = ng_utils.compartmentalize_species_pairs(
        sbml_dfs, source_species.index.tolist(), dest_species.index.tolist()
    )

    (
        all_shortest_reaction_paths_df,
        _,
        _,
        _,
    ) = paths.find_all_shortest_reaction_paths(
        napistu_graph,
        sbml_dfs,
        target_species_paths,
        weight_var=NAPISTU_GRAPH_EDGES.WEIGHT,
    )

    # undirected graph
    (
        all_shortest_reaction_paths_df,
        _,
        _,
        _,
    ) = paths.find_all_shortest_reaction_paths(
        napistu_graph_undirected,
        sbml_dfs,
        target_species_paths,
        weight_var=NAPISTU_GRAPH_EDGES.WEIGHT,
    )

    assert all_shortest_reaction_paths_df.shape[0] == 3


def test_net_polarity():
    polarity_series = pd.Series(
        [NET_POLARITY.AMBIGUOUS, NET_POLARITY.AMBIGUOUS],
        index=[0, 1],
        name=NET_POLARITY.LINK_POLARITY,
    )
    assert all(
        [
            x == NET_POLARITY.AMBIGUOUS
            for x in paths._calculate_net_polarity(polarity_series)
        ]
    )

    polarity_series = pd.Series(
        [
            NET_POLARITY.ACTIVATION,
            NET_POLARITY.INHIBITION,
            NET_POLARITY.INHIBITION,
            NET_POLARITY.AMBIGUOUS,
        ],
        index=range(0, 4),
        name=NET_POLARITY.LINK_POLARITY,
    )
    assert paths._calculate_net_polarity(polarity_series) == [
        NET_POLARITY.ACTIVATION,
        NET_POLARITY.INHIBITION,
        NET_POLARITY.ACTIVATION,
        NET_POLARITY.AMBIGUOUS_ACTIVATION,
    ]
    assert (
        paths._terminal_net_polarity(polarity_series)
        == NET_POLARITY.AMBIGUOUS_ACTIVATION
    )
