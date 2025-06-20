import pandas as pd

from napistu.network import paths
from napistu.network import ng_utils


def test_shortest_paths(sbml_dfs, napistu_graph, napistu_graph_undirected):
    species = sbml_dfs.species
    source_species = species[species["s_name"] == "NADH"]
    dest_species = species[species["s_name"] == "NAD+"]
    target_species_paths = ng_utils.compartmentalize_species_pairs(
        sbml_dfs, source_species.index.tolist(), dest_species.index.tolist()
    )

    (
        all_shortest_reaction_paths_df,
        _,
        _,
        _,
    ) = paths.find_all_shortest_reaction_paths(
        napistu_graph, sbml_dfs, target_species_paths, weight_var="weights"
    )

    # undirected graph
    (
        all_shortest_reaction_paths_df,
        all_shortest_reaction_path_edges_df,
        edge_sources,
        paths_graph,
    ) = paths.find_all_shortest_reaction_paths(
        napistu_graph_undirected, sbml_dfs, target_species_paths, weight_var="weights"
    )

    assert all_shortest_reaction_paths_df.shape[0] == 3


def test_net_polarity():
    polarity_series = pd.Series(
        ["ambiguous", "ambiguous"], index=[0, 1], name="link_polarity"
    )
    assert all(
        [x == "ambiguous" for x in paths._calculate_net_polarity(polarity_series)]
    )

    polarity_series = pd.Series(
        ["activation", "inhibition", "inhibition", "ambiguous"],
        index=range(0, 4),
        name="link_polarity",
    )
    assert paths._calculate_net_polarity(polarity_series) == [
        "activation",
        "inhibition",
        "activation",
        "ambiguous activation",
    ]
    assert paths._terminal_net_polarity(polarity_series) == "ambiguous activation"
