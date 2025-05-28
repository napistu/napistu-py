from __future__ import annotations

import os
import pytest

import pandas as pd
from napistu import sbml_dfs_core
from napistu.constants import DEFAULT_WT_TRANS
from napistu.constants import MINI_SBO_FROM_NAME
from napistu.ingestion import sbml
from napistu.network import neighborhoods
from napistu.network import net_create
from napistu.network import net_utils
from napistu.network import paths

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")

sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")
sbml_model = sbml.SBML(sbml_path).model
sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)

# create a dict containing reaction species for a few example reactions
reaction_species_examples_dict = dict()

# stub with a random reaction
r_id = sbml_dfs.reactions.index[0]

reaction_species_examples_dict["valid_interactor"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id],
        "sbo_term": [
            MINI_SBO_FROM_NAME["interactor"],
            MINI_SBO_FROM_NAME["interactor"],
        ],
        "sc_id": ["sc1", "sc2"],
        "stoichiometry": [0, 0],
    }
).set_index(["r_id", "sbo_term"])

reaction_species_examples_dict["invalid_interactor"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id],
        "sbo_term": [
            MINI_SBO_FROM_NAME["interactor"],
            MINI_SBO_FROM_NAME["product"],
        ],
        "sc_id": ["sc1", "sc2"],
        "stoichiometry": [0, 0],
    }
).set_index(["r_id", "sbo_term"])


# simple reaction with just substrates and products
reaction_species_examples_dict["sub_and_prod"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id],
        "sbo_term": [MINI_SBO_FROM_NAME["reactant"], MINI_SBO_FROM_NAME["product"]],
        "sc_id": ["sub", "prod"],
        "stoichiometry": [-1, 1],
    }
).set_index(["r_id", "sbo_term"])

reaction_species_examples_dict["stimulator"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id, r_id],
        "sbo_term": [
            MINI_SBO_FROM_NAME["reactant"],
            MINI_SBO_FROM_NAME["product"],
            MINI_SBO_FROM_NAME["stimulator"],
        ],
        "sc_id": ["sub", "prod", "stim"],
        "stoichiometry": [-1, 1, 0],
    }
).set_index(["r_id", "sbo_term"])

reaction_species_examples_dict["all_entities"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id, r_id, r_id],
        "sbo_term": [
            MINI_SBO_FROM_NAME["reactant"],
            MINI_SBO_FROM_NAME["product"],
            MINI_SBO_FROM_NAME["stimulator"],
            MINI_SBO_FROM_NAME["catalyst"],
        ],
        "sc_id": ["sub", "prod", "stim", "cat"],
        "stoichiometry": [-1, 1, 0, 0],
    }
).set_index(["r_id", "sbo_term"])

reaction_species_examples_dict["no_substrate"] = pd.DataFrame(
    {
        "r_id": [r_id, r_id, r_id, r_id, r_id],
        "sbo_term": [
            MINI_SBO_FROM_NAME["product"],
            MINI_SBO_FROM_NAME["stimulator"],
            MINI_SBO_FROM_NAME["stimulator"],
            MINI_SBO_FROM_NAME["inhibitor"],
            MINI_SBO_FROM_NAME["catalyst"],
        ],
        "sc_id": ["prod", "stim1", "stim2", "inh", "cat"],
        "stoichiometry": [1, 0, 0, 0, 0],
    }
).set_index(["r_id", "sbo_term"])


def test_create_cpr_graph():
    _ = net_create.create_cpr_graph(sbml_dfs, graph_type="bipartite")
    _ = net_create.create_cpr_graph(sbml_dfs, graph_type="regulatory")
    _ = net_create.create_cpr_graph(sbml_dfs, graph_type="surrogate")


def test_create_cpr_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.create_cpr_graph(
        sbml_dfs, reaction_graph_attrs=None, graph_type="bipartite"
    )


def test_igraph_construction():
    _ = net_create.process_cpr_graph(sbml_dfs)


def test_process_cpr_graph_none_attrs():
    # Should not raise when reaction_graph_attrs is None
    _ = net_create.process_cpr_graph(sbml_dfs, reaction_graph_attrs=None)


@pytest.mark.skip_on_windows
def test_igraph_loading():
    # test read/write of an igraph network
    directeds = [True, False]
    graph_types = ["bipartite", "regulatory"]

    net_utils.export_networks(
        sbml_dfs,
        model_prefix="tmp",
        outdir="/tmp",
        directeds=directeds,
        graph_types=graph_types,
    )

    for graph_type in graph_types:
        for directed in directeds:
            import_pkl_path = net_utils._create_network_save_string(
                model_prefix="tmp",
                outdir="/tmp",
                directed=directed,
                graph_type=graph_type,
            )
            network_graph = net_utils.read_network_pkl(
                model_prefix="tmp",
                network_dir="/tmp",
                directed=directed,
                graph_type=graph_type,
            )

            assert network_graph.is_directed() == directed
            # cleanup
            os.unlink(import_pkl_path)


def test_shortest_paths():
    species = sbml_dfs.species
    source_species = species[species["s_name"] == "NADH"]
    dest_species = species[species["s_name"] == "NAD+"]
    target_species_paths = net_utils.compartmentalize_species_pairs(
        sbml_dfs, source_species.index.tolist(), dest_species.index.tolist()
    )

    # directed graph
    cpr_graph = net_create.process_cpr_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )
    (
        all_shortest_reaction_paths_df,
        all_shortest_reaction_path_edges_df,
        edge_sources,
        paths_graph,
    ) = paths.find_all_shortest_reaction_paths(
        cpr_graph, sbml_dfs, target_species_paths, weight_var="weights"
    )

    # undirected graph
    cpr_graph = net_create.process_cpr_graph(
        sbml_dfs, directed=False, weighting_strategy="topology"
    )
    (
        all_shortest_reaction_paths_df,
        all_shortest_reaction_path_edges_df,
        edge_sources,
        paths_graph,
    ) = paths.find_all_shortest_reaction_paths(
        cpr_graph, sbml_dfs, target_species_paths, weight_var="weights"
    )

    assert all_shortest_reaction_paths_df.shape[0] == 3


def test_neighborhood():
    species = sbml_dfs.species
    source_species = species[species["s_name"] == "NADH"].index.tolist()

    query_sc_species = net_utils.compartmentalize_species(sbml_dfs, source_species)
    compartmentalized_species = query_sc_species["sc_id"].tolist()

    cpr_graph = net_create.process_cpr_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    neighborhood = neighborhoods.find_neighborhoods(
        sbml_dfs,
        cpr_graph,
        compartmentalized_species=compartmentalized_species,
        order=3,
    )

    assert neighborhood["species_73473"]["vertices"].shape[0] == 6


def test_format_interactors():
    # interactions are formatted

    graph_hierarchy_df = net_create._create_graph_hierarchy_df("regulatory")

    assert (
        net_create._format_tiered_reaction_species(
            r_id,
            reaction_species_examples_dict["valid_interactor"],
            sbml_dfs,
            graph_hierarchy_df,
        ).shape[0]
        == 1
    )

    print("Re-enable test once Issue #102 is solved")

    # catch error from invalid interactor specification
    # with pytest.raises(ValueError) as excinfo:
    #    net_create._format_tiered_reaction_species(
    #        r_id, reaction_species_examples_dict["invalid_interactor"], sbml_dfs
    #    )
    # assert str(excinfo.value).startswith("Invalid combinations of SBO_terms")

    # simple reaction with just substrates and products
    assert (
        net_create._format_tiered_reaction_species(
            r_id,
            reaction_species_examples_dict["sub_and_prod"],
            sbml_dfs,
            graph_hierarchy_df,
        ).shape[0]
        == 2
    )

    # add a stimulator (activator)
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id, reaction_species_examples_dict["stimulator"], sbml_dfs, graph_hierarchy_df
    )

    assert rxn_edges.shape[0] == 3
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]

    # add catalyst + stimulator
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["cat", "sub"]

    # no substrate
    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["no_substrate"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 5
    # stimulator -> reactant
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim1", "cat"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["stim2", "cat"]
    assert rxn_edges.iloc[2][["from", "to"]].tolist() == ["inh", "cat"]

    # use the surrogate model tiered layout also

    graph_hierarchy_df = net_create._create_graph_hierarchy_df("surrogate")

    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )

    assert rxn_edges.shape[0] == 4
    assert rxn_edges.iloc[0][["from", "to"]].tolist() == ["stim", "sub"]
    assert rxn_edges.iloc[1][["from", "to"]].tolist() == ["sub", "cat"]


def test_reverse_network_edges():
    graph_hierarchy_df = net_create._create_graph_hierarchy_df("regulatory")

    rxn_edges = net_create._format_tiered_reaction_species(
        r_id,
        reaction_species_examples_dict["all_entities"],
        sbml_dfs,
        graph_hierarchy_df,
    )
    augmented_network_edges = rxn_edges.assign(r_isreversible=True)
    augmented_network_edges["sc_parents"] = range(0, augmented_network_edges.shape[0])
    augmented_network_edges["sc_children"] = range(
        augmented_network_edges.shape[0], 0, -1
    )

    assert net_create._reverse_network_edges(augmented_network_edges).shape[0] == 2


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


def test_entity_validation():
    entity_attrs = {"table": "reactions", "variable": "foo"}

    assert net_create._EntityAttrValidator(**entity_attrs).model_dump() == {
        **entity_attrs,
        **{"trans": DEFAULT_WT_TRANS},
    }


################################################
# __main__
################################################

if __name__ == "__main__":
    test_create_cpr_graph()
    test_igraph_loading()
    test_igraph_construction()
    test_shortest_paths()
    test_neighborhood()
    test_format_interactors()
    test_reverse_network_edges()
    test_entity_validation()
