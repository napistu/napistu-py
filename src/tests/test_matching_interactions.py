import pandas as pd

from napistu.network import net_create
from napistu.network import precompute
from napistu.matching.interactions import (
    edgelist_to_pathway_species,
    edgelist_to_scids,
    filter_to_direct_mechanistic_interactions,
    filter_to_indirect_mechanistic_interactions,
)
from napistu.network.constants import NAPISTU_GRAPH_EDGES


def test_edgelist_to_pathway_species(sbml_dfs):

    edgelist = pd.DataFrame(
        [
            {"identifier_upstream": "17996", "identifier_downstream": "16526"},
            {"identifier_upstream": "15377", "identifier_downstream": "17544"},
            {"identifier_upstream": "15378", "identifier_downstream": "57945"},
            {"identifier_upstream": "57540", "identifier_downstream": "17996"},
        ]
    )
    species_identifiers = sbml_dfs.get_identifiers("species").query("bqb == 'BQB_IS'")

    edgelist_w_sids = edgelist_to_pathway_species(
        edgelist, species_identifiers, ontologies={"chebi", "uniprot"}
    )
    assert edgelist_w_sids.shape == (4, 4)

    egelist_w_scids = edgelist_to_scids(
        edgelist, sbml_dfs, species_identifiers, ontologies={"chebi"}
    )

    assert egelist_w_scids.shape == (12, 6)

    direct_interactions = filter_to_direct_mechanistic_interactions(
        edgelist, sbml_dfs, species_identifiers, ontologies={"chebi"}
    )

    assert direct_interactions.shape == (2, 10)


def test_direct_and_indirect_mechanism_matching(sbml_dfs_glucose_metabolism):

    napistu_graph = net_create.process_napistu_graph(sbml_dfs_glucose_metabolism)

    edgelist = pd.DataFrame(
        [
            {
                "identifier_upstream": "17925",
                "identifier_downstream": "32966",
            },  # glu, fbp
            {
                "identifier_upstream": "57634",
                "identifier_downstream": "32966",
            },  # f6p, fbp
            {
                "identifier_upstream": "32966",
                "identifier_downstream": "57642",
            },  # fbp, dhap
            {
                "identifier_upstream": "17925",
                "identifier_downstream": "15361",
            },  # glu, pyr
        ]
    )

    species_identifiers = sbml_dfs_glucose_metabolism.get_identifiers("species")

    direct_interactions = filter_to_direct_mechanistic_interactions(
        formatted_edgelist=edgelist,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        species_identifiers=species_identifiers,
        ontologies={"chebi"},
    )

    assert direct_interactions.shape == (2, 10)

    indirect_interactions = filter_to_indirect_mechanistic_interactions(
        formatted_edgelist=edgelist,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        species_identifiers=species_identifiers,
        napistu_graph=napistu_graph,
        ontologies={"chebi"},
        precomputed_distances=None,
        max_path_length=10,
    )

    assert indirect_interactions.shape == (6, 12)

    # confirm that we get the same thing even when using precomputed distances
    precomputed_distances = precompute.precompute_distances(
        napistu_graph, weight_vars=[NAPISTU_GRAPH_EDGES.WEIGHT]
    )

    indirect_interactions_w_precompute = filter_to_indirect_mechanistic_interactions(
        formatted_edgelist=edgelist,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        species_identifiers=species_identifiers,
        napistu_graph=napistu_graph,
        ontologies={"chebi"},
        precomputed_distances=precomputed_distances,
        max_path_length=10,
    )

    assert all(
        indirect_interactions["weight"] == indirect_interactions_w_precompute["weight"]
    )
