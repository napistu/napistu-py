from napistu.network import ng_utils
from napistu.network import neighborhoods


def test_neighborhood(sbml_dfs, napistu_graph):
    species = sbml_dfs.species
    source_species = species[species["s_name"] == "NADH"].index.tolist()

    query_sc_species = ng_utils.compartmentalize_species(sbml_dfs, source_species)
    compartmentalized_species = query_sc_species["sc_id"].tolist()

    neighborhood = neighborhoods.find_neighborhoods(
        sbml_dfs,
        napistu_graph,
        compartmentalized_species=compartmentalized_species,
        order=3,
    )

    assert neighborhood["species_73473"]["vertices"].shape[0] == 6
