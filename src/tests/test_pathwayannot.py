from __future__ import annotations

import os

from napistu.modify import pathwayannot

from napistu.constants import SBML_DFS
from napistu.modify.constants import COFACTOR_SCHEMA
from napistu.modify.constants import COFACTOR_CHEBI_IDS

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")
reduced_neo4j_members_path = os.path.join(test_data, "reduced_neo4j_members.csv")
reduced_neo4j_cross_refs_path = os.path.join(test_data, "reduced_neo4j_members.csv")


def test_cofactor_specifications():
    # cofactors are manually annotated with their ChEBI ID so that they can be matched
    # to the corresponding ChEBI ID of molecular species

    cofactor_labels_set = set(COFACTOR_CHEBI_IDS["cofactor"].tolist())

    deep_cofactor_list = [list(v.values()) for v in COFACTOR_SCHEMA.values()]
    shallow_cofactor_list = [item for sublist in deep_cofactor_list for item in sublist]
    unique_cofactors = {item for sublist in shallow_cofactor_list for item in sublist}

    # check whether all species in cofactor schema are tied with an ID
    undefined_labels = unique_cofactors.difference(cofactor_labels_set)
    if len(undefined_labels) != 0:
        raise ValueError(
            f"{', '.join(undefined_labels)} are not defined in \"COFACTOR_CHEBI_IDS\""
        )


def test_add_reactome_entity_sets(sbml_dfs_glucose_metabolism):

    # annotate the GPCR sbml_df with a reduced subset of the
    # Reactome entity set exports

    sbml_dfs_with_entity_sets = pathwayannot.add_entity_sets(
        sbml_dfs_glucose_metabolism, reduced_neo4j_members_path
    )

    lost_species = set(sbml_dfs_glucose_metabolism.species[SBML_DFS.S_NAME]).difference(
        sbml_dfs_with_entity_sets.species[SBML_DFS.S_NAME]
    )
    assert len(lost_species) == 0
    new_species = set(sbml_dfs_with_entity_sets.species[SBML_DFS.S_NAME]).difference(
        sbml_dfs_glucose_metabolism.species[SBML_DFS.S_NAME]
    )
    assert new_species == {
        "HK1",
        "HK2",
        "HK3",
        "PRKACA",
        "PRKACB",
        "PRKACG",
        "SLC25A12",
        "SLC25A13",
        "SLC37A1",
    }

    lost_reactions = set(
        sbml_dfs_glucose_metabolism.reactions[SBML_DFS.R_NAME]
    ).difference(sbml_dfs_with_entity_sets.reactions[SBML_DFS.R_NAME])
    assert len(lost_reactions) == 0
    new_reactions = set(
        sbml_dfs_with_entity_sets.reactions[SBML_DFS.R_NAME]
    ).difference(sbml_dfs_glucose_metabolism.reactions[SBML_DFS.R_NAME])
    assert len(new_reactions) == 10


def test_add_reactome_cross_refs(sbml_dfs_glucose_metabolism):

    # test adding cross-references to a Reactome model

    sbml_dfs_with_cross_refs = pathwayannot.add_reactome_identifiers(
        sbml_dfs_glucose_metabolism,
        os.path.join(test_data, "reduced_neo4j_cross_refs.csv"),
    )

    sbml_dfs_glucose_metabolism.reaction_species.shape[
        0
    ] == sbml_dfs_with_cross_refs.reaction_species.shape[0]

    previous_species_identifiers = sbml_dfs_glucose_metabolism.get_identifiers(
        SBML_DFS.SPECIES
    )
    updated_species_identifiers = sbml_dfs_with_cross_refs.get_identifiers(
        SBML_DFS.SPECIES
    )
    assert (
        updated_species_identifiers.shape[0] - previous_species_identifiers.shape[0]
        == 88
    )
