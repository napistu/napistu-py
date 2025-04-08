from __future__ import annotations

import os

import pandas as pd
from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.modify import curation

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
sbml_path = os.path.join(test_path, "test_data", "R-HSA-1237044.sbml")

if not os.path.isfile(sbml_path):
    raise ValueError(f"{sbml_path} not found")

# setup mock curations
curation_dict = dict()
curation_dict["species"] = pd.DataFrame(
    [
        {
            "species": "hello",
            "uri": "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:35828",
            "curator": "Sean",
        },
        {"species": "good day", "uri": None, "curator": "Sean"},
    ]
)

curation_dict["compartmentalized_species"] = pd.DataFrame(
    [
        {
            "compartmentalized_species": "hello [cytosol]",
            "s_name": "hello",
            "c_name": "cytosol",
            "curator": "Sean",
        }
    ]
)

curation_dict["reactions"] = pd.DataFrame(
    [
        {
            "reactions": "there",
            "stoichiometry": "hello [cytosol] -> CO2 [cytosol]",
            "uri": None,
            "evidence": "how is",
            "curator": "Sean",
        },
        {
            "reactions": "where",
            "stoichiometry": "CO2 [cytosol] -> hello [cytosol]",
            "uri": None,
            "evidence": "your family",
            "curator": "Sean",
        },
    ]
)

curation_dict["reaction_species"] = pd.DataFrame(
    [
        {
            "reaction_species": "NADH [cytosol]",
            "r_name": "CYB5Rs reduce MetHb to HbA",
            "stoichiometry": 0,
            "sbo_term_name": "stimulator",
            "evidence": "weeeee",
            "curator": "Sean",
        }
    ]
)

curation_dict["remove"] = pd.DataFrame(
    [
        {"remove": "reaction_1237042", "table": "reactions", "variable": "r_id"},
        {
            "remove": "CYB5Rs reduce MetHb to HbA",
            "table": "reactions",
            "variable": "r_name",
        },
        {"remove": "CO2", "table": "species", "variable": "s_name"},
    ]
)


def test_remove_entities():
    sbml_model = sbml.SBML(sbml_path)
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    sbml_dfs.validate()

    invalid_entities_dict = curation._find_invalid_entities(
        sbml_dfs, curation_dict["remove"]
    )
    invalid_pks = set(invalid_entities_dict.keys())

    assert invalid_pks == {"sc_id", "rsc_id", "r_id", "s_id"}

    n_species = sbml_dfs.species.shape[0]
    n_reactions = sbml_dfs.reactions.shape[0]
    n_compartmentalized_species = sbml_dfs.compartmentalized_species.shape[0]
    n_reaction_species = sbml_dfs.reaction_species.shape[0]
    # should be untouched
    n_compartments = sbml_dfs.compartments.shape[0]

    sbml_dfs = curation._remove_entities(sbml_dfs, invalid_entities_dict)

    assert n_species - sbml_dfs.species.shape[0] == 1
    assert n_reactions - sbml_dfs.reactions.shape[0] == 2
    assert (
        n_compartmentalized_species - sbml_dfs.compartmentalized_species.shape[0] == 2
    )
    assert n_reaction_species - sbml_dfs.reaction_species.shape[0] == 14
    assert n_compartments - sbml_dfs.compartments.shape[0] == 0


def test_add_entities():
    sbml_model = sbml.SBML(sbml_path)
    sbml_dfs = sbml_dfs_core.SBML_dfs(sbml_model)
    sbml_dfs.validate()

    new_entities = curation.format_curations(curation_dict, sbml_dfs)

    assert new_entities["species"].shape == (2, 3)
    assert new_entities["reactions"].shape == (2, 4)
    assert new_entities["compartmentalized_species"].shape == (1, 4)
    assert new_entities["reaction_species"].shape == (5, 4)


################################################
# __main__
################################################

if __name__ == "__main__":
    test_remove_entities()
    test_add_entities()
