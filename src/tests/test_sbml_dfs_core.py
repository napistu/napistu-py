from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.modify import pathwayannot

from napistu import identifiers as napistu_identifiers
from napistu.constants import SBML_DFS
from napistu.sbml_dfs_core import SBML_dfs


def test_drop_cofactors(sbml_dfs):
    starting_rscs = sbml_dfs.reaction_species.shape[0]
    reduced_dfs = pathwayannot.drop_cofactors(sbml_dfs)

    assert starting_rscs - reduced_dfs.reaction_species.shape[0] == 20


def test_sbml_dfs_from_dict_required(sbml_dfs):
    val_dict = {k: getattr(sbml_dfs, k) for k in sbml_dfs._required_entities}
    sbml_dfs2 = sbml_dfs_core.SBML_dfs(val_dict)
    sbml_dfs2.validate()

    for k in sbml_dfs._required_entities:
        assert getattr(sbml_dfs2, k).equals(getattr(sbml_dfs, k))


def test_sbml_dfs_species_data(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]}, index=sbml_dfs.species.iloc[:3].index)
    sbml_dfs.add_species_data("test", data)
    sbml_dfs.validate()


def test_sbml_dfs_species_data_existing(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]}, index=sbml_dfs.species.iloc[:3].index)
    sbml_dfs.add_species_data("test", data)
    with pytest.raises(ValueError):
        sbml_dfs.add_species_data("test", data)


def test_sbml_dfs_species_data_validation(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]})
    sbml_dfs.species_data["test"] = data
    with pytest.raises(ValueError):
        sbml_dfs.validate()


def test_sbml_dfs_species_data_missing_idx(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]})
    with pytest.raises(ValueError):
        sbml_dfs.add_species_data("test", data)


def test_sbml_dfs_species_data_duplicated_idx(sbml_dfs):
    an_s_id = sbml_dfs.species.iloc[0].index[0]
    dup_idx = pd.Series([an_s_id, an_s_id], name="s_id")
    data = pd.DataFrame({"bla": [1, 2]}, index=dup_idx)

    with pytest.raises(ValueError):
        sbml_dfs.add_species_data("test", data)


def test_sbml_dfs_species_data_wrong_idx(sbml_dfs):
    data = pd.DataFrame(
        {"bla": [1, 2, 3]}, index=pd.Series(["bla1", "bla2", "bla3"], name="s_id")
    )
    with pytest.raises(ValueError):
        sbml_dfs.add_species_data("test", data)


def test_sbml_dfs_reactions_data(sbml_dfs):
    reactions_data = pd.DataFrame(
        {"bla": [1, 2, 3]}, index=sbml_dfs.reactions.iloc[:3].index
    )
    sbml_dfs.add_reactions_data("test", reactions_data)
    sbml_dfs.validate()


def test_sbml_dfs_reactions_data_existing(sbml_dfs):
    reactions_data = pd.DataFrame(
        {"bla": [1, 2, 3]}, index=sbml_dfs.reactions.iloc[:3].index
    )
    sbml_dfs.add_reactions_data("test", reactions_data)
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", reactions_data)


def test_sbml_dfs_reactions_data_validate(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]})
    sbml_dfs.reactions_data["test"] = data
    with pytest.raises(ValueError):
        sbml_dfs.validate()


def test_sbml_dfs_reactions_data_missing_idx(sbml_dfs):
    data = pd.DataFrame({"bla": [1, 2, 3]})
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", data)


def test_sbml_dfs_reactions_data_duplicated_idx(sbml_dfs):
    an_r_id = sbml_dfs.reactions.iloc[0].index[0]
    dup_idx = pd.Series([an_r_id, an_r_id], name="r_id")
    data = pd.DataFrame({"bla": [1, 2]}, index=dup_idx)
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", data)


def test_sbml_dfs_reactions_data_wrong_idx(sbml_dfs):
    data = pd.DataFrame(
        {"bla": [1, 2, 3]}, index=pd.Series(["bla1", "bla2", "bla3"], name="r_id")
    )
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", data)


def test_sbml_dfs_remove_species_check_species(sbml_dfs):
    s_id = [sbml_dfs.species.index[0]]
    sbml_dfs._remove_species(s_id)
    assert s_id[0] not in sbml_dfs.species.index
    sbml_dfs.validate()


def test_sbml_dfs_remove_species_check_cspecies(sbml_dfs):
    s_id = [sbml_dfs.compartmentalized_species["s_id"].iloc[0]]
    sbml_dfs._remove_species(s_id)
    assert s_id[0] not in sbml_dfs.compartmentalized_species.index
    sbml_dfs.validate()


@pytest.fixture
def sbml_dfs_w_data(sbml_dfs):
    sbml_dfs.add_species_data(
        "test_species",
        pd.DataFrame({"test1": [1, 2]}, index=sbml_dfs.species.index[:2]),
    )
    sbml_dfs.add_reactions_data(
        "test_reactions",
        pd.DataFrame({"test2": [1, 2, 3]}, index=sbml_dfs.reactions.index[:3]),
    )
    return sbml_dfs


def test_sbml_dfs_remove_species_check_data(sbml_dfs_w_data):
    data = list(sbml_dfs_w_data.species_data.values())[0]
    s_id = [data.index[0]]
    sbml_dfs_w_data._remove_species(s_id)
    data_2 = list(sbml_dfs_w_data.species_data.values())[0]
    assert s_id[0] not in data_2.index
    sbml_dfs_w_data.validate()


def test_sbml_dfs_remove_cspecies_check_cspecies(sbml_dfs):
    s_id = [sbml_dfs.compartmentalized_species.index[0]]
    sbml_dfs._remove_compartmentalized_species(s_id)
    assert s_id[0] not in sbml_dfs.compartmentalized_species.index
    sbml_dfs.validate()


def test_sbml_dfs_remove_cspecies_check_reaction_species(sbml_dfs):
    sc_id = [sbml_dfs.reaction_species["sc_id"].iloc[0]]
    sbml_dfs._remove_compartmentalized_species(sc_id)
    assert sc_id[0] not in sbml_dfs.reaction_species["sc_id"]
    sbml_dfs.validate()


def test_sbml_dfs_remove_reactions_check_reactions(sbml_dfs):
    r_id = [sbml_dfs.reactions.index[0]]
    sbml_dfs.remove_reactions(r_id)
    assert r_id[0] not in sbml_dfs.reactions.index
    sbml_dfs.validate()


def test_sbml_dfs_remove_reactions_check_reaction_species(sbml_dfs):
    r_id = [sbml_dfs.reaction_species["r_id"].iloc[0]]
    sbml_dfs.remove_reactions(r_id)
    assert r_id[0] not in sbml_dfs.reaction_species["r_id"]
    sbml_dfs.validate()


def test_sbml_dfs_remove_reactions_check_data(sbml_dfs_w_data):
    data = list(sbml_dfs_w_data.reactions_data.values())[0]
    r_id = [data.index[0]]
    sbml_dfs_w_data.remove_reactions(r_id)
    data_2 = list(sbml_dfs_w_data.reactions_data.values())[0]
    assert r_id[0] not in data_2.index
    sbml_dfs_w_data.validate()


def test_sbml_dfs_remove_reactions_check_species(sbml_dfs):
    # find all r_ids for a species and check if
    # removing all these reactions also removes the species
    s_id = sbml_dfs.species.index[0]
    dat = (
        sbml_dfs.compartmentalized_species.query("s_id == @s_id").merge(
            sbml_dfs.reaction_species, on="sc_id"
        )
    )[["r_id", "sc_id"]]
    r_ids = dat["r_id"]
    sc_ids = dat["sc_id"]
    sbml_dfs.remove_reactions(r_ids, remove_species=True)
    for sc_id in sc_ids:
        assert sc_id not in sbml_dfs.compartmentalized_species.index
    assert s_id not in sbml_dfs.species.index


def test_formula(sbml_dfs):
    # create a formula string

    an_r_id = sbml_dfs.reactions.index[0]

    reaction_species_df = sbml_dfs.reaction_species[
        sbml_dfs.reaction_species["r_id"] == an_r_id
    ].merge(sbml_dfs.compartmentalized_species, left_on="sc_id", right_index=True)

    formula_str = sbml_dfs_core.construct_formula_string(
        reaction_species_df, sbml_dfs.reactions, name_var="sc_name"
    )

    assert isinstance(formula_str, str)
    assert (
        formula_str
        == "CO2 [extracellular region] -> CO2 [cytosol] ---- modifiers: AQP1 tetramer [plasma membrane]]"
    )


def test_read_sbml_with_invalid_ids():
    SBML_W_BAD_IDS = "R-HSA-166658.sbml"
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_w_bad_ids_path = os.path.join(test_path, "test_data", SBML_W_BAD_IDS)
    assert os.path.isfile(sbml_w_bad_ids_path)

    # invalid identifiers still create a valid sbml_dfs
    sbml_w_bad_ids = sbml.SBML(sbml_w_bad_ids_path)
    assert isinstance(sbml_dfs_core.SBML_dfs(sbml_w_bad_ids), sbml_dfs_core.SBML_dfs)


def test_stubbed_compartment():
    compartment = sbml_dfs_core._stub_compartments()

    assert compartment["c_Identifiers"].iloc[0].ids[0] == {
        "ontology": "go",
        "identifier": "GO:0005575",
        "url": "https://www.ebi.ac.uk/QuickGO/term/GO:0005575",
        "bqb": "BQB_IS",
    }


def test_get_table(sbml_dfs):
    assert isinstance(sbml_dfs.get_table("species"), pd.DataFrame)
    assert isinstance(sbml_dfs.get_table("species", {"id"}), pd.DataFrame)

    # invalid table
    with pytest.raises(ValueError):
        sbml_dfs.get_table("foo", {"id"})

    # bad type
    with pytest.raises(TypeError):
        sbml_dfs.get_table("reaction_species", "id")

    # reaction species don't have ids
    with pytest.raises(ValueError):
        sbml_dfs.get_table("reaction_species", {"id"})


def test_search_by_name(sbml_dfs_metabolism):
    assert sbml_dfs_metabolism.search_by_name("atp", "species", False).shape[0] == 1
    assert sbml_dfs_metabolism.search_by_name("pyr", "species").shape[0] == 3
    assert sbml_dfs_metabolism.search_by_name("kinase", "reactions").shape[0] == 4


def test_search_by_id(sbml_dfs_metabolism):
    identifiers_tbl = sbml_dfs_metabolism.get_identifiers("species")
    ids, species = sbml_dfs_metabolism.search_by_ids(
        ["P40926"], "species", identifiers_tbl
    )
    assert ids.shape[0] == 1
    assert species.shape[0] == 1

    ids, species = sbml_dfs_metabolism.search_by_ids(
        ["57540", "30744"], "species", identifiers_tbl, {"chebi"}
    )
    assert ids.shape[0] == 2
    assert species.shape[0] == 2

    ids, species = sbml_dfs_metabolism.search_by_ids(
        ["baz"], "species", identifiers_tbl
    )
    assert ids.shape[0] == 0
    assert species.shape[0] == 0


def test_species_status(sbml_dfs):

    species = sbml_dfs.species
    select_species = species[species["s_name"] == "OxyHbA"]
    assert select_species.shape[0] == 1

    status = sbml_dfs_core.species_status(select_species.index[0], sbml_dfs)
    assert (
        status["r_formula_str"][0]
        == "4.0 H+ + OxyHbA + 4.0 CO2 -> 4.0 O2 + Protonated Carbamino DeoxyHbA [cytosol]"
    )


def test_get_identifiers_handles_missing_values():

    # Minimal DataFrame with all types
    df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A", "B", "C", "D"],
            SBML_DFS.S_IDENTIFIERS: [
                napistu_identifiers.Identifiers([]),
                None,
                np.nan,
                pd.NA,
            ],
            SBML_DFS.S_SOURCE: [None, None, None, None],
        },
        index=["s1", "s2", "s3", "s4"],
    )
    df.index.name = SBML_DFS.S_ID

    sbml_dict = {
        SBML_DFS.COMPARTMENTS: pd.DataFrame(
            {
                SBML_DFS.C_NAME: ["cytosol"],
                SBML_DFS.C_IDENTIFIERS: [None],
                SBML_DFS.C_SOURCE: [None],
            },
            index=["c1"],
        ),
        SBML_DFS.SPECIES: df,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: pd.DataFrame(
            {
                SBML_DFS.SC_NAME: ["A [cytosol]"],
                SBML_DFS.S_ID: ["s1"],
                SBML_DFS.C_ID: ["c1"],
                SBML_DFS.SC_SOURCE: [None],
            },
            index=["sc1"],
        ),
        SBML_DFS.REACTIONS: pd.DataFrame(
            {
                SBML_DFS.R_NAME: [],
                SBML_DFS.R_IDENTIFIERS: [],
                SBML_DFS.R_SOURCE: [],
                SBML_DFS.R_ISREVERSIBLE: [],
            },
            index=[],
        ),
        SBML_DFS.REACTION_SPECIES: pd.DataFrame(
            {
                SBML_DFS.R_ID: [],
                SBML_DFS.SC_ID: [],
                SBML_DFS.STOICHIOMETRY: [],
                SBML_DFS.SBO_TERM: [],
            },
            index=[],
        ),
    }
    sbml = SBML_dfs(sbml_dict, validate=False)
    result = sbml.get_identifiers(SBML_DFS.SPECIES)
    assert result.shape[0] == 0 or all(
        result[SBML_DFS.S_ID] == "s1"
    ), "Only Identifiers objects should be returned."
