from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from napistu import identifiers
from napistu import sbml_dfs_core
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source
from napistu.ingestion import sbml
from napistu.modify import pathwayannot

from napistu import identifiers as napistu_identifiers
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    SBML_DFS,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
    ONTOLOGIES,
)
from napistu.ingestion.constants import (
    INTERACTION_EDGELIST_DEFS,
    INTERACTION_EDGELIST_DEFAULTS,
)


@pytest.fixture
def test_data():
    """Create test data for SBML integration tests."""

    blank_id = identifiers.Identifiers([])

    # Test compartments
    compartments_df = pd.DataFrame(
        [
            {SBML_DFS.C_NAME: "nucleus", SBML_DFS.C_IDENTIFIERS: blank_id},
            {SBML_DFS.C_NAME: "cytoplasm", SBML_DFS.C_IDENTIFIERS: blank_id},
        ]
    )

    # Test species with extra data
    species_df = pd.DataFrame(
        [
            {
                SBML_DFS.S_NAME: "TP53",
                SBML_DFS.S_IDENTIFIERS: blank_id,
                "gene_type": "tumor_suppressor",
            },
            {
                SBML_DFS.S_NAME: "MDM2",
                SBML_DFS.S_IDENTIFIERS: blank_id,
                "gene_type": "oncogene",
            },
            {
                SBML_DFS.S_NAME: "CDKN1A",
                SBML_DFS.S_IDENTIFIERS: blank_id,
                "gene_type": "cell_cycle",
            },
        ]
    )

    # Test interactions with extra data
    interaction_edgelist = pd.DataFrame(
        [
            {
                INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: "TP53",
                INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: "CDKN1A",
                INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: "nucleus",
                INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: "nucleus",
                SBML_DFS.R_NAME: "TP53_activates_CDKN1A",
                INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.STIMULATOR,
                SBML_DFS.R_IDENTIFIERS: blank_id,
                "confidence": 0.95,
            },
            {
                INTERACTION_EDGELIST_DEFS.UPSTREAM_NAME: "MDM2",
                INTERACTION_EDGELIST_DEFS.DOWNSTREAM_NAME: "TP53",
                INTERACTION_EDGELIST_DEFS.UPSTREAM_COMPARTMENT: "cytoplasm",
                INTERACTION_EDGELIST_DEFS.DOWNSTREAM_COMPARTMENT: "nucleus",
                SBML_DFS.R_NAME: "MDM2_inhibits_TP53",
                INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME: SBOTERM_NAMES.INHIBITOR,
                SBML_DFS.R_IDENTIFIERS: blank_id,
                "confidence": 0.87,
            },
        ]
    )

    return [interaction_edgelist, species_df, compartments_df]


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
    dup_idx = pd.Series([an_s_id, an_s_id], name=SBML_DFS.S_ID)
    data = pd.DataFrame({"bla": [1, 2]}, index=dup_idx)

    with pytest.raises(ValueError):
        sbml_dfs.add_species_data("test", data)


def test_sbml_dfs_species_data_wrong_idx(sbml_dfs):
    data = pd.DataFrame(
        {"bla": [1, 2, 3]},
        index=pd.Series(["bla1", "bla2", "bla3"], name=SBML_DFS.S_ID),
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
    dup_idx = pd.Series([an_r_id, an_r_id], name=SBML_DFS.R_ID)
    data = pd.DataFrame({"bla": [1, 2]}, index=dup_idx)
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", data)


def test_sbml_dfs_reactions_data_wrong_idx(sbml_dfs):
    data = pd.DataFrame(
        {"bla": [1, 2, 3]},
        index=pd.Series(["bla1", "bla2", "bla3"], name=SBML_DFS.R_ID),
    )
    with pytest.raises(ValueError):
        sbml_dfs.add_reactions_data("test", data)


def test_sbml_dfs_remove_species_check_species(sbml_dfs):
    s_id = [sbml_dfs.species.index[0]]
    sbml_dfs._remove_species(s_id)
    assert s_id[0] not in sbml_dfs.species.index
    sbml_dfs.validate()


def test_sbml_dfs_remove_species_check_cspecies(sbml_dfs):
    s_id = [sbml_dfs.compartmentalized_species[SBML_DFS.S_ID].iloc[0]]
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
    sc_id = [sbml_dfs.reaction_species[SBML_DFS.SC_ID].iloc[0]]
    sbml_dfs._remove_compartmentalized_species(sc_id)
    assert sc_id[0] not in sbml_dfs.reaction_species[SBML_DFS.SC_ID]
    sbml_dfs.validate()


def test_sbml_dfs_remove_reactions_check_reactions(sbml_dfs):
    r_id = [sbml_dfs.reactions.index[0]]
    sbml_dfs.remove_reactions(r_id)
    assert r_id[0] not in sbml_dfs.reactions.index
    sbml_dfs.validate()


def test_sbml_dfs_remove_reactions_check_reaction_species(sbml_dfs):
    r_id = [sbml_dfs.reaction_species[SBML_DFS.R_ID].iloc[0]]
    sbml_dfs.remove_reactions(r_id)
    assert r_id[0] not in sbml_dfs.reaction_species[SBML_DFS.R_ID]
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
    dat = sbml_dfs.compartmentalized_species.query("s_id == @s_id").merge(
        sbml_dfs.reaction_species, left_index=True, right_on=SBML_DFS.SC_ID
    )
    r_ids = dat[SBML_DFS.R_ID].unique()
    sbml_dfs.remove_reactions(r_ids, remove_species=True)
    assert s_id not in sbml_dfs.species.index
    sbml_dfs.validate()


def test_read_sbml_with_invalid_ids():
    SBML_W_BAD_IDS = "R-HSA-166658.sbml"
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_w_bad_ids_path = os.path.join(test_path, "test_data", SBML_W_BAD_IDS)
    assert os.path.isfile(sbml_w_bad_ids_path)

    # invalid identifiers still create a valid sbml_dfs
    sbml_w_bad_ids = sbml.SBML(sbml_w_bad_ids_path)
    assert isinstance(sbml_dfs_core.SBML_dfs(sbml_w_bad_ids), sbml_dfs_core.SBML_dfs)


def test_get_table(sbml_dfs):
    assert isinstance(sbml_dfs.get_table(SBML_DFS.SPECIES), pd.DataFrame)
    assert isinstance(
        sbml_dfs.get_table(SBML_DFS.SPECIES, {SCHEMA_DEFS.ID}), pd.DataFrame
    )

    # invalid table
    with pytest.raises(ValueError):
        sbml_dfs.get_table("foo", {SCHEMA_DEFS.ID})

    # bad type
    with pytest.raises(TypeError):
        sbml_dfs.get_table(SBML_DFS.REACTION_SPECIES, SCHEMA_DEFS.ID)

    # reaction species don't have ids
    with pytest.raises(ValueError):
        sbml_dfs.get_table(SBML_DFS.REACTION_SPECIES, {SCHEMA_DEFS.ID})


def test_search_by_name(sbml_dfs_metabolism):
    assert (
        sbml_dfs_metabolism.search_by_name("atp", SBML_DFS.SPECIES, False).shape[0] == 1
    )
    assert sbml_dfs_metabolism.search_by_name("pyr", SBML_DFS.SPECIES).shape[0] == 3
    assert (
        sbml_dfs_metabolism.search_by_name("kinase", SBML_DFS.REACTIONS).shape[0] == 4
    )


def test_search_by_id(sbml_dfs_metabolism):
    identifiers_tbl = sbml_dfs_metabolism.get_identifiers(SBML_DFS.SPECIES)
    ids, species = sbml_dfs_metabolism.search_by_ids(
        identifiers_tbl, identifiers=["P40926"]
    )
    assert ids.shape[0] == 1
    assert species.shape[0] == 1

    ids, species = sbml_dfs_metabolism.search_by_ids(
        identifiers_tbl,
        identifiers=["57540", "30744"],
        ontologies={ONTOLOGIES.CHEBI},
    )
    assert ids.shape[0] == 2
    assert species.shape[0] == 2

    with pytest.raises(
        ValueError, match="None of the requested identifiers are present"
    ):
        ids, species = sbml_dfs_metabolism.search_by_ids(
            identifiers_tbl, identifiers=["baz"]  # Non-existent identifier
        )


def test_species_status(sbml_dfs):

    species = sbml_dfs.species
    select_species = species[species[SBML_DFS.S_NAME] == "OxyHbA"]
    assert select_species.shape[0] == 1

    status = sbml_dfs.species_status(select_species.index[0])

    # expected columns
    expected_columns = [
        SBML_DFS.SC_NAME,
        SBML_DFS.STOICHIOMETRY,
        SBML_DFS.R_NAME,
        "r_formula_str",
    ]
    assert all(col in status.columns for col in expected_columns)

    assert (
        status["r_formula_str"][0]
        == "cytosol: 4.0 CO2 + 4.0 H+ + OxyHbA -> 4.0 O2 + Protonated Carbamino DeoxyHbA"
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


def test_remove_entity_data_success(sbml_dfs_w_data):
    """Test successful removal of entity data."""
    # Get initial data
    initial_species_data_keys = set(sbml_dfs_w_data.species_data.keys())
    initial_reactions_data_keys = set(sbml_dfs_w_data.reactions_data.keys())

    # Remove species data
    sbml_dfs_w_data._remove_entity_data(SBML_DFS.SPECIES, "test_species")
    assert "test_species" not in sbml_dfs_w_data.species_data
    assert set(sbml_dfs_w_data.species_data.keys()) == initial_species_data_keys - {
        "test_species"
    }

    # Remove reactions data
    sbml_dfs_w_data._remove_entity_data(SBML_DFS.REACTIONS, "test_reactions")
    assert "test_reactions" not in sbml_dfs_w_data.reactions_data
    assert set(sbml_dfs_w_data.reactions_data.keys()) == initial_reactions_data_keys - {
        "test_reactions"
    }

    # Validate the model is still valid after removals
    sbml_dfs_w_data.validate()


def test_remove_entity_data_nonexistent(sbml_dfs_w_data, caplog):
    """Test warning when trying to remove nonexistent entity data."""
    # Try to remove nonexistent species data
    sbml_dfs_w_data._remove_entity_data(SBML_DFS.SPECIES, "nonexistent_label")
    assert "Label 'nonexistent_label' not found in species_data" in caplog.text
    assert set(sbml_dfs_w_data.species_data.keys()) == {"test_species"}

    # Clear the log
    caplog.clear()

    # Try to remove nonexistent reactions data
    sbml_dfs_w_data._remove_entity_data(SBML_DFS.REACTIONS, "nonexistent_label")
    assert "Label 'nonexistent_label' not found in reactions_data" in caplog.text
    assert set(sbml_dfs_w_data.reactions_data.keys()) == {"test_reactions"}

    # Validate the model is still valid
    sbml_dfs_w_data.validate()


def test_get_characteristic_species_ids():
    """
    Test get_characteristic_species_ids function with both dogmatic and non-dogmatic cases.
    """
    # Create mock species identifiers data
    mock_species_ids = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["s1", "s2", "s3", "s4", "s5"],
            IDENTIFIERS.IDENTIFIER: [
                "P12345",
                "CHEBI:15377",
                "GO:12345",
                "P67890",
                "P67890",
            ],
            IDENTIFIERS.ONTOLOGY: ["uniprot", "chebi", "go", "uniprot", "chebi"],
            IDENTIFIERS.BQB: [
                BQB.IS,
                BQB.IS,
                BQB.HAS_PART,
                BQB.HAS_VERSION,
                BQB.ENCODES,
            ],
        }
    )

    # Create minimal required tables for SBML_dfs
    compartments = pd.DataFrame(
        {SBML_DFS.C_NAME: ["cytosol"], SBML_DFS.C_IDENTIFIERS: [None]}, index=["C1"]
    )
    compartments.index.name = SBML_DFS.C_ID
    species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A"],
            SBML_DFS.S_IDENTIFIERS: [None],
            SBML_DFS.S_SOURCE: [None],
        },
        index=["s1"],
    )
    species.index.name = SBML_DFS.S_ID
    compartmentalized_species = pd.DataFrame(
        {
            SBML_DFS.SC_NAME: ["A [cytosol]"],
            SBML_DFS.S_ID: ["s1"],
            SBML_DFS.C_ID: ["C1"],
            SBML_DFS.SC_SOURCE: [None],
        },
        index=["SC1"],
    )
    compartmentalized_species.index.name = SBML_DFS.SC_ID
    reactions = pd.DataFrame(
        {
            SBML_DFS.R_NAME: ["rxn1"],
            SBML_DFS.R_IDENTIFIERS: [None],
            SBML_DFS.R_SOURCE: [None],
            SBML_DFS.R_ISREVERSIBLE: [False],
        },
        index=["R1"],
    )
    reactions.index.name = SBML_DFS.R_ID
    reaction_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: ["R1"],
            SBML_DFS.SC_ID: ["SC1"],
            SBML_DFS.STOICHIOMETRY: [1],
            SBML_DFS.SBO_TERM: ["SBO:0000459"],
        },
        index=["RSC1"],
    )
    reaction_species.index.name = SBML_DFS.RSC_ID

    sbml_dict = {
        SBML_DFS.COMPARTMENTS: compartments,
        SBML_DFS.SPECIES: species,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species,
        SBML_DFS.REACTIONS: reactions,
        SBML_DFS.REACTION_SPECIES: reaction_species,
    }
    sbml_dfs = SBML_dfs(sbml_dict, validate=False, resolve=False)

    # Test dogmatic case (default)
    expected_bqbs = BQB_DEFINING_ATTRS + [BQB.HAS_PART]  # noqa: F841
    with patch.object(sbml_dfs, "get_identifiers", return_value=mock_species_ids):
        dogmatic_result = sbml_dfs.get_characteristic_species_ids()
        expected_dogmatic = mock_species_ids.query("bqb in @expected_bqbs")
        pd.testing.assert_frame_equal(
            dogmatic_result, expected_dogmatic, check_like=True
        )

    # Test non-dogmatic case
    expected_bqbs = BQB_DEFINING_ATTRS_LOOSE + [BQB.HAS_PART]  # noqa: F841
    with patch.object(sbml_dfs, "get_identifiers", return_value=mock_species_ids):
        non_dogmatic_result = sbml_dfs.get_characteristic_species_ids(dogmatic=False)
        expected_non_dogmatic = mock_species_ids.query("bqb in @expected_bqbs")
        pd.testing.assert_frame_equal(
            non_dogmatic_result, expected_non_dogmatic, check_like=True
        )


def test_sbml_basic_functionality(test_data):
    """Test basic SBML_dfs creation from edgelist."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species_df, compartments_df
    )

    assert isinstance(result, SBML_dfs)
    assert len(result.species) == 3
    assert len(result.compartments) == 2
    assert len(result.reactions) == 2
    assert (
        len(result.compartmentalized_species) == 3
    )  # TP53[nucleus], CDKN1A[nucleus], MDM2[cytoplasm]
    assert len(result.reaction_species) == 4  # 2 reactions * 2 species each


def test_sbml_extra_data_preservation(test_data):
    """Test that extra columns are preserved when requested."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        keep_species_data=True,
        keep_reactions_data="experiment",
    )

    assert hasattr(result, SBML_DFS.SPECIES_DATA)
    assert hasattr(result, SBML_DFS.REACTIONS_DATA)
    assert "gene_type" in result.species_data["source"].columns
    assert "confidence" in result.reactions_data["experiment"].columns


def test_sbml_compartmentalized_naming(test_data):
    """Test compartmentalized species naming convention."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species_df, compartments_df
    )

    comp_names = result.compartmentalized_species[SBML_DFS.SC_NAME].tolist()
    assert "TP53 [nucleus]" in comp_names
    assert "MDM2 [cytoplasm]" in comp_names
    assert "CDKN1A [nucleus]" in comp_names


def test_sbml_custom_defaults(test_data):
    """Test custom stoichiometry parameters."""
    interaction_edgelist, species_df, compartments_df = test_data

    custom_defaults = INTERACTION_EDGELIST_DEFAULTS.copy()
    custom_defaults[INTERACTION_EDGELIST_DEFS.UPSTREAM_STOICHIOMETRY] = -2
    custom_defaults[INTERACTION_EDGELIST_DEFS.DOWNSTREAM_STOICHIOMETRY] = 3
    custom_defaults[INTERACTION_EDGELIST_DEFS.UPSTREAM_SBO_TERM_NAME] = (
        SBOTERM_NAMES.REACTANT
    )
    custom_defaults[INTERACTION_EDGELIST_DEFS.DOWNSTREAM_SBO_TERM_NAME] = (
        SBOTERM_NAMES.PRODUCT
    )

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        interaction_edgelist_defaults=custom_defaults,
    )

    stoichiometries = result.reaction_species[SBML_DFS.STOICHIOMETRY].unique()
    assert -2 in stoichiometries  # upstream
    assert 3 in stoichiometries  # downstream
    assert (
        MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]
        in result.reaction_species[SBML_DFS.SBO_TERM].unique()
    )
    # upstream sbo terms are provided so the default shouldn't be used
    assert (
        MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]
        not in result.reaction_species[SBML_DFS.SBO_TERM].unique()
    )


def test_validate_schema_missing(minimal_valid_sbml_dfs):
    """Test validation fails when schema is missing."""
    delattr(minimal_valid_sbml_dfs, "schema")
    with pytest.raises(ValueError, match="No schema found"):
        minimal_valid_sbml_dfs.validate()


def test_validate_table(minimal_valid_sbml_dfs):
    """Test _validate_table fails for various table structure issues."""
    # Wrong index name
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.species.index.name = "wrong_name"
    with pytest.raises(ValueError, match="the index name for species was not the pk"):
        sbml_dfs.validate()

    # Duplicate primary keys
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    duplicate_species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["ATP", "ADP"],
            SBML_DFS.S_IDENTIFIERS: [
                identifiers.Identifiers([]),
                identifiers.Identifiers([]),
            ],
            SBML_DFS.S_SOURCE: [Source(init=True), Source(init=True)],
        },
        index=pd.Index(["S00001", "S00001"], name=SBML_DFS.S_ID),
    )
    sbml_dfs.species = duplicate_species
    with pytest.raises(ValueError, match="primary keys were duplicated"):
        sbml_dfs.validate()

    # Missing required variables
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.species = sbml_dfs.species.drop(columns=[SBML_DFS.S_NAME])
    with pytest.raises(ValueError, match="Missing .+ required variables for species"):
        sbml_dfs.validate()

    # Empty table
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: [],
            SBML_DFS.S_IDENTIFIERS: [],
            SBML_DFS.S_SOURCE: [],
        },
        index=pd.Index([], name=SBML_DFS.S_ID),
    )
    with pytest.raises(ValueError, match="species contained no entries"):
        sbml_dfs.validate()


def test_check_pk_fk_correspondence(minimal_valid_sbml_dfs):
    """Test _check_pk_fk_correspondence fails for various foreign key issues."""
    # Missing species reference
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.compartmentalized_species[SBML_DFS.S_ID] = ["S99999"]
    with pytest.raises(
        ValueError,
        match="s_id values were found in compartmentalized_species but missing from species",
    ):
        sbml_dfs.validate()

    # Missing compartment reference
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.compartmentalized_species[SBML_DFS.C_ID] = ["C99999"]
    with pytest.raises(
        ValueError,
        match="c_id values were found in compartmentalized_species but missing from compartments",
    ):
        sbml_dfs.validate()

    # Null foreign keys
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.compartmentalized_species[SBML_DFS.S_ID] = [None]
    with pytest.raises(
        ValueError, match="compartmentalized_species included missing s_id values"
    ):
        sbml_dfs.validate()


def test_validate_reaction_species(minimal_valid_sbml_dfs):
    """Test _validate_reaction_species fails for various reaction species issues."""
    # Null stoichiometry
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] = [None]
    with pytest.raises(ValueError, match="All reaction_species.* must be not null"):
        sbml_dfs.validate()

    # Null SBO terms
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.reaction_species[SBML_DFS.SBO_TERM] = [None]
    with pytest.raises(
        ValueError, match="sbo_terms were None; all terms should be defined"
    ):
        sbml_dfs.validate()

    # Invalid SBO terms
    sbml_dfs = minimal_valid_sbml_dfs.copy()
    sbml_dfs.reaction_species[SBML_DFS.SBO_TERM] = ["INVALID_SBO_TERM"]
    with pytest.raises(ValueError, match="sbo_terms were not defined"):
        sbml_dfs.validate()


def test_validate_identifiers(minimal_valid_sbml_dfs):
    """Test _validate_identifiers fails when identifiers are missing."""
    minimal_valid_sbml_dfs.species[SBML_DFS.S_IDENTIFIERS] = [None]
    with pytest.raises(ValueError, match="species has .+ missing ids"):
        minimal_valid_sbml_dfs.validate()


def test_validate_sources(minimal_valid_sbml_dfs):
    """Test _validate_sources fails when sources are missing."""
    minimal_valid_sbml_dfs.species[SBML_DFS.S_SOURCE] = [None]
    with pytest.raises(ValueError, match="species has .+ missing sources"):
        minimal_valid_sbml_dfs.validate()


def test_validate_species_data(minimal_valid_sbml_dfs):
    """Test _validate_species_data fails when species_data has invalid structure."""
    invalid_data = pd.DataFrame(
        {"extra_info": ["test"]}, index=pd.Index(["S99999"], name=SBML_DFS.S_ID)
    )  # Non-existent species
    minimal_valid_sbml_dfs.species_data["invalid"] = invalid_data
    with pytest.raises(ValueError, match="species data invalid was invalid"):
        minimal_valid_sbml_dfs.validate()


def test_validate_reactions_data(minimal_valid_sbml_dfs):
    """Test _validate_reactions_data fails when reactions_data has invalid structure."""
    invalid_data = pd.DataFrame(
        {"extra_info": ["test"]}, index=pd.Index(["R99999"], name=SBML_DFS.R_ID)
    )  # Non-existent reaction
    minimal_valid_sbml_dfs.reactions_data["invalid"] = invalid_data
    with pytest.raises(ValueError, match="reactions data invalid was invalid"):
        minimal_valid_sbml_dfs.validate()


def test_validate_passes_with_valid_data(minimal_valid_sbml_dfs):
    """Test that validation passes with completely valid data."""
    minimal_valid_sbml_dfs.validate()  # Should not raise any exceptions
