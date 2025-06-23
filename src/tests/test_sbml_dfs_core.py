from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from napistu import sbml_dfs_core
from napistu.source import Source
from napistu.ingestion import sbml
from napistu.modify import pathwayannot
from napistu.sbml_dfs_utils import _stub_ids

from napistu import identifiers as napistu_identifiers
from napistu.constants import (
    SBML_DFS,
    SBOTERM_NAMES,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    BQB,
    IDENTIFIERS,
)
from napistu.sbml_dfs_core import SBML_dfs


@pytest.fixture
def test_data():
    """Create test data for SBML integration tests."""

    # Test compartments
    compartments_df = pd.DataFrame(
        [
            {"c_name": "nucleus", "c_Identifiers": _stub_ids([])},
            {"c_name": "cytoplasm", "c_Identifiers": _stub_ids([])},
        ]
    )

    # Test species with extra data
    species_df = pd.DataFrame(
        [
            {
                "s_name": "TP53",
                "s_Identifiers": _stub_ids([]),
                "gene_type": "tumor_suppressor",
            },
            {"s_name": "MDM2", "s_Identifiers": _stub_ids([]), "gene_type": "oncogene"},
            {
                "s_name": "CDKN1A",
                "s_Identifiers": _stub_ids([]),
                "gene_type": "cell_cycle",
            },
        ]
    )

    # Test interactions with extra data
    interaction_edgelist = pd.DataFrame(
        [
            {
                "upstream_name": "TP53",
                "downstream_name": "CDKN1A",
                "upstream_compartment": "nucleus",
                "downstream_compartment": "nucleus",
                "r_name": "TP53_activates_CDKN1A",
                "sbo_term": "SBO:0000459",
                "r_Identifiers": _stub_ids([]),
                "r_isreversible": False,
                "confidence": 0.95,
            },
            {
                "upstream_name": "MDM2",
                "downstream_name": "TP53",
                "upstream_compartment": "cytoplasm",
                "downstream_compartment": "nucleus",
                "r_name": "MDM2_inhibits_TP53",
                "sbo_term": "SBO:0000020",
                "r_Identifiers": _stub_ids([]),
                "r_isreversible": False,
                "confidence": 0.87,
            },
        ]
    )

    return [interaction_edgelist, species_df, compartments_df, Source(init=True)]


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
    dat = sbml_dfs.compartmentalized_species.query("s_id == @s_id").merge(
        sbml_dfs.reaction_species, left_index=True, right_on="sc_id"
    )
    r_ids = dat["r_id"].unique()
    sbml_dfs.remove_reactions(r_ids, remove_species=True)
    assert s_id not in sbml_dfs.species.index
    sbml_dfs.validate()


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


def test_find_underspecified_reactions():

    reaction_w_regulators = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["A", "B", "C", "D", "E", "F", "G"],
            SBML_DFS.STOICHIOMETRY: [-1, -1, 1, 1, 0, 0, 0],
            SBML_DFS.SBO_TERM: [
                SBOTERM_NAMES.REACTANT,
                SBOTERM_NAMES.REACTANT,
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.PRODUCT,
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.CATALYST,
                SBOTERM_NAMES.STIMULATOR,
            ],
        }
    ).assign(r_id="bar")
    reaction_w_regulators[SBML_DFS.RSC_ID] = [
        f"rsc_{i}" for i in range(len(reaction_w_regulators))
    ]
    reaction_w_regulators.set_index(SBML_DFS.RSC_ID, inplace=True)
    reaction_w_regulators = sbml_dfs_core.add_sbo_role(reaction_w_regulators)

    reaction_w_interactors = pd.DataFrame(
        {
            SBML_DFS.SC_ID: ["A", "B"],
            SBML_DFS.STOICHIOMETRY: [-1, 1],
            SBML_DFS.SBO_TERM: [SBOTERM_NAMES.REACTANT, SBOTERM_NAMES.REACTANT],
        }
    ).assign(r_id="baz")
    reaction_w_interactors[SBML_DFS.RSC_ID] = [
        f"rsc_{i}" for i in range(len(reaction_w_interactors))
    ]
    reaction_w_interactors.set_index(SBML_DFS.RSC_ID, inplace=True)
    reaction_w_interactors = sbml_dfs_core.add_sbo_role(reaction_w_interactors)

    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_0", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing one enzyme -> operable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_4", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == set()

    # missing one product -> inoperable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_2", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing all enzymes -> inoperable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_4", "new"] = False
    working_reactions.loc["rsc_5", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == {"bar"}

    # missing regulators -> operable
    working_reactions = reaction_w_regulators.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_6", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == set()

    # remove an interactor
    working_reactions = reaction_w_interactors.copy()
    working_reactions["new"] = True
    working_reactions.loc["rsc_0", "new"] = False
    working_reactions
    result = sbml_dfs_core.find_underspecified_reactions(working_reactions)
    assert result == {"baz"}


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


def test_filter_to_characteristic_species_ids():

    species_ids_dict = {
        SBML_DFS.S_ID: ["large_complex"] * 6
        + ["small_complex"] * 2
        + ["proteinA", "proteinB"]
        + ["proteinC"] * 3
        + [
            "promiscuous_complexA",
            "promiscuous_complexB",
            "promiscuous_complexC",
            "promiscuous_complexD",
            "promiscuous_complexE",
        ],
        IDENTIFIERS.ONTOLOGY: ["complexportal"]
        + ["HGNC"] * 7
        + ["GO"] * 2
        + ["ENSG", "ENSP", "pubmed"]
        + ["HGNC"] * 5,
        IDENTIFIERS.IDENTIFIER: [
            "CPX-BIG",
            "mem1",
            "mem2",
            "mem3",
            "mem4",
            "mem5",
            "part1",
            "part2",
            "GO:1",
            "GO:2",
            "dna_seq",
            "protein_seq",
            "my_cool_pub",
        ]
        + ["promiscuous_complex"] * 5,
        IDENTIFIERS.BQB: [BQB.IS]
        + [BQB.HAS_PART] * 7
        + [BQB.IS] * 2
        + [
            # these are retained if BQB_DEFINING_ATTRS_LOOSE is used
            BQB.ENCODES,
            BQB.IS_ENCODED_BY,
            # this should always be removed
            BQB.IS_DESCRIBED_BY,
        ]
        + [BQB.HAS_PART] * 5,
    }

    species_ids = pd.DataFrame(species_ids_dict)

    characteristic_ids_narrow = sbml_dfs_core.filter_to_characteristic_species_ids(
        species_ids,
        defining_biological_qualifiers=BQB_DEFINING_ATTRS,
        max_complex_size=4,
        max_promiscuity=4,
    )

    EXPECTED_IDS = ["CPX-BIG", "GO:1", "GO:2", "part1", "part2"]
    assert characteristic_ids_narrow[IDENTIFIERS.IDENTIFIER].tolist() == EXPECTED_IDS

    characteristic_ids_loose = sbml_dfs_core.filter_to_characteristic_species_ids(
        species_ids,
        # include encodes and is_encoded_by as equivalent to is
        defining_biological_qualifiers=BQB_DEFINING_ATTRS_LOOSE,
        max_complex_size=4,
        # expand promiscuity to default value
        max_promiscuity=20,
    )

    EXPECTED_IDS = [
        "CPX-BIG",
        "GO:1",
        "GO:2",
        "dna_seq",
        "protein_seq",
        "part1",
        "part2",
    ] + ["promiscuous_complex"] * 5
    assert characteristic_ids_loose[IDENTIFIERS.IDENTIFIER].tolist() == EXPECTED_IDS


def test_sbml_basic_functionality(test_data):
    """Test basic SBML_dfs creation from edgelist."""
    interaction_edgelist, species_df, compartments_df, interaction_source = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species_df, compartments_df, interaction_source
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
    interaction_edgelist, species_df, compartments_df, interaction_source = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        interaction_source,
        keep_species_data=True,
        keep_reactions_data="experiment",
    )

    assert hasattr(result, "species_data")
    assert hasattr(result, "reactions_data")
    assert "gene_type" in result.species_data["source"].columns
    assert "confidence" in result.reactions_data["experiment"].columns


def test_sbml_compartmentalized_naming(test_data):
    """Test compartmentalized species naming convention."""
    interaction_edgelist, species_df, compartments_df, interaction_source = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist, species_df, compartments_df, interaction_source
    )

    comp_names = result.compartmentalized_species["sc_name"].tolist()
    assert "TP53 [nucleus]" in comp_names
    assert "MDM2 [cytoplasm]" in comp_names
    assert "CDKN1A [nucleus]" in comp_names


def test_sbml_custom_stoichiometry(test_data):
    """Test custom stoichiometry parameters."""
    interaction_edgelist, species_df, compartments_df, interaction_source = test_data

    result = sbml_dfs_core.sbml_dfs_from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        interaction_source,
        upstream_stoichiometry=2,
        downstream_stoichiometry=3,
    )

    stoichiometries = result.reaction_species["stoichiometry"].unique()
    assert 2 in stoichiometries  # upstream
    assert 3 in stoichiometries  # downstream
