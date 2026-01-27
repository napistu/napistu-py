from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fs.errors import ResourceNotFound

from napistu import sbml_dfs_utils
from napistu.constants import (
    BQB,
    BQB_DEFINING_ATTRS,
    BQB_DEFINING_ATTRS_LOOSE,
    CONSENSUS_CHECKS,
    IDENTIFIERS,
    MINI_SBO_FROM_NAME,
    NAPISTU_STANDARD_OUTPUTS,
    ONTOLOGIES,
    SBML_DFS,
    SBML_DFS_SCHEMA,
    SBOTERM_NAMES,
    SCHEMA_DEFS,
    SOURCE_SPEC,
    VALID_SBO_TERM_NAMES,
    VALID_SBO_TERMS,
)
from napistu.identifiers import Identifiers, _check_species_identifiers_table
from napistu.ingestion import sbml
from napistu.ingestion.constants import (
    COMPARTMENTS,
    INTERACTION_EDGELIST_DEFAULTS,
    INTERACTION_EDGELIST_DEFS,
)
from napistu.sbml_dfs_core import SBML_dfs
from napistu.source import Source


@pytest.fixture
def test_data():
    """Create test data for SBML integration tests."""

    blank_id = Identifiers([])

    # Test compartments
    compartments_df = pd.DataFrame(
        [
            {SBML_DFS.C_NAME: COMPARTMENTS.NUCLEUS, SBML_DFS.C_IDENTIFIERS: blank_id},
            {SBML_DFS.C_NAME: COMPARTMENTS.CYTOPLASM, SBML_DFS.C_IDENTIFIERS: blank_id},
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
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "TP53",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "CDKN1A",
                INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: COMPARTMENTS.NUCLEUS,
                INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: COMPARTMENTS.NUCLEUS,
                SBML_DFS.R_NAME: "TP53_activates_CDKN1A",
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.STIMULATOR,
                SBML_DFS.R_IDENTIFIERS: blank_id,
                "confidence": 0.95,
            },
            {
                INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: "MDM2",
                INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: "TP53",
                INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: COMPARTMENTS.CYTOPLASM,
                INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: COMPARTMENTS.NUCLEUS,
                SBML_DFS.R_NAME: "MDM2_inhibits_TP53",
                INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: SBOTERM_NAMES.INHIBITOR,
                SBML_DFS.R_IDENTIFIERS: blank_id,
                "confidence": 0.87,
            },
        ]
    )

    return [interaction_edgelist, species_df, compartments_df]


def test_sbml_dfs_from_dict_required(sbml_dfs, model_source_stub):
    val_dict = {k: getattr(sbml_dfs, k) for k in sbml_dfs._required_entities}
    sbml_dfs2 = SBML_dfs(val_dict, model_source_stub)
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


def test_removing_species_removes_data(sbml_dfs_w_data):
    """Test that removing a species removes the species data"""
    data = list(sbml_dfs_w_data.species_data.values())[0]
    s_id = [data.index[0]]
    sbml_dfs_w_data.remove_entities(SBML_DFS.SPECIES, s_id, remove_references=True)
    data_2 = list(sbml_dfs_w_data.species_data.values())[0]
    assert s_id[0] not in data_2.index
    sbml_dfs_w_data.validate()


def test_removing_reaction_removes_data(sbml_dfs_w_data):
    """Test that removing a reaction removes the reactions data"""
    data = list(sbml_dfs_w_data.reactions_data.values())[0]
    r_id = [data.index[0]]
    sbml_dfs_w_data.remove_entities(SBML_DFS.REACTIONS, r_id, remove_references=True)
    data_2 = list(sbml_dfs_w_data.reactions_data.values())[0]
    assert r_id[0] not in data_2.index
    sbml_dfs_w_data.validate()


def test_read_sbml_with_invalid_ids(model_source_stub):
    SBML_W_BAD_IDS = "R-HSA-166658.sbml"
    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    sbml_w_bad_ids_path = os.path.join(test_path, "test_data", SBML_W_BAD_IDS)
    assert os.path.isfile(sbml_w_bad_ids_path)

    # invalid identifiers still create a valid sbml_dfs
    sbml_w_bad_ids = sbml.SBML(sbml_w_bad_ids_path)
    assert isinstance(
        SBML_dfs(sbml_w_bad_ids, model_source_stub),
        SBML_dfs,
    )


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


def test_entity_removal_consistency(sbml_dfs):
    """Test that entity removal produces consistent results regardless of starting point."""

    # 1. Choose the first compartment in the sbml_dfs fixture
    print(sbml_dfs.compartments)
    c_id = "compartment_984"  # cytosol

    # 2. Call find_entity_references to find all affected entities
    baseline_removals = sbml_dfs.find_entity_references(SBML_DFS.COMPARTMENTS, [c_id])

    # Store the baseline results for comparison
    baseline_total = sum(len(entities) for entities in baseline_removals.values())
    print(f"Baseline total entities to be removed: {baseline_total}")
    print(
        f"Baseline removals by type: {[(k, len(v)) for k, v in baseline_removals.items() if v]}"
    )

    # Collect affected entities for each table type
    starting_points = {}
    for table_type, affected_ids in baseline_removals.items():
        if affected_ids:
            starting_points[table_type] = list(affected_ids)

    # First, compare find_entity_references results across all starting points
    print("\n=== Comparing find_entity_references results ===")
    reference_results = {}

    for start_table, start_ids in starting_points.items():
        if not start_ids:
            continue

        # Get fresh copy and find references
        test_sbml_dfs = sbml_dfs.copy()
        predicted_removals = test_sbml_dfs.find_entity_references(
            start_table, start_ids
        )

        total_predicted = sum(len(entities) for entities in predicted_removals.values())
        reference_results[start_table] = {
            "predicted": predicted_removals,
            "total": total_predicted,
        }

        print(f"{start_table}: {total_predicted} total entities predicted")
        print(f"  Details: {[(k, len(v)) for k, v in predicted_removals.items() if v]}")

    # Check if all find_entity_references results are identical
    reference_totals = [result["total"] for result in reference_results.values()]
    if len(set(reference_totals)) == 1:
        print(
            "✓ All find_entity_references results are identical - only testing actual removal once"
        )

        # Only test actual removal from the baseline starting point
        test_sbml_dfs = sbml_dfs.copy()
        test_sbml_dfs.remove_entities(
            SBML_DFS.COMPARTMENTS, [c_id], remove_references=True
        )
        test_sbml_dfs.validate()
        print("✓ Actual removal and validation successful")

    else:
        print("✗ find_entity_references results are inconsistent!")
        for start_table, result in reference_results.items():
            print(f"  {start_table}: {result['total']} entities")

        # This indicates a bug in the find_entity_references logic
        assert (
            False
        ), f"find_entity_references produces inconsistent results: {reference_totals}"


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


def test_get_identifiers_handles_missing_values(model_source_stub):

    # Minimal DataFrame with all types
    df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A", "B", "C", "D"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers([]),
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
    sbml = SBML_dfs(sbml_dict, model_source_stub, validate=False)
    result = sbml.get_identifiers(SBML_DFS.SPECIES)
    assert result.shape[0] == 0 or all(
        result[SBML_DFS.S_ID] == "s1"
    ), "Only Identifiers objects should be returned."


def test_get_identifiers_keep_source(sbml_dfs):
    """Test that get_identifiers excludes source column by default and includes it when keep_source=True."""
    # Test default behavior (keep_source=False) - source should be excluded
    result_default = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)
    assert SBML_DFS.S_SOURCE not in result_default.columns

    # Test with keep_source=True - source should be included
    result_with_source = sbml_dfs.get_identifiers(SBML_DFS.SPECIES, keep_source=True)
    assert SBML_DFS.S_SOURCE in result_with_source.columns


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


def test_remove_entity_data_nonexistent(sbml_dfs_w_data):
    """Test ValueError when trying to remove nonexistent entity data."""
    # Try to remove nonexistent species data
    with pytest.raises(
        ValueError, match="Label 'nonexistent_label' not found in species_data"
    ):
        sbml_dfs_w_data._remove_entity_data(SBML_DFS.SPECIES, "nonexistent_label")

    # Verify the data is unchanged
    assert set(sbml_dfs_w_data.species_data.keys()) == {"test_species"}

    # Try to remove nonexistent reactions data
    with pytest.raises(
        ValueError, match="Label 'nonexistent_label' not found in reactions_data"
    ):
        sbml_dfs_w_data._remove_entity_data(SBML_DFS.REACTIONS, "nonexistent_label")

    # Verify the data is unchanged
    assert set(sbml_dfs_w_data.reactions_data.keys()) == {"test_reactions"}

    # Validate the model is still valid
    sbml_dfs_w_data.validate()


def test_get_characteristic_species_ids(sbml_dfs_characteristic_test_data):
    """
    Test get_characteristic_species_ids function with both dogmatic and non-dogmatic cases.
    """
    sbml_dfs = sbml_dfs_characteristic_test_data
    mock_species_ids = sbml_dfs._mock_species_ids

    # Test dogmatic case (default)
    expected_bqbs = BQB_DEFINING_ATTRS + [BQB.HAS_PART]  # noqa: F841
    with patch.object(sbml_dfs, "get_identifiers", return_value=mock_species_ids):
        dogmatic_result = sbml_dfs.get_characteristic_species_ids()
        expected_dogmatic = mock_species_ids.query(
            f"{IDENTIFIERS.BQB} in @expected_bqbs"
        )
        pd.testing.assert_frame_equal(
            dogmatic_result, expected_dogmatic, check_like=True
        )

    # Test non-dogmatic case
    expected_bqbs = BQB_DEFINING_ATTRS_LOOSE + [BQB.HAS_PART]  # noqa: F841
    with patch.object(sbml_dfs, "get_identifiers", return_value=mock_species_ids):
        non_dogmatic_result = sbml_dfs.get_characteristic_species_ids(dogmatic=False)
        expected_non_dogmatic = mock_species_ids.query(
            f"{IDENTIFIERS.BQB} in @expected_bqbs"
        )
        pd.testing.assert_frame_equal(
            non_dogmatic_result, expected_non_dogmatic, check_like=True
        )


def test_sbml_basic_functionality(test_data, model_source_stub):
    """Test basic SBML_dfs creation from edgelist."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = SBML_dfs.from_edgelist(
        interaction_edgelist, species_df, compartments_df, model_source_stub
    )

    assert isinstance(result, SBML_dfs)
    assert len(result.species) == 3
    assert len(result.compartments) == 2
    assert len(result.reactions) == 2
    assert (
        len(result.compartmentalized_species) == 3
    )  # TP53[nucleus], CDKN1A[nucleus], MDM2[cytoplasm]
    assert len(result.reaction_species) == 4  # 2 reactions * 2 species each


def test_sbml_extra_data_preservation(test_data, model_source_stub):
    """Test that extra columns are preserved when requested."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = SBML_dfs.from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        model_source_stub,
        keep_species_data=True,
        keep_reactions_data="experiment",
    )

    assert hasattr(result, SBML_DFS.SPECIES_DATA)
    assert hasattr(result, SBML_DFS.REACTIONS_DATA)
    assert "gene_type" in result.species_data["source"].columns
    assert "confidence" in result.reactions_data["experiment"].columns


def test_sbml_compartmentalized_naming(test_data, model_source_stub):
    """Test compartmentalized species naming convention."""
    interaction_edgelist, species_df, compartments_df = test_data

    result = SBML_dfs.from_edgelist(
        interaction_edgelist, species_df, compartments_df, model_source_stub
    )

    comp_names = result.compartmentalized_species[SBML_DFS.SC_NAME].tolist()
    assert "TP53 [nucleus]" in comp_names
    assert "MDM2 [cytoplasm]" in comp_names
    assert "CDKN1A [nucleus]" in comp_names


def test_sbml_custom_defaults(test_data, model_source_stub):
    """Test custom stoichiometry parameters."""
    interaction_edgelist, species_df, compartments_df = test_data

    custom_defaults = INTERACTION_EDGELIST_DEFAULTS.copy()
    custom_defaults[INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM] = -2
    custom_defaults[INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM] = 3
    custom_defaults[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM] = (
        SBOTERM_NAMES.REACTANT
    )
    custom_defaults[INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM] = (
        SBOTERM_NAMES.PRODUCT
    )

    result = SBML_dfs.from_edgelist(
        interaction_edgelist,
        species_df,
        compartments_df,
        model_source_stub,
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
                Identifiers([]),
                Identifiers([]),
            ],
            SBML_DFS.S_SOURCE: [Source.empty(), Source.empty()],
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


@pytest.mark.skip_on_windows
def test_to_pickle_and_from_pickle(sbml_dfs):
    """Test saving and loading an SBML_dfs via pickle."""

    # Save to pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        pickle_path = tmp_file.name

    try:
        sbml_dfs.to_pickle(pickle_path)

        # Load from pickle
        loaded_sbml_dfs = SBML_dfs.from_pickle(pickle_path)

        # Verify the loaded SBML_dfs is identical
        assert isinstance(loaded_sbml_dfs, SBML_dfs)
        assert len(loaded_sbml_dfs.compartments) == len(sbml_dfs.compartments)
        assert len(loaded_sbml_dfs.species) == len(sbml_dfs.species)
        assert len(loaded_sbml_dfs.reactions) == len(sbml_dfs.reactions)
        assert len(loaded_sbml_dfs.reaction_species) == len(sbml_dfs.reaction_species)

        # Compare each table, excluding identifier and source columns that contain custom objects
        for table_name in SBML_DFS_SCHEMA.REQUIRED_ENTITIES:
            original_df = getattr(sbml_dfs, table_name)
            loaded_df = getattr(loaded_sbml_dfs, table_name)

            # Get the schema for this table
            table_schema = SBML_DFS_SCHEMA.SCHEMA[table_name]

            # Create copies to avoid modifying the original DataFrames
            original_copy = original_df.copy()
            loaded_copy = loaded_df.copy()

            # Drop identifier and source columns if they exist
            if SCHEMA_DEFS.ID in table_schema:
                id_col = table_schema[SCHEMA_DEFS.ID]
                if id_col in original_copy.columns:
                    original_copy = original_copy.drop(columns=[id_col])
                if id_col in loaded_copy.columns:
                    loaded_copy = loaded_copy.drop(columns=[id_col])

            if SCHEMA_DEFS.SOURCE in table_schema:
                source_col = table_schema[SCHEMA_DEFS.SOURCE]
                if source_col in original_copy.columns:
                    original_copy = original_copy.drop(columns=[source_col])
                if source_col in loaded_copy.columns:
                    loaded_copy = loaded_copy.drop(columns=[source_col])

            # Compare the DataFrames without custom object columns
            pd.testing.assert_frame_equal(original_copy, loaded_copy)

    finally:
        # Clean up
        if os.path.exists(pickle_path):
            os.unlink(pickle_path)


@pytest.mark.skip_on_windows
def test_from_pickle_nonexistent_file():
    """Test that from_pickle raises appropriate error for nonexistent file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_path = os.path.join(temp_dir, "nonexistent_file.pkl")
        with pytest.raises(ResourceNotFound):
            SBML_dfs.from_pickle(nonexistent_path)


@pytest.mark.skip_on_windows
def test_pickle_with_species_data(sbml_dfs):
    """Test pickle functionality with species_data."""
    # Use the existing sbml_dfs fixture and add species_data

    # Get actual species IDs from the fixture
    species_ids = sbml_dfs.species.index.tolist()[:2]  # Use first 2 species

    # Add species data
    species_data = pd.DataFrame(
        {"expression": [10.5, 20.3], "confidence": [0.8, 0.9]}, index=species_ids
    )
    species_data.index.name = SBML_DFS.S_ID
    sbml_dfs.add_species_data("test_data", species_data)

    # Save to pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        pickle_path = tmp_file.name

    try:
        sbml_dfs.to_pickle(pickle_path)

        # Load from pickle
        loaded_sbml_dfs = SBML_dfs.from_pickle(pickle_path)

        # Verify species_data is preserved
        assert "test_data" in loaded_sbml_dfs.species_data
        pd.testing.assert_frame_equal(
            loaded_sbml_dfs.species_data["test_data"],
            sbml_dfs.species_data["test_data"],
        )

    finally:
        # Clean up
        if os.path.exists(pickle_path):
            os.unlink(pickle_path)


def test_get_sources(sbml_dfs_metabolism):
    """Test get_sources method returns unnest sources table."""
    # Test with species
    sources_df = sbml_dfs_metabolism.get_sources(SBML_DFS.SPECIES)
    assert sources_df is not None
    assert SOURCE_SPEC.PATHWAY_ID in sources_df.columns

    # Test with reactions
    sources_df = sbml_dfs_metabolism.get_sources(SBML_DFS.REACTIONS)
    assert sources_df is not None
    assert SOURCE_SPEC.PATHWAY_ID in sources_df.columns

    # Test invalid entity type raises error
    with pytest.raises(
        ValueError, match="reaction_species does not have a source attribute"
    ):
        sbml_dfs_metabolism.get_sources(SBML_DFS.REACTION_SPECIES)


def test_get_source_occurrence(sbml_dfs_metabolism):
    """Test get_source_occurrence method returns source occurrence summary."""
    # Test with species
    occurrence_df = sbml_dfs_metabolism.get_source_occurrence(SBML_DFS.SPECIES)
    assert isinstance(occurrence_df, pd.DataFrame)
    assert occurrence_df.shape == (129, 4)  # Expected: 129 pathways, 4 columns

    # Test binarize functionality with species
    binarized_df = sbml_dfs_metabolism.get_source_occurrence(
        SBML_DFS.SPECIES, binarize=True
    )
    assert binarized_df.shape == occurrence_df.shape
    assert binarized_df.isin([0, 1]).all().all()
    pd.testing.assert_frame_equal(binarized_df, (occurrence_df > 0).astype(int))

    # Test with reactions
    occurrence_df = sbml_dfs_metabolism.get_source_occurrence(SBML_DFS.REACTIONS)
    assert isinstance(occurrence_df, pd.DataFrame)
    assert occurrence_df.shape == (86, 4)  # Expected: 86 pathways, 4 columns


def test_get_source_cooccurrence(sbml_dfs_metabolism):
    """Test get_source_cooccurrence method returns co-occurrence matrix."""
    # Test with species
    cooccurrence_df = sbml_dfs_metabolism.get_source_cooccurrence(SBML_DFS.SPECIES)
    assert isinstance(cooccurrence_df, pd.DataFrame)
    assert cooccurrence_df.shape == (4, 4)  # Expected: 4x4 square matrix

    # Test with reactions
    cooccurrence_df = sbml_dfs_metabolism.get_source_cooccurrence(SBML_DFS.REACTIONS)
    assert isinstance(cooccurrence_df, pd.DataFrame)
    assert cooccurrence_df.shape == (4, 4)  # Expected: 4x4 square matrix


def test_pathway_methods_single_source_error(sbml_dfs):
    """Test that source methods raise error for single-source models."""
    # Test get_source_occurrence raises error for single-source model
    with pytest.raises(ValueError, match="The Source tables for species were empty"):
        sbml_dfs.get_source_occurrence(SBML_DFS.SPECIES)

    with pytest.raises(ValueError, match="The Source tables for reactions were empty"):
        sbml_dfs.get_source_occurrence(SBML_DFS.REACTIONS)

    # Test get_pathway_cooccurrence raises error for single-source model
    with pytest.raises(ValueError, match="The Source tables for species were empty"):
        sbml_dfs.get_source_cooccurrence(SBML_DFS.SPECIES)

    with pytest.raises(ValueError, match="The Source tables for reactions were empty"):
        sbml_dfs.get_source_cooccurrence(SBML_DFS.REACTIONS)


def test_priority_pathways_filtering(sbml_dfs_metabolism):
    """Test that priority_pathways parameter correctly filters sources."""
    # First, get the full co-occurrence matrix to see pathway names
    full_cooccurrence = sbml_dfs_metabolism.get_source_cooccurrence(SBML_DFS.SPECIES)

    # Select first 2 pathways for testing
    priority_pathways = list(full_cooccurrence.index)[:2]

    # Test with filtered pathways
    filtered_cooccurrence = sbml_dfs_metabolism.get_source_cooccurrence(
        SBML_DFS.SPECIES, priority_pathways=priority_pathways
    )

    # Verify the filtered result is (2,2)
    assert filtered_cooccurrence.shape == (2, 2)
    assert list(filtered_cooccurrence.index) == priority_pathways
    assert list(filtered_cooccurrence.columns) == priority_pathways


def test_get_ontology_occurrence(sbml_dfs_metabolism):
    """Test get_ontology_occurrence method returns ontology occurrence summary."""
    # Test with species
    occurrence_df = sbml_dfs_metabolism.get_ontology_occurrence(SBML_DFS.SPECIES)
    assert isinstance(occurrence_df, pd.DataFrame)
    assert occurrence_df.shape == (129, 7)  # Expected: 129 entities, 7 ontologies

    # Test binarize functionality with species
    binarized_df = sbml_dfs_metabolism.get_ontology_occurrence(
        SBML_DFS.SPECIES, binarize=True
    )
    assert binarized_df.shape == occurrence_df.shape
    assert binarized_df.isin([0, 1]).all().all()
    pd.testing.assert_frame_equal(binarized_df, (occurrence_df > 0).astype(int))

    # Test with reactions
    occurrence_df = sbml_dfs_metabolism.get_ontology_occurrence(SBML_DFS.REACTIONS)
    assert isinstance(occurrence_df, pd.DataFrame)
    assert occurrence_df.shape == (86, 4)  # Expected: 86 entities, 4 ontologies


def test_get_ontology_cooccurrence(sbml_dfs_metabolism):
    """Test get_ontology_cooccurrence method returns co-occurrence matrix."""
    # Test with species
    cooccurrence_df = sbml_dfs_metabolism.get_ontology_cooccurrence(SBML_DFS.SPECIES)
    assert isinstance(cooccurrence_df, pd.DataFrame)
    assert cooccurrence_df.shape == (7, 7)  # Expected: 7x7 square matrix

    # Test with reactions
    cooccurrence_df = sbml_dfs_metabolism.get_ontology_cooccurrence(SBML_DFS.REACTIONS)
    assert isinstance(cooccurrence_df, pd.DataFrame)
    assert cooccurrence_df.shape == (4, 4)  # Expected: 4x4 square matrix


def test_get_ontology_cooccurrence_multindex(sbml_dfs_metabolism):
    """Test get_ontology_cooccurrence method with multi-index columns."""
    # Test with species and allow_col_multindex=True
    cooccurrence_df = sbml_dfs_metabolism.get_ontology_cooccurrence(
        SBML_DFS.SPECIES, allow_col_multindex=True
    )
    assert isinstance(cooccurrence_df, pd.DataFrame)

    # Verify it's a square matrix
    assert cooccurrence_df.shape[0] == cooccurrence_df.shape[1]
    assert cooccurrence_df.shape == (7, 7)  # Expected: 7x7 square matrix

    # Verify multi-index is present
    assert isinstance(cooccurrence_df.columns, pd.MultiIndex)
    assert (
        len(cooccurrence_df.columns.names) == 2
    )  # Should have 2 levels (ontology, bqb)
    assert cooccurrence_df.columns.names == [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]

    # Verify expected ontologies are present
    expected_ontologies = {
        ONTOLOGIES.CHEBI,
        ONTOLOGIES.PUBMED,
        ONTOLOGIES.REACTOME,
        ONTOLOGIES.UNIPROT,
    }
    actual_ontologies = set(
        cooccurrence_df.columns.get_level_values(IDENTIFIERS.ONTOLOGY)
    )
    assert expected_ontologies.issubset(actual_ontologies)

    # Test with reactions and allow_col_multindex=True
    cooccurrence_df = sbml_dfs_metabolism.get_ontology_cooccurrence(
        SBML_DFS.REACTIONS, allow_col_multindex=True
    )
    assert isinstance(cooccurrence_df, pd.DataFrame)

    # Verify it's a square matrix
    assert cooccurrence_df.shape[0] == cooccurrence_df.shape[1]
    assert cooccurrence_df.shape == (4, 4)  # Expected: 4x4 square matrix

    # Verify multi-index is present
    assert isinstance(cooccurrence_df.columns, pd.MultiIndex)
    assert (
        len(cooccurrence_df.columns.names) == 2
    )  # Should have 2 levels (ontology, bqb)
    assert cooccurrence_df.columns.names == [IDENTIFIERS.ONTOLOGY, IDENTIFIERS.BQB]

    # Verify expected ontologies are present
    expected_ontologies = {
        ONTOLOGIES.EC_CODE,
        ONTOLOGIES.GO,
        ONTOLOGIES.PUBMED,
        ONTOLOGIES.REACTOME,
    }
    actual_ontologies = set(
        cooccurrence_df.columns.get_level_values(IDENTIFIERS.ONTOLOGY)
    )
    assert expected_ontologies.issubset(actual_ontologies)


def test_get_ontology_occurrence_characteristic_only(sbml_dfs_characteristic_test_data):
    """Test get_ontology_occurrence method with characteristic_only parameter using the new fixture."""
    sbml_dfs = sbml_dfs_characteristic_test_data

    # Test with species and characteristic_only=True
    occurrence_df_char = sbml_dfs.get_ontology_occurrence(
        SBML_DFS.SPECIES, characteristic_only=True
    )

    assert isinstance(occurrence_df_char, pd.DataFrame)
    assert occurrence_df_char.shape[0] == 1
    assert occurrence_df_char.shape[1] == 3

    # Test with species and characteristic_only=False
    occurrence_df_all = sbml_dfs.get_ontology_occurrence(
        SBML_DFS.SPECIES, characteristic_only=False
    )

    assert isinstance(occurrence_df_all, pd.DataFrame)
    assert occurrence_df_all.shape[0] == 1
    assert occurrence_df_all.shape[1] == 5


def test_get_ontology_x_source_cooccurrence(sbml_dfs_metabolism):
    """
    Test get_ontology_x_source_cooccurrence method with metabolism data.

    This test verifies that the method correctly creates a co-occurrence matrix
    between ontologies and sources using the metabolism fixture.
    """
    # Test basic functionality
    cooccurrence_matrix = sbml_dfs_metabolism.get_ontology_x_source_cooccurrence(
        SBML_DFS.SPECIES
    )

    # Verify the result is a DataFrame with correct data types
    assert isinstance(cooccurrence_matrix, pd.DataFrame)
    assert (cooccurrence_matrix >= 0).all().all(), "All values should be non-negative"
    assert cooccurrence_matrix.dtypes.apply(
        lambda x: pd.api.types.is_integer_dtype(x)
    ).all(), "All values should be integers"

    # Test with characteristic_only=True
    char_cooccurrence = sbml_dfs_metabolism.get_ontology_x_source_cooccurrence(
        SBML_DFS.SPECIES, characteristic_only=True
    )
    assert isinstance(char_cooccurrence, pd.DataFrame)

    # Test with custom priority pathways (use actual pathway names from the data)
    custom_pathways = cooccurrence_matrix.columns[
        :2
    ].tolist()  # Use first 2 pathways from the data
    custom_cooccurrence = sbml_dfs_metabolism.get_ontology_x_source_cooccurrence(
        SBML_DFS.SPECIES, priority_pathways=custom_pathways
    )
    assert isinstance(custom_cooccurrence, pd.DataFrame)

    # Test with reactions
    reaction_cooccurrence = sbml_dfs_metabolism.get_ontology_x_source_cooccurrence(
        SBML_DFS.REACTIONS
    )
    assert isinstance(reaction_cooccurrence, pd.DataFrame)

    # Test exact dimensions
    assert cooccurrence_matrix.shape == (
        7,
        4,
    ), f"Expected species co-occurrence shape (7, 4), got {cooccurrence_matrix.shape}"
    assert reaction_cooccurrence.shape == (
        4,
        4,
    ), f"Expected reaction co-occurrence shape (4, 4), got {reaction_cooccurrence.shape}"


def test_post_consensus_checks(sbml_dfs_metabolism):
    """Test the post_consensus_checks method with a consensus model."""

    # Test with default parameters
    results = sbml_dfs_metabolism.post_consensus_checks()

    # Verify the structure of the results
    assert isinstance(results, dict), "Results should be a dictionary"

    # Check that we have results for the default entity types
    expected_entity_types = [SBML_DFS.SPECIES, SBML_DFS.COMPARTMENTS]
    for entity_type in expected_entity_types:
        assert entity_type in results, f"Results should contain {entity_type}"
        assert isinstance(
            results[entity_type], dict
        ), f"Results for {entity_type} should be a dictionary"

    # Check that we have results for the default check types
    expected_check_types = [
        CONSENSUS_CHECKS.SOURCE_COOCCURRENCE,
        CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE,
    ]
    for entity_type in expected_entity_types:
        for check_type in expected_check_types:
            assert (
                check_type in results[entity_type]
            ), f"Results should contain {check_type} for {entity_type}"
            assert isinstance(
                results[entity_type][check_type], pd.DataFrame
            ), f"Result for {entity_type}/{check_type} should be a DataFrame"

    # Test with custom entity types
    custom_results = sbml_dfs_metabolism.post_consensus_checks(
        entity_types=[SBML_DFS.SPECIES],
        check_types=[CONSENSUS_CHECKS.SOURCE_COOCCURRENCE],
    )

    assert SBML_DFS.SPECIES in custom_results, "Custom results should contain species"
    assert (
        SBML_DFS.COMPARTMENTS not in custom_results
    ), "Custom results should not contain compartments when not requested"
    assert (
        CONSENSUS_CHECKS.SOURCE_COOCCURRENCE in custom_results[SBML_DFS.SPECIES]
    ), "Custom results should contain source_cooccurrence"
    assert (
        CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE
        not in custom_results[SBML_DFS.SPECIES]
    ), "Custom results should not contain ontology_x_source_cooccurrence when not requested"

    # Test with invalid check types
    with pytest.raises(ValueError, match="Invalid check types"):
        sbml_dfs_metabolism.post_consensus_checks(check_types=["invalid_check_type"])

    # Test exact dimensions based on the output
    species_source_cooccurrence = results[SBML_DFS.SPECIES][
        CONSENSUS_CHECKS.SOURCE_COOCCURRENCE
    ]
    species_ontology_cooccurrence = results[SBML_DFS.SPECIES][
        CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE
    ]
    compartments_source_cooccurrence = results[SBML_DFS.COMPARTMENTS][
        CONSENSUS_CHECKS.SOURCE_COOCCURRENCE
    ]
    compartments_ontology_cooccurrence = results[SBML_DFS.COMPARTMENTS][
        CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE
    ]

    # Test exact dimensions
    assert species_source_cooccurrence.shape == (
        4,
        4,
    ), f"Expected species source co-occurrence shape (4, 4), got {species_source_cooccurrence.shape}"
    assert species_ontology_cooccurrence.shape == (
        7,
        4,
    ), f"Expected species ontology co-occurrence shape (7, 4), got {species_ontology_cooccurrence.shape}"
    assert compartments_source_cooccurrence.shape == (
        4,
        4,
    ), f"Expected compartments source co-occurrence shape (4, 4), got {compartments_source_cooccurrence.shape}"
    assert compartments_ontology_cooccurrence.shape == (
        1,
        4,
    ), f"Expected compartments ontology co-occurrence shape (1, 4), got {compartments_ontology_cooccurrence.shape}"

    # Test that all results are numeric and not empty
    for entity_type in [SBML_DFS.SPECIES, SBML_DFS.COMPARTMENTS]:
        for check_type in [
            CONSENSUS_CHECKS.SOURCE_COOCCURRENCE,
            CONSENSUS_CHECKS.ONTOLOGY_X_SOURCE_COOCCURRENCE,
        ]:
            df = results[entity_type][check_type]
            assert (
                not df.empty
            ), f"DataFrame for {entity_type}/{check_type} should not be empty"
            assert pd.api.types.is_numeric_dtype(
                df.values
            ), f"Values for {entity_type}/{check_type} should be numeric"


def test_get_sbo_term_occurrence(sbml_dfs_metabolism):
    """Test get_sbo_term_occurrence method returns SBO term occurrence summary."""
    # Test basic functionality
    occurrence_df = sbml_dfs_metabolism.get_sbo_term_occurrence()

    # Print dimensions for inspection
    print(f"SBO term occurrence dimensions: {occurrence_df.shape}")
    print(f"Columns: {list(occurrence_df.columns)}")

    assert isinstance(occurrence_df, pd.DataFrame), "Should return a DataFrame"
    assert occurrence_df.shape == (
        86,
        5,
    ), f"Expected (86, 5), got {occurrence_df.shape}"

    # Test that columns are valid SBO term names (default name_terms=True)
    column_names = occurrence_df.columns.tolist()
    assert all(
        isinstance(col, str) for col in column_names
    ), "Column names should be strings when name_terms=True"
    assert all(
        col in VALID_SBO_TERM_NAMES for col in column_names
    ), f"All column names should be valid SBO term names. Got: {column_names}"

    # Test with name_terms=False
    occurrence_df_numeric = sbml_dfs_metabolism.get_sbo_term_occurrence(
        name_terms=False
    )
    assert isinstance(occurrence_df_numeric, pd.DataFrame), "Should return a DataFrame"
    assert occurrence_df_numeric.shape == (
        86,
        5,
    ), f"Expected (86, 5), got {occurrence_df_numeric.shape}"

    # Test that columns are valid SBO terms (numeric codes) when name_terms=False
    column_terms = occurrence_df_numeric.columns.tolist()
    assert all(
        col in VALID_SBO_TERMS for col in column_terms
    ), f"All column terms should be valid SBO terms. Got: {column_terms}"

    # Values should be non-negative integers (counts)
    assert (occurrence_df >= 0).all().all(), "All values should be non-negative"
    assert occurrence_df.dtypes.apply(
        lambda x: x.kind in "iu"
    ).all(), "All values should be integers"


def test_get_sbo_term_x_source_cooccurrence(sbml_dfs_metabolism):
    """Test get_sbo_term_x_source_cooccurrence method returns co-occurrence matrix."""
    # Test basic functionality
    cooccurrence_matrix = sbml_dfs_metabolism.get_sbo_term_x_source_cooccurrence()

    # Verify the result is a DataFrame with correct data types
    assert isinstance(cooccurrence_matrix, pd.DataFrame), "Should return a DataFrame"
    assert cooccurrence_matrix.shape == (
        5,
        4,
    ), f"Expected (5, 4), got {cooccurrence_matrix.shape}"
    assert (cooccurrence_matrix >= 0).all().all(), "All values should be non-negative"
    assert cooccurrence_matrix.dtypes.apply(
        lambda x: pd.api.types.is_integer_dtype(x)
    ).all(), "All values should be integers"

    # Test that row names are valid SBO term names (default name_terms=True)
    row_names = cooccurrence_matrix.index.tolist()
    assert all(
        name in VALID_SBO_TERM_NAMES for name in row_names
    ), f"All row names should be valid SBO term names. Got: {row_names}"

    # Test with name_terms=False
    numeric_cooccurrence = sbml_dfs_metabolism.get_sbo_term_x_source_cooccurrence(
        name_terms=False
    )
    assert isinstance(numeric_cooccurrence, pd.DataFrame), "Should return a DataFrame"
    assert numeric_cooccurrence.shape == (
        5,
        4,
    ), f"Expected (5, 4), got {numeric_cooccurrence.shape}"

    # Test that row names are valid SBO terms (numeric codes) when name_terms=False
    row_terms = numeric_cooccurrence.index.tolist()
    assert all(
        term in VALID_SBO_TERMS for term in row_terms
    ), f"All row terms should be valid SBO terms. Got: {row_terms}"


def test_force_edgelist_consistency(model_source_stub):
    """Test that force_edgelist_consistency filters out invalid species references."""

    # Create species table with only some of the species that will be referenced
    species_df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["protein_A", "protein_B", "metabolite_X"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P12345", "bqb": "is"}]
                ),
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P67890", "bqb": "is"}]
                ),
                Identifiers(
                    [{"ontology": "chebi", "identifier": "CHEBI:123", "bqb": "is"}]
                ),
            ],
        }
    )

    # Create compartments
    compartments_df = sbml_dfs_utils.stub_compartments()

    # Create interaction edgelist with INVALID species references
    interaction_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: [
                "protein_A",
                "protein_B",
                "missing_protein_C",
                "protein_A",
            ],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: [
                "protein_B",
                "missing_protein_D",
                "metabolite_X",
                "missing_protein_D",
            ],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: [
                COMPARTMENTS.CELLULAR_COMPONENT
            ]
            * 4,
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: [
                COMPARTMENTS.CELLULAR_COMPONENT
            ]
            * 4,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: [SBOTERM_NAMES.STIMULATOR]
            * 4,
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: [SBOTERM_NAMES.MODIFIED]
            * 4,
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: [0, 0, 0, 0],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: [0, 0, 0, 0],
            SBML_DFS.R_ISREVERSIBLE: [False] * 4,
            SBML_DFS.R_NAME: ["rxn1", "rxn2", "rxn3", "rxn4"],
            SBML_DFS.R_IDENTIFIERS: [Identifiers([]) for _ in range(4)],
        }
    )

    # Test WITHOUT force_edgelist_consistency - should fail validation
    with pytest.raises(ValueError, match="Invalid references"):
        sbml_dfs = SBML_dfs.from_edgelist(
            interaction_edgelist=interaction_edgelist,
            species_df=species_df,
            compartments_df=compartments_df,
            model_source=model_source_stub,
            force_edgelist_consistency=False,
        )

    # Test WITH force_edgelist_consistency - should succeed with warnings
    with patch("napistu.sbml_dfs_utils.logger") as mock_logger:

        sbml_dfs = SBML_dfs.from_edgelist(
            interaction_edgelist=interaction_edgelist,
            species_df=species_df,
            compartments_df=compartments_df,
            model_source=model_source_stub,
            force_edgelist_consistency=True,
        )

        # Should have logged warnings about missing species and filtering
        assert mock_logger.warning.call_count > 0

        # Check that warnings about missing species were logged
        warning_messages = [call[0][0] for call in mock_logger.warning.call_args_list]
        assert any(
            "missing_protein_C" in msg and "missing_protein_D" in msg
            for msg in warning_messages
        )

        # Check that filtering warning was logged
        assert any(
            "Filtered" in msg and "interactions" in msg for msg in warning_messages
        )

    # Verify the resulting SBML_dfs only contains valid interactions
    assert sbml_dfs.reactions.shape[0] == 1  # Only rxn1 is valid should remain
    assert (
        sbml_dfs.species.shape[0] == 2
    )  # metabolite X is droppeed since its only linked to missing species
    assert sbml_dfs.reaction_species.shape[0] == 2  # 2 reaction species for 1 reaction


def test_force_edgelist_consistency_with_valid_data(model_source_stub):
    """Test that force_edgelist_consistency doesn't modify valid data."""

    # Create completely valid data
    species_df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["protein_A", "protein_B"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P12345", "bqb": "is"}]
                ),
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P67890", "bqb": "is"}]
                ),
            ],
        }
    )

    compartments_df = sbml_dfs_utils.stub_compartments()

    interaction_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["protein_A"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["protein_B"],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: ["cellular_component"],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: ["cellular_component"],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: ["stimulator"],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: ["modified"],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: [0],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: [0],
            SBML_DFS.R_ISREVERSIBLE: [False],
            SBML_DFS.R_NAME: ["rxn1"],
            SBML_DFS.R_IDENTIFIERS: [Identifiers([])],
        }
    )

    # Should work with or without force_edgelist_consistency
    for force_consistency in [True, False]:
        sbml_dfs = SBML_dfs.from_edgelist(
            interaction_edgelist=interaction_edgelist,
            species_df=species_df,
            compartments_df=compartments_df,
            model_source=model_source_stub,
            force_edgelist_consistency=force_consistency,
        )

        assert sbml_dfs.reactions.shape[0] == 1
        assert sbml_dfs.species.shape[0] == 2


def test_force_edgelist_consistency_invalid_compartments(model_source_stub):
    """Test that invalid compartments still raise errors even with force_edgelist_consistency=True."""

    species_df = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["protein_A", "protein_B"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P12345", "bqb": "is"}]
                ),
                Identifiers(
                    [{"ontology": "uniprot", "identifier": "P67890", "bqb": "is"}]
                ),
            ],
        }
    )

    compartments_df = (
        sbml_dfs_utils.stub_compartments()
    )  # Only has "cellular_component"

    # Create edgelist with invalid compartment reference
    interaction_edgelist = pd.DataFrame(
        {
            INTERACTION_EDGELIST_DEFS.NAME_UPSTREAM: ["protein_A"],
            INTERACTION_EDGELIST_DEFS.NAME_DOWNSTREAM: ["protein_B"],
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_UPSTREAM: [
                "invalid_compartment"
            ],  # Invalid!
            INTERACTION_EDGELIST_DEFS.COMPARTMENT_DOWNSTREAM: ["cellular_component"],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_UPSTREAM: ["stimulator"],
            INTERACTION_EDGELIST_DEFS.SBO_TERM_NAME_DOWNSTREAM: ["modified"],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_UPSTREAM: [0],
            INTERACTION_EDGELIST_DEFS.STOICHIOMETRY_DOWNSTREAM: [0],
            SBML_DFS.R_ISREVERSIBLE: [False],
            SBML_DFS.R_NAME: ["rxn1"],
            SBML_DFS.R_IDENTIFIERS: [Identifiers([])],
        }
    )

    # Should still raise error for invalid compartments even with force_edgelist_consistency=True
    with pytest.raises(ValueError, match="Missing compartments"):
        _ = SBML_dfs.from_edgelist(
            interaction_edgelist=interaction_edgelist,
            species_df=species_df,
            compartments_df=compartments_df,
            model_source=model_source_stub,
            force_edgelist_consistency=True,
        )


def test_export_sbml_dfs(sbml_dfs):
    """Test that export_sbml_dfs creates all expected files with correct formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_prefix = "test_"
        sbml_dfs.export_sbml_dfs(
            model_prefix=model_prefix,
            outdir=tmpdir,
            overwrite=True,
            dogmatic=True,
        )

        # Expected files
        expected_files = {
            model_prefix + NAPISTU_STANDARD_OUTPUTS.SPECIES_IDENTIFIERS: ".tsv",
            model_prefix
            + NAPISTU_STANDARD_OUTPUTS.REACTIONS_SOURCE_TOTAL_COUNTS: ".tsv",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.SID_TO_SCIDS: ".tsv",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.SPECIES: ".json",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTIONS: ".json",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.REACTION_SPECIES: ".json",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.COMPARTMENTS: ".json",
            model_prefix + NAPISTU_STANDARD_OUTPUTS.COMPARTMENTALIZED_SPECIES: ".json",
        }

        # Verify all files exist with correct extensions
        for filename, expected_ext in expected_files.items():
            filepath = os.path.join(tmpdir, filename)
            assert os.path.exists(filepath), f"Expected file {filename} not found"
            assert filename.endswith(
                expected_ext
            ), f"File {filename} should have extension {expected_ext}"

        # Read and validate species_identifiers.tsv
        species_identifiers_path = os.path.join(
            tmpdir, model_prefix + NAPISTU_STANDARD_OUTPUTS.SPECIES_IDENTIFIERS
        )
        species_identifiers = pd.read_csv(species_identifiers_path, sep="\t")
        _check_species_identifiers_table(species_identifiers)

        # Read and validate sid_to_scids.tsv
        sid_to_scids_path = os.path.join(
            tmpdir, model_prefix + NAPISTU_STANDARD_OUTPUTS.SID_TO_SCIDS
        )
        sid_to_scids = pd.read_csv(sid_to_scids_path, sep="\t")
        expected_cols = {SBML_DFS.S_ID, SBML_DFS.SC_ID}
        actual_cols = set(sid_to_scids.columns)
        assert (
            actual_cols == expected_cols
        ), f"sid_to_scids should have columns {expected_cols}, got {actual_cols}"
        assert len(sid_to_scids) > 0, "sid_to_scids table should not be empty"
