from __future__ import annotations

import pandas as pd
import pytest

from napistu.constants import SBML_DFS, SBML_DFS_SCHEMA, SCHEMA_DEFS, SOURCE_SPEC
from napistu.modify import curation
from napistu.modify.constants import CURATION_DEFS
from napistu.sbml_dfs_utils import id_formatter_inv
from napistu.source import Source


@pytest.fixture(scope="module")
def curation_dict():
    """Module-level fixture providing mock curation data."""
    curation_dict = dict()
    curation_dict[CURATION_DEFS.SPECIES] = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "hello",
                CURATION_DEFS.URI: "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:35828",
                CURATION_DEFS.CURATOR: "Sean",
            },
            {
                CURATION_DEFS.SPECIES: "good day",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "Sean",
            },
        ]
    )

    curation_dict[CURATION_DEFS.COMPARTMENTALIZED_SPECIES] = pd.DataFrame(
        [
            {
                CURATION_DEFS.COMPARTMENTALIZED_SPECIES: "hello [cytosol]",
                SBML_DFS.S_NAME: "hello",
                SBML_DFS.C_NAME: "cytosol",
                CURATION_DEFS.CURATOR: "Sean",
            }
        ]
    )
    curation_dict[CURATION_DEFS.REACTIONS] = pd.DataFrame(
        [
            {
                CURATION_DEFS.REACTIONS: "there",
                SBML_DFS.STOICHIOMETRY: "hello [cytosol] -> CO2 [cytosol]",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.EVIDENCE: "how is",
                CURATION_DEFS.CURATOR: "Sean",
            },
            {
                CURATION_DEFS.REACTIONS: "where",
                SBML_DFS.STOICHIOMETRY: "CO2 [cytosol] -> hello [cytosol]",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.EVIDENCE: "your family",
                CURATION_DEFS.CURATOR: "Sean",
            },
        ]
    )
    curation_dict[CURATION_DEFS.REACTION_SPECIES] = pd.DataFrame(
        [
            {
                CURATION_DEFS.REACTION_SPECIES: "NADH [cytosol]",
                SBML_DFS.R_NAME: "CYB5Rs reduce MetHb to HbA",
                SBML_DFS.STOICHIOMETRY: 0,
                CURATION_DEFS.SBO_TERM_NAME: "stimulator",
                CURATION_DEFS.EVIDENCE: "weeeee",
                CURATION_DEFS.CURATOR: "Sean",
            }
        ]
    )
    curation_dict[CURATION_DEFS.REMOVE] = pd.DataFrame(
        [
            {
                CURATION_DEFS.REMOVE: "reaction_1237042",
                CURATION_DEFS.TABLE: CURATION_DEFS.REACTIONS,
                CURATION_DEFS.VARIABLE: SBML_DFS.R_ID,
            },
            {
                CURATION_DEFS.REMOVE: "CYB5Rs reduce MetHb to HbA",
                CURATION_DEFS.TABLE: CURATION_DEFS.REACTIONS,
                CURATION_DEFS.VARIABLE: SBML_DFS.R_NAME,
            },
            {
                CURATION_DEFS.REMOVE: "CO2",
                CURATION_DEFS.TABLE: CURATION_DEFS.SPECIES,
                CURATION_DEFS.VARIABLE: SBML_DFS.S_NAME,
            },
        ]
    )

    return curation_dict


def test_remove_entities(sbml_dfs, curation_dict):
    invalid_entities_dict = curation._find_invalid_entities(
        sbml_dfs, curation_dict[CURATION_DEFS.REMOVE]
    )
    invalid_pks = set(invalid_entities_dict.keys())

    assert invalid_pks == {
        SBML_DFS.SC_ID,
        SBML_DFS.RSC_ID,
        SBML_DFS.R_ID,
        SBML_DFS.S_ID,
    }

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


def test_add_entities(sbml_dfs, curation_dict):
    new_entities = curation.format_curations(curation_dict, sbml_dfs)

    assert new_entities[SBML_DFS.SPECIES].shape == (2, 3)
    assert new_entities[SBML_DFS.REACTIONS].shape == (2, 4)
    assert new_entities[SBML_DFS.COMPARTMENTALIZED_SPECIES].shape == (1, 4)
    assert new_entities[SBML_DFS.REACTION_SPECIES].shape == (5, 4)


def test_format_curated_entities_species_with_uri(sbml_dfs):
    """Test formatting species entities with URI identifiers."""
    new_species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "test_species",
                CURATION_DEFS.URI: "http://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:35828",
                CURATION_DEFS.CURATOR: "TestCurator",
            }
        ]
    )

    new_entities = {}
    result = curation.format_curated_entities(
        SBML_DFS.SPECIES, new_species_df, new_entities, sbml_dfs
    )

    # Check shape and columns
    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]
    assert result.shape[0] == 1
    assert set(result.columns) == set(schema[SCHEMA_DEFS.VARS])
    assert result.index.name == schema[SCHEMA_DEFS.PK]

    # Check that primary key was generated (uppercase format)
    assert result.index[0].startswith("S")

    # Check label was set
    assert result[schema[SCHEMA_DEFS.LABEL]].iloc[0] == "test_species"

    # Check identifier was created from URI
    identifiers = result[schema[SCHEMA_DEFS.ID]].iloc[0]
    assert identifiers is not None


def test_format_curated_entities_species_without_uri(sbml_dfs):
    """Test formatting species entities without URI (custom species)."""
    new_species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "custom_species",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "TestCurator",
            }
        ]
    )

    new_entities = {}
    result = curation.format_curated_entities(
        SBML_DFS.SPECIES, new_species_df, new_entities, sbml_dfs
    )

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]
    assert result.shape[0] == 1

    # Check identifier was created as custom
    identifiers = result[schema[SCHEMA_DEFS.ID]].iloc[0]
    assert identifiers is not None


def test_format_curated_entities_reactions(sbml_dfs):
    """Test formatting reaction entities."""
    new_reactions_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.REACTIONS: "test_reaction",
                SBML_DFS.STOICHIOMETRY: "A -> B",
                SBML_DFS.R_ISREVERSIBLE: False,
                CURATION_DEFS.URI: None,
                CURATION_DEFS.EVIDENCE: "test evidence",
                CURATION_DEFS.CURATOR: "TestCurator",
            }
        ]
    )

    new_entities = {}
    result = curation.format_curated_entities(
        SBML_DFS.REACTIONS, new_reactions_df, new_entities, sbml_dfs
    )

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.REACTIONS]
    assert result.shape[0] == 1
    assert set(result.columns) == set(schema[SCHEMA_DEFS.VARS])
    assert result.index.name == schema[SCHEMA_DEFS.PK]

    # Check label was set
    assert result[schema[SCHEMA_DEFS.LABEL]].iloc[0] == "test_reaction"

    # Check primary key was generated (uppercase format)
    assert result.index[0].startswith("R")


def test_format_curated_entities_compartmentalized_species(sbml_dfs):
    """Test formatting compartmentalized species with foreign keys."""
    # First add a species and compartment that we can reference
    species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "test_species",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "TestCurator",
            }
        ]
    )
    new_entities = {}
    species_result = curation.format_curated_entities(
        SBML_DFS.SPECIES, species_df, new_entities, sbml_dfs
    )
    new_entities[SBML_DFS.SPECIES] = species_result

    # Get an existing compartment name (label), not ID
    compartment_name = sbml_dfs.compartments[SBML_DFS.C_NAME].iloc[0]
    compartment_id = sbml_dfs.compartments.index[0]

    # Now add compartmentalized species
    comp_species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.COMPARTMENTALIZED_SPECIES: f"test_species [{compartment_name}]",
                SBML_DFS.S_NAME: "test_species",
                SBML_DFS.C_NAME: compartment_name,
                CURATION_DEFS.CURATOR: "TestCurator",
            }
        ]
    )

    result = curation.format_curated_entities(
        SBML_DFS.COMPARTMENTALIZED_SPECIES, comp_species_df, new_entities, sbml_dfs
    )

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.COMPARTMENTALIZED_SPECIES]
    assert result.shape[0] == 1
    assert set(result.columns) == set(schema[SCHEMA_DEFS.VARS])

    # Check foreign keys were set correctly
    assert result[SBML_DFS.S_ID].iloc[0] == species_result.index[0]
    assert result[SBML_DFS.C_ID].iloc[0] == compartment_id


def test_format_curated_entities_primary_key_generation(sbml_dfs):
    """Test that primary keys are generated sequentially."""
    # Get initial max PK
    pk_nums = id_formatter_inv(sbml_dfs.species.index.tolist())
    initial_max = max(pk_nums) if len(pk_nums) > 0 else -1

    # Add multiple species
    new_species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "species1",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "TestCurator",
            },
            {
                CURATION_DEFS.SPECIES: "species2",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "TestCurator",
            },
        ]
    )

    new_entities = {}
    result = curation.format_curated_entities(
        SBML_DFS.SPECIES, new_species_df, new_entities, sbml_dfs
    )

    assert result.shape[0] == 2

    # Check PKs are sequential
    pk1_num = id_formatter_inv([result.index[0]])[0]
    pk2_num = id_formatter_inv([result.index[1]])[0]

    assert pk1_num == initial_max + 1
    assert pk2_num == initial_max + 2


def test_format_curated_entities_source_creation(sbml_dfs):
    """Test that Source objects are created correctly."""
    new_species_df = pd.DataFrame(
        [
            {
                CURATION_DEFS.SPECIES: "test_species",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: "TestCurator",
            },
            {
                CURATION_DEFS.SPECIES: "test_species2",
                CURATION_DEFS.URI: None,
                CURATION_DEFS.CURATOR: None,
            },
        ]
    )

    new_entities = {}
    result = curation.format_curated_entities(
        SBML_DFS.SPECIES,
        new_species_df,
        new_entities,
        sbml_dfs,
        curation_id="test_curation",
    )

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]
    sources = result[schema[SCHEMA_DEFS.SOURCE]]

    # Check sources were created
    assert isinstance(sources.iloc[0], Source)
    assert isinstance(sources.iloc[1], Source)

    # Check curator None was filled with "unknown"
    assert sources.iloc[1].source.iloc[0][SOURCE_SPEC.MODEL] == CURATION_DEFS.UNKNOWN
