from __future__ import annotations

import logging
import os

import pandas as pd
import pytest

from napistu import consensus, indices, sbml_dfs_core, source
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    SBML_DFS,
    SBML_DFS_METADATA,
    SBML_DFS_SCHEMA,
    SCHEMA_DEFS,
    SOURCE_SPEC,
)
from napistu.identifiers import Identifiers
from napistu.ingestion import sbml
from napistu.modify.cofactors import drop_cofactors

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_reduce_to_consensus_ids():
    sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")

    # test aggregating by IDs, by moving from compartmentalized_species -> species

    sbml_model = sbml.SBML(sbml_path)
    comp_species_df = sbml_model._define_cspecies()
    comp_species_df.index.names = [SBML_DFS.S_ID]
    consensus_species, species_lookup = consensus._reduce_to_consensus_ids(
        comp_species_df,
        {
            SCHEMA_DEFS.PK: SBML_DFS.S_ID,
            SCHEMA_DEFS.ID: SBML_DFS.S_IDENTIFIERS,
            SCHEMA_DEFS.TABLE: SBML_DFS.SPECIES,
        },
    )

    assert isinstance(consensus_species, pd.DataFrame)
    assert consensus_species.shape == (18, 4)
    assert isinstance(species_lookup, pd.Series)
    assert species_lookup.size == 23


def test_consensus():
    pw_index = indices.PWIndex(os.path.join(test_data, SOURCE_SPEC.PW_INDEX_FILE))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)

    consensus_model = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)
    assert consensus_model.species.shape == (38, 3)
    assert consensus_model.reactions.shape == (30, 4)
    assert consensus_model.reaction_species.shape == (137, 4)

    consensus_model.validate()

    consensus_model = drop_cofactors(consensus_model)
    assert consensus_model.species.shape == (30, 3)
    assert consensus_model.reaction_species.shape == (52, 4)
    # update reaction_species.shape after more cofactors identified

    consensus_model.validate()


def test_source_tracking():
    # create input data
    table_schema = {SCHEMA_DEFS.SOURCE: "source_var", SCHEMA_DEFS.PK: "primary_key"}

    # define existing sources and the new_id entity they belong to
    # here, we are assuming that each model has a blank source object
    # as if it came from a non-consensus model
    agg_tbl = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
        }
    )
    agg_tbl[table_schema[SCHEMA_DEFS.SOURCE]] = source.Source.empty()

    # define new_ids and the models they came from
    # these models will be matched to the pw_index to flush out metadata
    lookup_table = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
            SOURCE_SPEC.MODEL: [
                "R-HSA-1237044",
                "R-HSA-425381",
                "R-HSA-1237044",
                "R-HSA-425381",
            ],
        }
    )

    # use an existing pw_index since pw_index currently checks for the existence of the source file
    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))

    # test create source table
    source_table = source.create_source_table(lookup_table, table_schema, pw_index)
    assert source_table["source_var"][0].source.shape == (2, 8)

    # test create_consensus_sources
    consensus_sources = consensus._create_consensus_sources(
        agg_tbl, lookup_table, table_schema, pw_index
    )
    assert consensus_sources[0].source.shape == (2, 8)

    # lets add a model which does not have a reference in the pw_index
    invalid_lookup_table = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
            SOURCE_SPEC.MODEL: [
                "R-HSA-1237044",
                "R-HSA-425381",
                "R-HSA-1237044",
                "typo",
            ],
        }
    )

    # expect a ValueError when the model is not found
    with pytest.raises(ValueError) as _:
        source.create_source_table(invalid_lookup_table, table_schema, pw_index)

    # now we will aggregate the consensus model above with a new single model (which has some
    # overlapping entries with the consensusd (id 1) and some new ids (id 2)

    agg_tbl2 = pd.DataFrame(
        {
            "new_id": [0, 1, 1, 2],
        }
    )

    agg_tbl2[table_schema[SCHEMA_DEFS.SOURCE]] = consensus_sources.tolist() + [
        source.Source.empty() for i in range(0, 2)
    ]

    lookup_table2 = pd.DataFrame(
        {
            "new_id": [0, 1, 1, 2],
            # the model for the first two entries should really correspond to the "consensus"
            # but since this is not a file I will stub with one of the pw_index entries
            "model": [
                "R-HSA-1247673",
                "R-HSA-1247673",
                "R-HSA-1475029",
                "R-HSA-1475029",
            ],
        }
    )

    source_table = source.create_source_table(lookup_table2, table_schema, pw_index)
    assert source_table.shape == (3, 1)
    assert [
        source_table["source_var"][i].source.shape
        for i in range(0, source_table.shape[0])
    ] == [(1, 8), (2, 8), (1, 8)]

    consensus_sources = consensus._create_consensus_sources(
        agg_tbl2, lookup_table2, table_schema, pw_index
    )
    assert [
        consensus_sources[i].source.shape for i in range(0, consensus_sources.shape[0])
    ] == [(3, 8), (4, 8), (1, 8)]


def test_passing_entity_data():

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)

    for model in list(sbml_dfs_dict.keys())[0:3]:
        sbml_dfs_dict[model].add_species_data(
            "my_species_data",
            sbml_dfs_dict[model]
            .species.iloc[0:5]
            .assign(my_species_data_var="testing")["my_species_data_var"]
            .to_frame(),
        )
        sbml_dfs_dict[model].add_reactions_data(
            "my_reactions_data",
            sbml_dfs_dict[model]
            .reactions.iloc[0:5]
            .assign(my_reactions_data_var1="testing")
            .assign(my_reactions_data_var2="testing2")[
                ["my_reactions_data_var1", "my_reactions_data_var2"]
            ],
        )

    # create a consensus with perfect merges of overlapping id-table-variable values
    # i.e., when combined all merged entries have the same attributes
    consensus_model = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)

    assert len(consensus_model.species_data) == 1
    assert consensus_model.species_data["my_species_data"].shape == (10, 1)
    assert len(consensus_model.reactions_data) == 1
    assert consensus_model.reactions_data["my_reactions_data"].shape == (14, 2)

    # add different tables from different models
    for model in list(sbml_dfs_dict.keys())[3:5]:
        sbml_dfs_dict[model].add_species_data(
            "my_other_species_data",
            sbml_dfs_dict[model]
            .species.iloc[0:5]
            .assign(my_species_data="testing")["my_species_data"]
            .to_frame(),
        )

    consensus_model = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)
    assert len(consensus_model.species_data) == 2

    # create a case where reactions will be merged and the same reaction
    # in different models has a different value for its reactions_data
    minimal_pw_index = pw_index
    minimal_pw_index.index = minimal_pw_index.index.iloc[0:2]

    # Since we're working with a DataFrame, we can use loc to update the file value directly
    minimal_pw_index.index.loc[1, SOURCE_SPEC.FILE] = minimal_pw_index.index.loc[
        0, SOURCE_SPEC.FILE
    ]

    duplicated_sbml_dfs_dict = consensus.construct_sbml_dfs_dict(minimal_pw_index)
    # explicitely define the order we'll loop through models so that
    # the position of a model can be used to set mismatching attributes
    # for otherwise identical models
    model_order = list(duplicated_sbml_dfs_dict.keys())

    for model in duplicated_sbml_dfs_dict.keys():
        model_index = model_order.index(model)

        duplicated_sbml_dfs_dict[model].add_reactions_data(
            "my_mismatched_data",
            duplicated_sbml_dfs_dict[model]
            .reactions.iloc[0:5]
            .assign(my_reactions_data_var1=model)["my_reactions_data_var1"]
            .to_frame()
            .assign(numeric_var=[x + model_index for x in range(0, 5)])
            .assign(bool_var=[x + model_index % 2 == 0 for x in range(0, 5)]),
        )

    # assign reversibility is True for one model to
    # confirm that reversibility trumps irreversible
    # when merging reactions with identical stoichiometry but
    # different reversibility attributes

    duplicated_sbml_dfs_dict["R-HSA-1237044"].reactions = duplicated_sbml_dfs_dict[
        "R-HSA-1237044"
    ].reactions.assign(r_isreversible=True)

    consensus_model = consensus.construct_consensus_model(
        duplicated_sbml_dfs_dict, pw_index
    )
    assert consensus_model.reactions_data["my_mismatched_data"].shape == (5, 3)
    assert consensus_model.reactions[SBML_DFS.R_ISREVERSIBLE].eq(True).all()


def test_report_consensus_merges_reactions(tmp_path, model_source_stub):
    # Create two minimal SBML_dfs objects with a single reaction each, same r_id
    r_id = "R00000001"
    reactions = pd.DataFrame(
        {
            SBML_DFS.R_NAME: ["rxn1"],
            SBML_DFS.R_IDENTIFIERS: [None],
            SBML_DFS.R_SOURCE: [None],
            SBML_DFS.R_ISREVERSIBLE: [False],
        },
        index=[r_id],
    )
    reactions.index.name = SBML_DFS.R_ID
    reaction_species = pd.DataFrame(
        {
            SBML_DFS.R_ID: [r_id],
            SBML_DFS.SC_ID: ["SC0001"],
            SBML_DFS.STOICHIOMETRY: [1],
            SBML_DFS.SBO_TERM: ["SBO:0000459"],
        },
        index=["RSC0001"],
    )
    reaction_species.index.name = SBML_DFS.RSC_ID
    compartmentalized_species = pd.DataFrame(
        {
            SBML_DFS.SC_NAME: ["A [cytosol]"],
            SBML_DFS.S_ID: ["S0001"],
            SBML_DFS.C_ID: ["C0001"],
            SBML_DFS.SC_SOURCE: [None],
        },
        index=["SC0001"],
    )
    compartmentalized_species.index.name = SBML_DFS.SC_ID
    species = pd.DataFrame(
        {
            SBML_DFS.S_NAME: ["A"],
            SBML_DFS.S_IDENTIFIERS: [None],
            SBML_DFS.S_SOURCE: [None],
        },
        index=["S0001"],
    )
    species.index.name = SBML_DFS.S_ID
    compartments = pd.DataFrame(
        {
            SBML_DFS.C_NAME: ["cytosol"],
            SBML_DFS.C_IDENTIFIERS: [None],
            SBML_DFS.C_SOURCE: [None],
        },
        index=["C0001"],
    )
    compartments.index.name = SBML_DFS.C_ID
    sbml_dict = {
        SBML_DFS.COMPARTMENTS: compartments,
        SBML_DFS.SPECIES: species,
        SBML_DFS.COMPARTMENTALIZED_SPECIES: compartmentalized_species,
        SBML_DFS.REACTIONS: reactions,
        SBML_DFS.REACTION_SPECIES: reaction_species,
    }
    sbml1 = sbml_dfs_core.SBML_dfs(
        sbml_dict, model_source_stub, validate=False, resolve=False
    )
    sbml2 = sbml_dfs_core.SBML_dfs(
        sbml_dict, model_source_stub, validate=False, resolve=False
    )
    sbml_dfs_dict = {"mod1": sbml1, "mod2": sbml2}

    # Create a lookup_table that merges both reactions into a new_id
    lookup_table = pd.DataFrame(
        {
            SOURCE_SPEC.MODEL: ["mod1", "mod2"],
            SBML_DFS.R_ID: [r_id, r_id],
            "new_id": ["merged_rid", "merged_rid"],
        }
    )
    # Use the reactions schema
    table_schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.REACTIONS]

    # Call the function and check that it runs and the merge_labels are as expected
    consensus._report_consensus_merges(
        lookup_table.set_index([SOURCE_SPEC.MODEL, SBML_DFS.R_ID])[
            "new_id"
        ],  # this is a Series with name 'new_id'
        table_schema,
        sbml_dfs_dict=sbml_dfs_dict,
        n_example_merges=1,
    )
    # No assertion: this is a smoke test to ensure the Series output is handled without error


def test_build_consensus_identifiers_handles_merges_and_missing_ids():

    # Three entities:
    # - 'A' with identifier X
    # - 'B' with no identifiers
    # - 'C' with identifier X (should merge with 'A')
    df = pd.DataFrame(
        {
            SBML_DFS.S_ID: ["A", "B", "C"],
            SBML_DFS.S_IDENTIFIERS: [
                Identifiers(
                    [
                        {
                            IDENTIFIERS.ONTOLOGY: "test",
                            IDENTIFIERS.IDENTIFIER: "X",
                            IDENTIFIERS.BQB: BQB.IS,
                        }
                    ]
                ),
                Identifiers([]),
                Identifiers(
                    [
                        {
                            IDENTIFIERS.ONTOLOGY: "test",
                            IDENTIFIERS.IDENTIFIER: "X",
                            IDENTIFIERS.BQB: BQB.IS,
                        }
                    ]
                ),
            ],
        }
    ).set_index(SBML_DFS.S_ID)

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]

    indexed_cluster, cluster_consensus_identifiers = (
        consensus._build_consensus_identifiers(df, schema)
    )

    # All entities should be assigned to a cluster
    assert set(indexed_cluster.index) == set(df.index)
    assert not indexed_cluster.isnull().any()
    # There should be a consensus identifier for each cluster
    assert set(cluster_consensus_identifiers.index) == set(indexed_cluster.values)

    # Entities 'A' and 'C' should be merged (same cluster)
    assert indexed_cluster.loc["A"] == indexed_cluster.loc["C"]
    # Entity 'B' should be in a different cluster
    assert indexed_cluster.loc["B"] != indexed_cluster.loc["A"]

    # The consensus identifier for the merged cluster should include identifier X
    merged_cluster_id = indexed_cluster.loc["A"]
    ids_obj = cluster_consensus_identifiers.loc[
        merged_cluster_id, schema[SCHEMA_DEFS.ID]
    ]
    assert "X" in ids_obj.df[IDENTIFIERS.IDENTIFIER].values

    # The consensus identifier for the entity with no identifiers should be empty
    noid_cluster_id = indexed_cluster.loc["B"]
    ids_obj_noid = cluster_consensus_identifiers.loc[
        noid_cluster_id, schema[SCHEMA_DEFS.ID]
    ]
    assert hasattr(ids_obj_noid, "df")
    assert ids_obj_noid.df.shape[0] == 0


def test_consensus_round_trip_consistency(
    pw_index_metabolism, sbml_dfs_dict_metabolism, sbml_dfs_metabolism
):
    """
    Test that round-trip conversion between different data structures produces consistent results.

    This test verifies that:
    1. sbml_dfs_dict -> sbml_dfs_list -> (sbml_dfs_dict, pw_index) -> consensus_model
    2. pw_index -> sbml_dfs_dict -> consensus_model

    Both paths should produce the same consensus model.
    """

    # Path 1: pw_index -> sbml_dfs_dict -> consensus_model was already created in conftest.py
    pw_index = pw_index_metabolism
    sbml_dfs_dict = sbml_dfs_dict_metabolism
    sbml_dfs = sbml_dfs_metabolism

    # Path 2: sbml_dfs_dict -> sbml_dfs_list -> (sbml_dfs_dict, pw_index) -> consensus_model
    sbml_dfs_list = list(sbml_dfs_dict.values())
    sbml_dfs_dict_from_list, pw_index_from_list = consensus.prepare_consensus_model(
        sbml_dfs_list
    )
    sbml_dfs_from_list = consensus.construct_consensus_model(
        sbml_dfs_dict_from_list, pw_index_from_list
    )

    # Verify both paths produce identical consensus models
    assert (
        sbml_dfs.species.shape == sbml_dfs_from_list.species.shape
    ), "Species tables should have same shape"
    assert (
        sbml_dfs.reactions.shape == sbml_dfs_from_list.reactions.shape
    ), "Reactions tables should have same shape"
    assert (
        sbml_dfs.reaction_species.shape == sbml_dfs_from_list.reaction_species.shape
    ), "Reaction species tables should have same shape"

    # Verify the pw_index objects are equivalent
    assert (
        pw_index.index.shape == pw_index_from_list.index.shape
    ), "PWIndex objects should have same shape"

    # Compare columns using set comparison
    assert set(pw_index.index.columns) == set(
        pw_index_from_list.index.columns
    ), "PWIndex objects should have same columns"

    # Compare the actual pw_index data, ignoring row/column order and dtype differences
    pd.testing.assert_frame_equal(
        pw_index.index,
        pw_index_from_list.index,
        check_like=True,  # Ignore the order of index & columns
        check_dtype=False,  # Ignore dtype differences (e.g., float64 vs object for date column)
    )

    # Verify the sbml_dfs_dict objects are equivalent
    assert len(sbml_dfs_dict) == len(
        sbml_dfs_dict_from_list
    ), "SBML_dfs dictionaries should have same number of entries"
    assert set(sbml_dfs_dict.keys()) == set(
        sbml_dfs_dict_from_list.keys()
    ), "SBML_dfs dictionaries should have same keys"

    # Compare the metadata of each paired SBML_dfs object
    for key in sbml_dfs_dict.keys():
        original_sbml_dfs = sbml_dfs_dict[key]
        reconstructed_sbml_dfs = sbml_dfs_dict_from_list[key]

        # Compare metadata using pandas testing
        pd.testing.assert_frame_equal(
            original_sbml_dfs.metadata[SBML_DFS_METADATA.SBML_DFS_SOURCE].source,
            reconstructed_sbml_dfs.metadata[SBML_DFS_METADATA.SBML_DFS_SOURCE].source,
        )


def test_pre_consensus_compartment_check_compatible(
    sbml_dfs_dict_metabolism, pw_index_metabolism
):
    """Test that compatible models with overlapping compartments pass the check."""
    # This should not raise any errors or log warnings
    consensus._pre_consensus_compartment_check(
        sbml_dfs_dict_metabolism, pw_index_metabolism
    )


def test_pre_consensus_compartment_check_incompatible(
    caplog, minimal_valid_sbml_dfs, sbml_dfs
):
    """Test that incompatible models with non-overlapping compartments are detected."""

    # Create a pathway index with both models
    pw_index_data = pd.DataFrame(
        {
            SOURCE_SPEC.FILE: ["fake_model", "real_model"],
            SOURCE_SPEC.DATA_SOURCE: ["Test", "Reactome"],
            SOURCE_SPEC.ORGANISMAL_SPECIES: ["Homo sapiens", "Homo sapiens"],
            SOURCE_SPEC.PATHWAY_ID: ["fake_model", "real_model"],
            SOURCE_SPEC.NAME: ["Fake Pathway", "Real Pathway"],
            SOURCE_SPEC.DATE: ["2023-01-01", "2023-01-01"],
        }
    )

    # Create SBML_dfs dictionary with incompatible models
    # minimal_valid_sbml_dfs has no identifiers, so no merging will occur
    sbml_dfs_dict = {"fake_model": minimal_valid_sbml_dfs, "real_model": sbml_dfs}

    # Create pathway index object
    pw_index = indices.PWIndex(pw_index_data, validate_paths=False)

    # Capture log messages
    with caplog.at_level(logging.ERROR):
        consensus._pre_consensus_compartment_check(sbml_dfs_dict, pw_index)

    # Check that an error was logged about incompatible compartments
    assert len(caplog.records) > 0
    assert any("incompatible" in record.message.lower() for record in caplog.records)


def test_pre_consensus_ontology_check_compatible(sbml_dfs_dict_metabolism):
    """Test that compatible models with overlapping ontologies pass the check."""
    # This should not raise any errors or log warnings
    consensus._pre_consensus_ontology_check(sbml_dfs_dict_metabolism, SBML_DFS.SPECIES)


def test_pre_consensus_ontology_check_incompatible(
    caplog, minimal_valid_sbml_dfs, sbml_dfs
):
    """Test that incompatible models with non-overlapping ontologies are detected."""

    # Create a modified minimal SBML_dfs with fake ontology identifiers
    fake_sbml_dfs = minimal_valid_sbml_dfs.copy()

    # Create fake identifiers with a fake ontology
    fake_identifiers = Identifiers(
        [
            {
                IDENTIFIERS.ONTOLOGY: "FAKE_ONTOLOGY",
                IDENTIFIERS.IDENTIFIER: "FAKE_ID",
                IDENTIFIERS.BQB: BQB.IS,
            }
        ]
    )

    # Update the species identifiers to use the fake ontology
    fake_species = fake_sbml_dfs.species.copy()
    fake_species[SBML_DFS.S_IDENTIFIERS] = [fake_identifiers]
    fake_sbml_dfs.species = fake_species

    # Create SBML_dfs dictionary with incompatible models
    sbml_dfs_dict = {"fake_model": fake_sbml_dfs, "real_model": sbml_dfs}

    # Capture log messages
    with caplog.at_level(logging.ERROR):
        consensus._pre_consensus_ontology_check(sbml_dfs_dict, SBML_DFS.SPECIES)

    # Check that an error was logged about incompatible ontologies
    assert len(caplog.records) > 0
    assert any("ontologies" in record.message.lower() for record in caplog.records)
    assert any(
        "incompatible" in record.message.lower() or "overlap" in record.message.lower()
        for record in caplog.records
    )


def test_consensus_species_merged(sbml_dfs_metabolism):
    """Test that species identifiers are properly merged during consensus building."""
    # Check for species merging; are defining IDs shared across species?
    species_ids = sbml_dfs_metabolism.get_identifiers(
        SBML_DFS.SPECIES, filter_by_bqb="defining"
    )
    assert all(
        species_ids.value_counts([IDENTIFIERS.ONTOLOGY, IDENTIFIERS.IDENTIFIER]) == 1
    )


def test_update_foreign_keys():
    """Test _update_foreign_keys function with various scenarios."""
    # Create test data
    agg_tbl = pd.DataFrame(
        {
            SOURCE_SPEC.MODEL: ["model_a", "model_a", "model_b", "model_b"],
            "entity_id": ["e1", "e2", "e3", "e4"],
            SBML_DFS.S_ID: ["s1", "s2", "s1", "s3"],
            "data": [10, 20, 30, 40],
        }
    ).set_index([SOURCE_SPEC.MODEL, "entity_id"])

    # Create FK lookup table (multiindex: model, old_id -> new_id)
    fk_lookup = pd.DataFrame(
        {"new_id": ["new_s1", "new_s2", "new_s1", "new_s3"]},
        index=pd.MultiIndex.from_tuples(
            [
                ("model_a", "s1"),
                ("model_a", "s2"),
                ("model_b", "s1"),
                ("model_b", "s3"),
            ],
            names=[SOURCE_SPEC.MODEL, SBML_DFS.S_ID],
        ),
    )

    # Create table schema
    table_schema = {SCHEMA_DEFS.FK: [SBML_DFS.S_ID], SCHEMA_DEFS.PK: "entity_id"}

    fk_lookup_tables = {SBML_DFS.S_ID: fk_lookup}

    # Test successful FK update
    result = consensus._update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)

    # Verify the foreign keys were updated correctly
    expected_species_ids = ["new_s1", "new_s2", "new_s1", "new_s3"]
    assert result[SBML_DFS.S_ID].tolist() == expected_species_ids
    assert result["data"].tolist() == [10, 20, 30, 40]  # Other data preserved


def test_update_foreign_keys_missing_key():
    """Test _update_foreign_keys raises ValueError for missing keys."""
    # Create test data with a key that won't be in the lookup table
    agg_tbl = pd.DataFrame(
        {
            SOURCE_SPEC.MODEL: ["model_a", "model_a"],
            "entity_id": ["e1", "e2"],
            SBML_DFS.S_ID: ["s1", "s_missing"],  # s_missing not in lookup
            "data": [10, 20],
        }
    ).set_index([SOURCE_SPEC.MODEL, "entity_id"])

    # Create FK lookup table (missing s_missing)
    fk_lookup = pd.DataFrame(
        {"new_id": ["new_s1"]},
        index=pd.MultiIndex.from_tuples(
            [("model_a", "s1")], names=[SOURCE_SPEC.MODEL, SBML_DFS.S_ID]
        ),
    )

    table_schema = {SCHEMA_DEFS.FK: [SBML_DFS.S_ID], SCHEMA_DEFS.PK: "entity_id"}

    fk_lookup_tables = {SBML_DFS.S_ID: fk_lookup}

    # Should raise ValueError for missing key
    with pytest.raises(
        ValueError,
        match=f"keys from agg_tbl are missing from the {SBML_DFS.S_ID} lookup table",
    ):
        consensus._update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)


def test_update_foreign_keys_multiple_fks():
    """Test _update_foreign_keys with multiple foreign keys using compartmentalized species."""
    # Create test data for compartmentalized species (has both s_id and c_id FKs)
    agg_tbl = pd.DataFrame(
        {
            SOURCE_SPEC.MODEL: ["model_a", "model_a"],
            SBML_DFS.SC_ID: ["sc1", "sc2"],
            SBML_DFS.S_ID: ["s1", "s2"],
            SBML_DFS.C_ID: ["c1", "c2"],
            "data": [10, 20],
        }
    ).set_index([SOURCE_SPEC.MODEL, SBML_DFS.SC_ID])

    # Create lookup tables for both FKs
    species_lookup = pd.DataFrame(
        {"new_id": ["new_s1", "new_s2"]},
        index=pd.MultiIndex.from_tuples(
            [("model_a", "s1"), ("model_a", "s2")],
            names=[SOURCE_SPEC.MODEL, SBML_DFS.S_ID],
        ),
    )

    compartment_lookup = pd.DataFrame(
        {"new_id": ["new_c1", "new_c2"]},
        index=pd.MultiIndex.from_tuples(
            [("model_a", "c1"), ("model_a", "c2")],
            names=[SOURCE_SPEC.MODEL, SBML_DFS.C_ID],
        ),
    )

    table_schema = {
        SCHEMA_DEFS.FK: [SBML_DFS.S_ID, SBML_DFS.C_ID],
        SCHEMA_DEFS.PK: SBML_DFS.SC_ID,
    }

    fk_lookup_tables = {
        SBML_DFS.S_ID: species_lookup,
        SBML_DFS.C_ID: compartment_lookup,
    }

    # Test updating both FKs
    result = consensus._update_foreign_keys(agg_tbl, table_schema, fk_lookup_tables)

    # Verify both FKs were updated
    assert result[SBML_DFS.S_ID].tolist() == ["new_s1", "new_s2"]
    assert result[SBML_DFS.C_ID].tolist() == ["new_c1", "new_c2"]
    assert result["data"].tolist() == [10, 20]  # Other data preserved


def test_construct_consensus_model_no_rxn_pathway_ids(
    sbml_dfs_dict_metabolism, pw_index_metabolism, sbml_dfs_metabolism
):
    """Test construct_consensus_model with no_rxn_pathway_ids parameter."""
    # Test with one pathway marked as no_rxn_pathway_ids
    no_rxn_pathway_ids = ["tca"]  # Select TCA cycle as no-reaction pathway

    # Create consensus model with no_rxn_pathway_ids
    consensus_model = consensus.construct_consensus_model(
        sbml_dfs_dict_metabolism,
        pw_index_metabolism,
        no_rxn_pathway_ids=no_rxn_pathway_ids,
    )

    assert consensus_model.reactions.shape[0] < sbml_dfs_metabolism.reactions.shape[0]
    # cleanup generally removes some unused cspecies, species, and in some cases compartments
    assert consensus_model.species.shape[0] < sbml_dfs_metabolism.species.shape[0]
    assert (
        consensus_model.compartmentalized_species.shape[0]
        < sbml_dfs_metabolism.compartmentalized_species.shape[0]
    )

    # Test with invalid pathway ID
    invalid_no_rxn_pathway_ids = ["invalid_pathway_id"]

    # Should raise ValueError when invalid pathway ID is provided
    with pytest.raises(ValueError) as exc_info:
        consensus.construct_consensus_model(
            sbml_dfs_dict_metabolism,
            pw_index_metabolism,
            no_rxn_pathway_ids=invalid_no_rxn_pathway_ids,
        )

    # Verify the error message mentions the invalid pathway ID
    assert "invalid_pathway_id" in str(exc_info.value)
    assert "not found in the pw_index" in str(exc_info.value)
