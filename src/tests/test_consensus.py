from __future__ import annotations

import os

import pandas as pd
import pytest
from napistu import consensus
from napistu import identifiers
from napistu import indices
from napistu import source
from napistu import sbml_dfs_core
from napistu.ingestion import sbml
from napistu.modify import pathwayannot
from napistu.constants import SBML_DFS, SBML_DFS_SCHEMA, SCHEMA_DEFS, IDENTIFIERS, BQB

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_reduce_to_consensus_ids():
    sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")

    # test aggregating by IDs, by moving from compartmentalized_species -> species

    sbml_model = sbml.SBML(sbml_path)
    comp_species_df = sbml_model._define_cspecies()
    comp_species_df.index.names = [SBML_DFS.S_ID]
    consensus_species, species_lookup = consensus.reduce_to_consensus_ids(
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
    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)

    consensus_model = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)
    assert consensus_model.species.shape == (38, 3)
    assert consensus_model.reactions.shape == (30, 4)
    assert consensus_model.reaction_species.shape == (137, 4)

    consensus_model = pathwayannot.drop_cofactors(consensus_model)
    assert consensus_model.species.shape == (38, 3)
    assert consensus_model.reaction_species.shape == (52, 4)
    # update reaction_species.shape after more cofactors identified

    consensus_model.validate()


def test_source_tracking():
    # create input data
    table_schema = {"source": "source_var", "pk": "primary_key"}

    # define existing sources and the new_id entity they belong to
    # here, we are assuming that each model has a blank source object
    # as if it came from a non-consensus model
    agg_tbl = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
        }
    )
    agg_tbl[table_schema["source"]] = source.Source(init=True)

    # define new_ids and the models they came from
    # these models will be matched to the pw_index to flush out metadata
    lookup_table = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
            "model": ["R-HSA-1237044", "R-HSA-425381", "R-HSA-1237044", "R-HSA-425381"],
        }
    )

    # use an existing pw_index since pw_index currently checks for the existence of the source file
    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))

    # test create source table
    source_table = source.create_source_table(lookup_table, table_schema, pw_index)
    assert source_table["source_var"][0].source.shape == (2, 8)

    # test create_consensus_sources
    consensus_sources = consensus.create_consensus_sources(
        agg_tbl, lookup_table, table_schema, pw_index
    )
    assert consensus_sources[0].source.shape == (2, 8)

    # lets add a model which does not have a reference in the pw_index
    invalid_lookup_table = pd.DataFrame(
        {
            "new_id": [0, 0, 1, 1],
            "model": ["R-HSA-1237044", "R-HSA-425381", "R-HSA-1237044", "typo"],
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

    agg_tbl2[table_schema["source"]] = consensus_sources.tolist() + [
        source.Source(init=True) for i in range(0, 2)
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

    consensus_sources = consensus.create_consensus_sources(
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
    minimal_pw_index.index.loc[1, "file"] = minimal_pw_index.index.loc[0, "file"]

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
    assert consensus_model.reactions["r_isreversible"].eq(True).all()


def test_consensus_ontology_check():
    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))

    test_sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)
    test_consensus_model = consensus.construct_consensus_model(
        test_sbml_dfs_dict, pw_index
    )

    pre_shared_onto_sp_list, pre_onto_df = consensus.pre_consensus_ontology_check(
        test_sbml_dfs_dict, "species"
    )
    assert set(pre_shared_onto_sp_list) == {"chebi", "reactome", "uniprot"}

    post_shared_onto_sp_set = consensus.post_consensus_species_ontology_check(
        test_consensus_model
    )
    assert post_shared_onto_sp_set == {"chebi", "reactome", "uniprot"}


def test_report_consensus_merges_reactions(tmp_path):
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
    sbml1 = sbml_dfs_core.SBML_dfs(sbml_dict, validate=False, resolve=False)
    sbml2 = sbml_dfs_core.SBML_dfs(sbml_dict, validate=False, resolve=False)
    sbml_dfs_dict = {"mod1": sbml1, "mod2": sbml2}

    # Create a lookup_table that merges both reactions into a new_id
    lookup_table = pd.DataFrame(
        {
            "model": ["mod1", "mod2"],
            "r_id": [r_id, r_id],
            "new_id": ["merged_rid", "merged_rid"],
        }
    )
    # Use the reactions schema
    table_schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.REACTIONS]

    # Call the function and check that it runs and the merge_labels are as expected
    consensus.report_consensus_merges(
        lookup_table.set_index(["model", "r_id"])[
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
            "s_id": ["A", "B", "C"],
            "s_Identifiers": [
                identifiers.Identifiers(
                    [
                        {
                            IDENTIFIERS.ONTOLOGY: "test",
                            IDENTIFIERS.IDENTIFIER: "X",
                            IDENTIFIERS.BQB: BQB.IS,
                        }
                    ]
                ),
                identifiers.Identifiers([]),
                identifiers.Identifiers(
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
    ).set_index("s_id")

    schema = SBML_DFS_SCHEMA.SCHEMA[SBML_DFS.SPECIES]

    indexed_cluster, cluster_consensus_identifiers = (
        consensus.build_consensus_identifiers(df, schema)
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
    ids_obj = cluster_consensus_identifiers.loc[merged_cluster_id, schema["id"]]
    assert any(i["identifier"] == "X" for i in getattr(ids_obj, "ids", []))

    # The consensus identifier for the entity with no identifiers should be empty
    noid_cluster_id = indexed_cluster.loc["B"]
    ids_obj_noid = cluster_consensus_identifiers.loc[noid_cluster_id, schema["id"]]
    assert hasattr(ids_obj_noid, "ids")
    assert len(getattr(ids_obj_noid, "ids", [])) == 0


################################################
# __main__
################################################

if __name__ == "__main__":
    test_reduce_to_consensus_ids()
    test_consensus()
    test_source_tracking()
    test_passing_entity_data()
    test_consensus_ontology_check()
    test_report_consensus_merges_reactions()
    test_build_consensus_identifiers_handles_merges_and_missing_ids()
