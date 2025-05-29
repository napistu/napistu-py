from __future__ import annotations

import os

import pandas as pd
import pytest
from napistu import consensus
from napistu import indices
from napistu import source
from napistu.ingestion import sbml
from napistu.modify import pathwayannot

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_reduce_to_consensus_ids():
    sbml_path = os.path.join(test_data, "R-HSA-1237044.sbml")

    # test aggregating by IDs, by moving from compartmentalized_species -> species

    sbml_model = sbml.SBML(sbml_path).model
    comp_species_df = sbml.setup_cspecies(sbml_model)
    comp_species_df.index.names = ["s_id"]
    consensus_species, species_lookup = consensus.reduce_to_consensus_ids(
        comp_species_df, {"pk": "s_id", "id": "s_Identifiers"}
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


################################################
# __main__
################################################

if __name__ == "__main__":
    test_reduce_to_consensus_ids()
    test_consensus()
    test_source_tracking()
    test_passing_entity_data()
    test_consensus_ontology_check()
