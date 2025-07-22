from __future__ import annotations

import os
import pytest

import pandas as pd
from napistu import indices
from napistu import source
from napistu.network import ng_utils
from napistu.constants import SBML_DFS, SOURCE_SPEC

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_source():
    source_example_df = pd.DataFrame(
        [
            {"model": "fun", "id": "baz", "pathway_id": "fun"},
            {"model": "fun", "id": "bot", "pathway_id": "fun"},
            {"model": "time", "id": "boof", "pathway_id": "time"},
            {"model": "time", "id": "bor", "pathway_id": "time"},
        ]
    )

    source_obj = source.Source(source_example_df)
    source_init = source.Source(init=True)

    assert source.merge_sources([source_init, source_init]) == source_init

    pd._testing.assert_frame_equal(
        source.merge_sources([source_obj, source_init]).source, source_example_df
    )

    assert source.merge_sources([source_obj, source_obj]).source.shape[0] == 8

    alt_source_df = pd.DataFrame(
        [
            {"model": "fun", "identifier": "baz", "pathway_id": "fun"},
            {"model": "fun", "identifier": "baz", "pathway_id": "fun"},
        ]
    )
    alt_source_obj = source.Source(alt_source_df)

    assert source.merge_sources([source_obj, alt_source_obj]).source.shape == (6, 4)


def test_source_w_pwindex():
    # pathway_id not provided since this and other attributes will be found
    # in pw_index.tsv
    source_example_df = pd.DataFrame(
        [
            {"model": "R-HSA-1237044", "id": "baz"},
            {"model": "R-HSA-1237044", "id": "bot"},
        ]
    )

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index.tsv"))

    source_obj = source.Source(source_example_df, pw_index=pw_index)
    assert source_obj.source.shape == (2, 8)


def test_get_minimal_source_edges(sbml_dfs_metabolism):
    vertices = sbml_dfs_metabolism.reactions.reset_index().rename(
        columns={SBML_DFS.R_ID: "node"}
    )

    minimal_source_edges = ng_utils.get_minimal_sources_edges(
        vertices, sbml_dfs_metabolism
    )
    # print(minimal_source_edges.shape)
    assert minimal_source_edges.shape == (87, 3)


def test_source_set_coverage(sbml_dfs_metabolism):

    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions)

    # print(source_df.shape)
    assert source_df.shape == (111, 7)

    set_coverage = source.source_set_coverage(source_df)
    # print(set_coverage.shape)
    assert set_coverage.shape == (87, 6)


def test_source_set_coverage_enrichment(sbml_dfs_metabolism):

    source_total_counts = sbml_dfs_metabolism.get_source_total_counts("reactions")

    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions).head(40)

    set_coverage = source.source_set_coverage(
        source_df, source_total_counts=source_total_counts, sbml_dfs=sbml_dfs_metabolism
    )

    assert set_coverage.shape == (34, 6)


def test_source_set_coverage_missing_pathway_ids(sbml_dfs_metabolism):
    """
    Test source_set_coverage when source_total_counts is missing pathway_ids
    that are present in select_sources_df.
    """
    # Get the full source_df
    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions)

    # Get the source_total_counts
    source_total_counts = sbml_dfs_metabolism.get_source_total_counts("reactions")

    # Create a modified source_total_counts that's missing some pathway_ids
    # that are present in source_df
    pathway_ids_in_source = source_df[SOURCE_SPEC.PATHWAY_ID].unique()
    pathway_ids_in_counts = source_total_counts.index.tolist()

    # Remove some pathway_ids from source_total_counts
    pathway_ids_to_remove = pathway_ids_in_counts[:2]  # Remove first 2 pathway_ids
    modified_source_total_counts = source_total_counts.drop(pathway_ids_to_remove)

    # Verify that we have pathway_ids in source_df that are not in modified_source_total_counts
    missing_pathway_ids = set(pathway_ids_in_source) - set(
        modified_source_total_counts.index
    )
    assert (
        len(missing_pathway_ids) > 0
    ), "Test setup failed: no pathway_ids are missing from source_total_counts"

    # Test that the function raises a ValueError when pathway_ids are missing
    with pytest.raises(
        ValueError,
        match="The following pathways are present in `select_sources_df` but not in `source_total_counts`",
    ):
        source.source_set_coverage(
            source_df,
            source_total_counts=modified_source_total_counts,
            sbml_dfs=sbml_dfs_metabolism,
        )
