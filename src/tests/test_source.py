from __future__ import annotations

import os
import pytest

import pandas as pd
from napistu import indices
from napistu import source
from napistu.network import ng_utils
from napistu.constants import SBML_DFS, SOURCE_SPEC
from napistu.statistics.constants import CONTINGENCY_TABLE

test_path = os.path.abspath(os.path.join(__file__, os.pardir))
test_data = os.path.join(test_path, "test_data")


def test_source():
    source_example_df = pd.DataFrame(
        [
            {SOURCE_SPEC.MODEL: "fun", "id": "baz", SOURCE_SPEC.PATHWAY_ID: "fun"},
            {SOURCE_SPEC.MODEL: "fun", "id": "bot", SOURCE_SPEC.PATHWAY_ID: "fun"},
            {SOURCE_SPEC.MODEL: "time", "id": "boof", SOURCE_SPEC.PATHWAY_ID: "time"},
            {SOURCE_SPEC.MODEL: "time", "id": "bor", SOURCE_SPEC.PATHWAY_ID: "time"},
        ]
    )

    source_obj = source.Source(source_example_df)
    source_init = source.Source.empty()

    assert source.merge_sources([source_init, source_init]) == source_init

    pd._testing.assert_frame_equal(
        source.merge_sources([source_obj, source_init]).source, source_example_df
    )

    assert source.merge_sources([source_obj, source_obj]).source.shape[0] == 8

    alt_source_df = pd.DataFrame(
        [
            {
                SOURCE_SPEC.MODEL: "fun",
                "identifier": "baz",
                SOURCE_SPEC.PATHWAY_ID: "fun",
            },
            {
                SOURCE_SPEC.MODEL: "fun",
                "identifier": "baz",
                SOURCE_SPEC.PATHWAY_ID: "fun",
            },
        ]
    )
    alt_source_obj = source.Source(alt_source_df)

    assert source.merge_sources([source_obj, alt_source_obj]).source.shape == (6, 4)


def test_source_w_pwindex():
    # pathway_id not provided since this and other attributes will be found
    # in pw_index.tsv
    source_example_df = pd.DataFrame(
        [
            {SOURCE_SPEC.MODEL: "R-HSA-1237044", "id": "baz"},
            {SOURCE_SPEC.MODEL: "R-HSA-1237044", "id": "bot"},
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

    source_total_counts = sbml_dfs_metabolism.get_source_total_counts(
        SBML_DFS.REACTIONS
    )

    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions).head(40)

    set_coverage = source.source_set_coverage(
        source_df, source_total_counts=source_total_counts, sbml_dfs=sbml_dfs_metabolism
    )

    # Check that we have the expected number of rows and at least the required columns
    assert set_coverage.shape[0] == 34  # 34 rows
    assert (
        set_coverage.shape[1] >= 6
    )  # At least 6 columns (may have additional debug columns)


def test_source_set_coverage_missing_pathway_ids(sbml_dfs_metabolism):
    """
    Test source_set_coverage when source_total_counts is missing pathway_ids
    that are present in select_sources_df.
    """
    # Get the full source_df
    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions)

    # Get the source_total_counts
    source_total_counts = sbml_dfs_metabolism.get_source_total_counts(
        SBML_DFS.REACTIONS
    )

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


def test_ensure_source_total_counts():
    """
    Test that _ensure_source_total_counts properly validates and fixes source_total_counts structure.
    """
    # Test with a malformed Series
    malformed_series = pd.Series([10, 20], index=["path1", "path2"], name="wrong_name")

    # Test that this gets fixed by _ensure_source_total_counts
    fixed_series = source._ensure_source_total_counts(malformed_series)

    # Verify the Series is properly formatted
    assert fixed_series.name == CONTINGENCY_TABLE.TOTAL_COUNTS
    assert fixed_series.index.name == SOURCE_SPEC.PATHWAY_ID
    assert list(fixed_series.index) == ["path1", "path2"]
    assert list(fixed_series.values) == [10, 20]


def test_source_set_coverage_vertex_consistency(sbml_dfs_metabolism):
    """
    Test that source_set_coverage returns results where all vertices in the edgelist
    are also present in the vertices table.
    """
    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions)
    set_coverage = source.source_set_coverage(source_df)

    # Get the reaction IDs from the set coverage result
    reaction_ids_in_coverage = set(set_coverage.index.get_level_values(0).unique())

    # Get all reaction IDs from the reactions table
    all_reaction_ids = set(sbml_dfs_metabolism.reactions.index)

    # Verify that all reaction IDs in the coverage are present in the reactions table
    missing_reactions = reaction_ids_in_coverage - all_reaction_ids
    assert (
        len(missing_reactions) == 0
    ), f"Found reaction IDs in coverage that are not in reactions table: {missing_reactions}"

    # Also verify that the coverage contains reactions that exist in the table
    coverage_reactions_in_table = reaction_ids_in_coverage & all_reaction_ids
    assert (
        len(coverage_reactions_in_table) > 0
    ), "No reaction IDs from coverage found in reactions table"


def test_source_set_coverage_min_pw_size_filtering(sbml_dfs_metabolism):
    """
    Test that source_set_coverage properly filters pathways based on min_pw_size parameter.
    """
    source_df = source.unnest_sources(sbml_dfs_metabolism.reactions)

    # Test with different min_pw_size values
    for min_size in [1, 3, 5, 10]:
        set_coverage = source.source_set_coverage(source_df, min_pw_size=min_size)

        if len(set_coverage) > 0:
            # Check that each pathway in the result has at least min_size members
            for pathway_id in set_coverage[SOURCE_SPEC.PATHWAY_ID].unique():
                pathway_members = set_coverage[
                    set_coverage[SOURCE_SPEC.PATHWAY_ID] == pathway_id
                ]
                assert (
                    len(pathway_members) >= min_size
                ), f"Pathway {pathway_id} has {len(pathway_members)} members, less than min_pw_size={min_size}"
