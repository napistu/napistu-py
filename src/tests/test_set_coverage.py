from __future__ import annotations

from napistu import source
from napistu.network import net_utils


def test_get_minimal_source_edges(sbml_dfs_metabolism):
    vertices = sbml_dfs_metabolism.reactions.reset_index().rename(
        columns={"r_id": "node"}
    )

    minimal_source_edges = net_utils.get_minimal_sources_edges(
        vertices, sbml_dfs_metabolism
    )
    # print(minimal_source_edges.shape)
    assert minimal_source_edges.shape == (87, 3)


def test_greedy_set_coverge_of_sources(sbml_dfs_metabolism):
    table_schema = sbml_dfs_metabolism.schema["reactions"]

    source_df = source.unnest_sources(
        sbml_dfs_metabolism.reactions, source_var="r_Source"
    )
    # print(source_df.shape)
    assert source_df.shape == (111, 7)

    set_coverage = source.greedy_set_coverge_of_sources(source_df, table_schema)
    # print(set_coverage.shape)
    assert set_coverage.shape == (87, 6)


################################################
# __main__
################################################

if __name__ == "__main__":
    import os
    from napistu import indices
    from napistu import consensus

    test_path = os.path.abspath(os.path.join(__file__, os.pardir))
    test_data = os.path.join(test_path, "test_data")

    pw_index = indices.PWIndex(os.path.join(test_data, "pw_index_metabolism.tsv"))
    sbml_dfs_dict = consensus.construct_sbml_dfs_dict(pw_index)
    sbml_dfs_metabolism = consensus.construct_consensus_model(sbml_dfs_dict, pw_index)

    test_get_minimal_source_edges(sbml_dfs_metabolism)
    test_greedy_set_coverge_of_sources(sbml_dfs_metabolism)
