"""Tests for igraph utility functions."""

import pandas as pd

from napistu import utils


def test_find_weakly_connected_subgraphs():
    DEGEN_EDGELIST_DF_2 = pd.DataFrame(
        {
            "ind": ["a", "a", "b", "b", "c", "d"],
            "ont": ["X", "X", "X", "Y", "Y", "Y"],
            "val": ["A", "B", "C", "D", "D", "E"],
        }
    ).set_index("ind")

    edgelist_df = utils.format_identifiers_as_edgelist(
        DEGEN_EDGELIST_DF_2, ["ont", "val"]
    )
    edgelist = edgelist_df[["ind", "id"]]

    connected_indices = utils.find_weakly_connected_subgraphs(edgelist[["ind", "id"]])
    assert all(connected_indices["cluster"] == [0, 1, 1, 2])
