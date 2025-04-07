from __future__ import annotations

import pytest

import igraph as ig
import pandas as pd
from cpr.network import net_utils
from cpr.network import net_create


def test_safe_fill():
    safe_fill_test = ["a_very_long stringggg", ""]
    assert [net_utils.safe_fill(x) for x in safe_fill_test] == [
        "a_very_long\nstringggg",
        "",
    ]


def test_cpr_graph_to_pandas_dfs():
    graph_data = [
        (0, 1),
        (0, 2),
        (2, 3),
        (3, 4),
        (4, 2),
        (2, 5),
        (5, 0),
        (6, 3),
        (5, 6),
    ]

    g = ig.Graph(graph_data, directed=True)
    vs, es = net_utils.cpr_graph_to_pandas_dfs(g)

    assert all(vs["index"] == list(range(0, 7)))
    assert (
        pd.DataFrame(graph_data)
        .rename({0: "source", 1: "target"}, axis=1)
        .sort_values(["source", "target"])
        .equals(es.sort_values(["source", "target"]))
    )


def test_validate_graph_attributes(sbml_dfs):

    cpr_graph = net_create.process_cpr_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    assert (
        net_utils._validate_edge_attributes(cpr_graph, ["weights", "upstream_weights"])
        is None
    )
    assert net_utils._validate_vertex_attributes(cpr_graph, "node_type") is None
    with pytest.raises(ValueError):
        net_utils._validate_vertex_attributes(cpr_graph, "baz")


################################################
# __main__
################################################

if __name__ == "__main__":
    test_safe_fill()
    test_cpr_graph_to_pandas_dfs()
    test_validate_graph_attributes()
