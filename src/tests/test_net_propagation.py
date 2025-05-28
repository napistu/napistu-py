import pytest
import igraph as ig
import numpy as np
from napistu.network.net_propagation import personalized_pagerank_by_attribute


def test_personalized_pagerank_by_attribute_basic():
    g = ig.Graph.Full(3)
    g.vs["name"] = ["A", "B", "C"]
    g.vs["score"] = [1, 0, 2]
    df = personalized_pagerank_by_attribute(g, "score")
    assert set(df.columns) == {
        "name",
        "pagerank_by_attribute",
        "pagerank_uniform",
        "score",
    }
    assert np.isclose(df["score"].sum(), 3)
    assert np.isclose(df["pagerank_by_attribute"].sum(), 1)
    assert np.isclose(df["pagerank_uniform"].sum(), 1)
    # Uniform should only include A and C
    assert df.loc[df["name"] == "B", "pagerank_uniform"].values[0] > 0


def test_personalized_pagerank_by_attribute_no_uniform():
    g = ig.Graph.Full(3)
    g.vs["score"] = [1, 0, 2]
    df = personalized_pagerank_by_attribute(g, "score", calculate_uniform_dist=False)
    assert "pagerank_uniform" not in df.columns
    assert np.isclose(df["pagerank_by_attribute"].sum(), 1)


def test_personalized_pagerank_by_attribute_missing_and_negative():
    g = ig.Graph.Full(3)
    g.vs["score"] = [1, None, 2]
    # None should be treated as 0
    df = personalized_pagerank_by_attribute(g, "score")
    assert np.isclose(df["score"].sum(), 3)
    # Negative values should raise
    g.vs["score"] = [1, -1, 2]
    with pytest.raises(ValueError):
        personalized_pagerank_by_attribute(g, "score")


def test_personalized_pagerank_by_attribute_additional_args_directed():
    # create an asymmetric directed graph to test whether additional_propagation_args is respected
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2)])
    g.vs["score"] = [1, 0, 2]
    # Run with directed=False, which should treat the graph as undirected
    df_directed = personalized_pagerank_by_attribute(
        g, "score", additional_propagation_args={"directed": True}
    )
    df_undirected = personalized_pagerank_by_attribute(
        g, "score", additional_propagation_args={"directed": False}
    )
    # The results should differ for directed vs undirected
    assert not np.allclose(
        df_directed["pagerank_by_attribute"], df_undirected["pagerank_by_attribute"]
    )
    # Uniform should also be affected
    assert not np.allclose(
        df_directed["pagerank_uniform"], df_undirected["pagerank_uniform"]
    )


def test_personalized_pagerank_by_attribute_additional_args_invalid():
    g = ig.Graph.Full(3)
    g.vs["score"] = [1, 0, 2]
    # Passing an invalid argument should raise ValueError
    with pytest.raises(ValueError):
        personalized_pagerank_by_attribute(
            g, "score", additional_propagation_args={"not_a_real_arg": 123}
        )


def test_personalized_pagerank_by_attribute_all_missing():
    g = ig.Graph.Full(3)
    # No 'score' attribute at all
    with pytest.raises(ValueError, match="missing for all vertices"):
        personalized_pagerank_by_attribute(g, "score")


def test_personalized_pagerank_by_attribute_all_zero():
    g = ig.Graph.Full(3)
    g.vs["score"] = [0, 0, 0]
    with pytest.raises(ValueError, match="zero for all vertices"):
        personalized_pagerank_by_attribute(g, "score")
