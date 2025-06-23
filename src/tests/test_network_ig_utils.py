from __future__ import annotations

import pytest

from napistu.network import ig_utils
from napistu.network import net_create


@pytest.fixture
def multi_component_graph() -> ig_utils.ig.Graph:
    """Creates a graph with multiple disconnected components of different sizes."""
    g1 = ig_utils.ig.Graph.Ring(5)  # 5 vertices, 5 edges
    g2 = ig_utils.ig.Graph.Tree(3, 2)  # 3 vertices, 2 edges
    g3 = ig_utils.ig.Graph.Full(2)  # 2 vertices, 1 edge
    return ig_utils.ig.disjoint_union([g1, g2, g3])


def test_validate_graph_attributes(sbml_dfs):

    napistu_graph = net_create.process_napistu_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    assert (
        ig_utils.validate_edge_attributes(
            napistu_graph, ["weights", "upstream_weights"]
        )
        is None
    )
    assert ig_utils.validate_vertex_attributes(napistu_graph, "node_type") is None
    with pytest.raises(ValueError):
        ig_utils.validate_vertex_attributes(napistu_graph, "baz")


def test_filter_to_largest_subgraph(multi_component_graph):
    """Tests that the function returns only the single largest component."""
    largest = ig_utils.filter_to_largest_subgraph(multi_component_graph)
    assert isinstance(largest, ig_utils.ig.Graph)
    assert largest.vcount() == 5
    assert largest.ecount() == 5


def test_filter_to_largest_subgraphs(multi_component_graph):
    """Tests that the function returns the top K largest components."""
    # Test getting the top 2
    top_2 = ig_utils.filter_to_largest_subgraphs(multi_component_graph, top_k=2)
    assert isinstance(top_2, list)
    assert len(top_2) == 2
    assert all(isinstance(g, ig_utils.ig.Graph) for g in top_2)
    assert [g.vcount() for g in top_2] == [5, 3]

    # Test getting more than the total number of components
    top_5 = ig_utils.filter_to_largest_subgraphs(multi_component_graph, top_k=5)
    assert len(top_5) == 3
    assert [g.vcount() for g in top_5] == [5, 3, 2]

    # Test invalid top_k
    with pytest.raises(ValueError):
        ig_utils.filter_to_largest_subgraphs(multi_component_graph, top_k=0)
