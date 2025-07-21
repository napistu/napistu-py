from __future__ import annotations

import igraph as ig
import numpy as np
import pytest

from napistu.network import ig_utils
from napistu.network import net_create
from napistu.network.constants import NAPISTU_GRAPH_EDGES


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
            napistu_graph,
            [NAPISTU_GRAPH_EDGES.WEIGHT, NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT],
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


def test_mask_functions_valid_inputs():
    """Test mask functions with various valid input formats."""
    # Create real graph with attributes
    graph = ig.Graph(5)
    graph.vs["attr1"] = [0, 1, 2, 0, 3]
    graph.vs["attr2"] = [1, 0, 1, 2, 0]
    graph.vs["name"] = ["A", "B", "C", "D", "E"]

    attributes = ["attr1", "attr2"]

    # Test 1: None input
    specs = ig_utils._parse_mask_input(None, attributes)
    assert specs == {"attr1": None, "attr2": None}

    masks = ig_utils._get_attribute_masks(graph, specs)
    assert np.array_equal(masks["attr1"], np.ones(5, dtype=bool))
    assert np.array_equal(masks["attr2"], np.ones(5, dtype=bool))

    # Test 2: "attr" keyword
    specs = ig_utils._parse_mask_input("attr", attributes)
    assert specs == {"attr1": "attr1", "attr2": "attr2"}

    masks = ig_utils._get_attribute_masks(graph, specs)
    assert np.array_equal(masks["attr1"], np.array([False, True, True, False, True]))
    assert np.array_equal(masks["attr2"], np.array([True, False, True, True, False]))

    # Test 3: Single attribute name
    specs = ig_utils._parse_mask_input("attr1", attributes)
    assert specs == {"attr1": "attr1", "attr2": "attr1"}

    # Test 4: Boolean array
    bool_mask = np.array([True, False, True, False, False])
    specs = ig_utils._parse_mask_input(bool_mask, attributes)
    masks = ig_utils._get_attribute_masks(graph, specs)
    assert np.array_equal(masks["attr1"], bool_mask)
    assert np.array_equal(masks["attr2"], bool_mask)

    # Test 5: Node indices list
    indices = [0, 2, 4]
    specs = ig_utils._parse_mask_input(indices, attributes)
    masks = ig_utils._get_attribute_masks(graph, specs)
    expected = np.array([True, False, True, False, True])
    assert np.array_equal(masks["attr1"], expected)

    # Test 6: Node names list
    names = ["A", "C", "E"]
    specs = ig_utils._parse_mask_input(names, attributes)
    masks = ig_utils._get_attribute_masks(graph, specs)
    assert np.array_equal(masks["attr1"], expected)

    # Test 7: Dictionary input
    mask_dict = {"attr1": "attr1", "attr2": None}
    specs = ig_utils._parse_mask_input(mask_dict, attributes)
    assert specs == mask_dict

    masks = ig_utils._get_attribute_masks(graph, specs)
    assert np.array_equal(masks["attr1"], np.array([False, True, True, False, True]))
    assert np.array_equal(masks["attr2"], np.ones(5, dtype=bool))


def test_mask_functions_error_cases():
    """Test mask functions with invalid inputs that should raise errors."""
    # Graph without name attribute
    graph_no_names = ig.Graph(3)
    graph_no_names.vs["attr1"] = [1, 2, 3]

    # Graph with names
    graph = ig.Graph(3)
    graph.vs["attr1"] = [1, 2, 3]
    graph.vs["name"] = ["A", "B", "C"]

    attributes = ["attr1", "attr2"]

    # Test 1: Invalid mask type
    with pytest.raises(ValueError, match="Invalid mask input type"):
        ig_utils._parse_mask_input(123, attributes)

    # Test 2: Missing attribute in dictionary
    incomplete_dict = {"attr1": None}  # Missing 'attr2'
    with pytest.raises(
        ValueError, match="Attribute 'attr2' not found in mask dictionary"
    ):
        ig_utils._parse_mask_input(incomplete_dict, attributes)

    # Test 3: String mask for graph without names
    specs = {"attr1": ["A", "B"]}
    with pytest.raises(
        ValueError, match="Graph has no 'name' attribute for string mask"
    ):
        ig_utils._get_attribute_masks(graph_no_names, specs)

    # Test 4: Invalid mask specification type in _get_attribute_masks
    specs = {"attr1": 123}  # Invalid type
    with pytest.raises(
        ValueError, match="Invalid mask specification for attribute 'attr1'"
    ):
        ig_utils._get_attribute_masks(graph, specs)


def test_ensure_nonnegative_vertex_attribute():
    """Test _ensure_valid_attribute with various valid and invalid inputs."""
    # Create test graph
    graph = ig.Graph(4)
    graph.vs["good_attr"] = [1.0, 2.0, 0.0, 3.0]
    graph.vs["zero_attr"] = [0.0, 0.0, 0.0, 0.0]
    graph.vs["negative_attr"] = [1.0, -1.0, 2.0, 0.0]
    graph.vs["mixed_attr"] = [1.0, None, 2.0, 0.0]  # Some None values

    # Test 1: Valid attribute
    result = ig_utils._ensure_valid_attribute(graph, "good_attr")
    expected = np.array([1.0, 2.0, 0.0, 3.0])
    assert np.array_equal(result, expected)

    # Test 2: Attribute with None values (should be replaced with 0)
    result = ig_utils._ensure_valid_attribute(graph, "mixed_attr")
    expected = np.array([1.0, 0.0, 2.0, 0.0])
    assert np.array_equal(result, expected)

    # Test 3: All zero values
    with pytest.raises(ValueError, match="zero for all vertices"):
        ig_utils._ensure_valid_attribute(graph, "zero_attr")

    # Test 4: Negative values
    with pytest.raises(ValueError, match="contains negative values"):
        ig_utils._ensure_valid_attribute(graph, "negative_attr")

    # Test 5: Missing attribute
    with pytest.raises(ValueError, match="missing for all vertices"):
        ig_utils._ensure_valid_attribute(graph, "nonexistent_attr")

    # Test 6: Non-finite values (NaN and inf)
    graph.vs["nan_attr"] = [1.0, np.nan, 2.0, 0.0]
    graph.vs["inf_attr"] = [1.0, np.inf, 2.0, 0.0]
    with pytest.raises(ValueError, match="non-finite values"):
        ig_utils._ensure_valid_attribute(graph, "nan_attr")
    with pytest.raises(ValueError, match="non-finite values"):
        ig_utils._ensure_valid_attribute(graph, "inf_attr")
