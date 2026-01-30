from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from igraph import Graph, disjoint_union

from napistu.network import ig_utils, net_create
from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH_EDGES, UNIVERSE_GATES


@pytest.fixture
def multi_component_graph() -> Graph:
    """Creates a graph with multiple disconnected components of different sizes."""
    g1 = Graph.Ring(5)  # 5 vertices, 5 edges
    g2 = Graph.Tree(3, 2)  # 3 vertices, 2 edges
    g3 = Graph.Full(2)  # 2 vertices, 1 edge
    return disjoint_union([g1, g2, g3])


def test_validate_graph_attributes(sbml_dfs):

    napistu_graph = net_create.process_napistu_graph(
        sbml_dfs, directed=True, weighting_strategy="topology"
    )

    assert (
        ig_utils.validate_edge_attributes(
            napistu_graph,
            [NAPISTU_GRAPH_EDGES.WEIGHT, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM],
        )
        is None
    )
    assert ig_utils.validate_vertex_attributes(napistu_graph, "node_type") is None
    with pytest.raises(ValueError):
        ig_utils.validate_vertex_attributes(napistu_graph, "baz")


def test_filter_to_largest_subgraph(multi_component_graph):
    """Tests that the function returns only the single largest component."""
    largest = ig_utils.filter_to_largest_subgraph(multi_component_graph)
    assert isinstance(largest, Graph)
    assert largest.vcount() == 5
    assert largest.ecount() == 5


def test_filter_to_largest_subgraphs(multi_component_graph):
    """Tests that the function returns the top K largest components."""
    # Test getting the top 2
    top_2 = ig_utils.filter_to_largest_subgraphs(multi_component_graph, top_k=2)
    assert isinstance(top_2, list)
    assert len(top_2) == 2
    assert all(isinstance(g, Graph) for g in top_2)
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
    graph = Graph(5)
    graph.vs["attr1"] = [0, 1, 2, 0, 3]
    graph.vs["attr2"] = [1, 0, 1, 2, 0]
    graph.vs[IGRAPH_DEFS.NAME] = ["A", "B", "C", "D", "E"]

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
    graph_no_names = Graph(3)
    graph_no_names.vs["attr1"] = [1, 2, 3]

    # Graph with names
    graph = Graph(3)
    graph.vs["attr1"] = [1, 2, 3]
    graph.vs[IGRAPH_DEFS.NAME] = ["A", "B", "C"]

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
    graph = Graph(4)
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


def test_get_universe_vertex_names(simple_directed_graph):
    """Test _get_universe_vertex_names function."""

    # Test all vertices when None provided
    result = ig_utils._get_universe_vertex_names(simple_directed_graph, None)
    assert result == ["A", "B", "C", "D"]
    assert isinstance(result, list)

    # Test filtered vertices from list
    result = ig_utils._get_universe_vertex_names(simple_directed_graph, ["A", "C"])
    assert result == ["A", "C"]

    # Test filtered vertices from Series
    vertex_series = pd.Series(["B", "D"])
    result = ig_utils._get_universe_vertex_names(simple_directed_graph, vertex_series)
    assert result == ["B", "D"]
    assert isinstance(result, list)

    # Test all vertices explicitly included
    result = ig_utils._get_universe_vertex_names(
        simple_directed_graph, ["A", "B", "C", "D"]
    )
    assert result == ["A", "B", "C", "D"]

    # Test raises error for missing vertex
    with pytest.raises(ValueError, match="1 vertex name\\(s\\) not found in graph"):
        ig_utils._get_universe_vertex_names(simple_directed_graph, ["A", "E"])

    # Test empty list raises error
    with pytest.raises(
        ValueError, match="vertex_names must contain at least one vertex name"
    ):
        ig_utils._get_universe_vertex_names(simple_directed_graph, [])


def test_get_universe_edge_filters(simple_directed_graph):
    """Test _get_universe_edge_filters function."""

    # Test empty filters when no inputs
    result = ig_utils._get_universe_edge_filters(simple_directed_graph, None, False)
    assert result == []
    assert isinstance(result, list)

    # Test observed_only filter (returns all edges in the graph)
    result = ig_utils._get_universe_edge_filters(simple_directed_graph, None, True)
    assert len(result) == 1
    assert isinstance(result[0], pd.DataFrame)
    assert IGRAPH_DEFS.SOURCE in result[0].columns
    assert IGRAPH_DEFS.TARGET in result[0].columns
    # Should have all 3 edges in the graph
    assert len(result[0]) == 3
    expected_edges = {("A", "B"), ("B", "C"), ("C", "D")}
    actual_edges = {
        (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
        for _, row in result[0].iterrows()
    }
    assert actual_edges == expected_edges

    # Test edgelist filter
    edgelist = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "C"],
            IGRAPH_DEFS.TARGET: ["B", "D"],
        }
    )
    result = ig_utils._get_universe_edge_filters(simple_directed_graph, edgelist, False)
    assert len(result) == 1
    assert isinstance(result[0], pd.DataFrame)
    pd.testing.assert_frame_equal(
        result[0], edgelist[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]]
    )

    # Test both observed and edgelist
    result = ig_utils._get_universe_edge_filters(simple_directed_graph, edgelist, True)
    assert len(result) == 2
    assert isinstance(result[0], pd.DataFrame)  # observed edges
    assert isinstance(result[1], pd.DataFrame)  # edgelist

    # Test edgelist missing columns raises error
    bad_edgelist = pd.DataFrame({"from": ["A"], "to": ["B"]})
    with pytest.raises(
        ValueError, match="edgelist must have columns 'source' and 'target'"
    ):
        ig_utils._get_universe_edge_filters(simple_directed_graph, bad_edgelist, False)

    # Test observed_only returns all edges regardless of attributes
    g = Graph(directed=True)
    g.add_vertices(2)
    g.vs[IGRAPH_DEFS.NAME] = ["A", "B"]
    g.add_edges([(0, 1)])
    # No 'observed' attribute set, but all edges are returned
    result = ig_utils._get_universe_edge_filters(g, None, True)
    assert len(result) == 1
    assert len(result[0]) == 1  # All edges in graph are returned


def test_create_universe_edgelist(simple_undirected_graph):
    """Test _create_universe_edgelist function."""

    selected_names = ["X", "Y", "Z"]

    # Test complete directed graph when no filters
    result = ig_utils._create_universe_edgelist(
        [], UNIVERSE_GATES.AND, selected_names, is_directed=True
    )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 9  # 3x3 = 9 edges in complete directed graph
    assert IGRAPH_DEFS.SOURCE in result.columns
    assert IGRAPH_DEFS.TARGET in result.columns
    # Check all combinations exist
    expected_edges = {(src, tgt) for src in selected_names for tgt in selected_names}
    actual_edges = {
        (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
        for _, row in result.iterrows()
    }
    assert actual_edges == expected_edges

    # Test complete undirected graph when no filters
    result = ig_utils._create_universe_edgelist(
        [], UNIVERSE_GATES.AND, selected_names, is_directed=False
    )
    assert isinstance(result, pd.DataFrame)
    # Undirected: n*(n+1)/2 = 3*4/2 = 6 edges (includes self-loops in complete graph)
    assert len(result) == 6
    # Check that we have upper triangular + diagonal
    for _, row in result.iterrows():
        src_idx = selected_names.index(row[IGRAPH_DEFS.SOURCE])
        tgt_idx = selected_names.index(row[IGRAPH_DEFS.TARGET])
        assert src_idx <= tgt_idx  # Upper triangular or diagonal

    # Test single filter returns copy
    filter_df = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["C", "D"],
        }
    )
    result = ig_utils._create_universe_edgelist(
        [filter_df], UNIVERSE_GATES.AND, ["A", "B", "C", "D"], is_directed=True
    )
    pd.testing.assert_frame_equal(result, filter_df)
    # Verify it's a copy (modifying result shouldn't affect original)
    result.loc[0, IGRAPH_DEFS.SOURCE] = "X"
    assert filter_df.loc[0, IGRAPH_DEFS.SOURCE] == "A"

    # Test AND logic creates intersection
    filter1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "C"],
            IGRAPH_DEFS.TARGET: ["B", "C", "D"],
        }
    )
    filter2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
        }
    )
    result = ig_utils._create_universe_edgelist(
        [filter1, filter2],
        UNIVERSE_GATES.AND,
        ["A", "B", "C", "D"],
        is_directed=True,
    )
    assert len(result) == 2  # Only (A,B) and (B,C) in both
    expected_edges = {("A", "B"), ("B", "C")}
    actual_edges = {
        (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
        for _, row in result.iterrows()
    }
    assert actual_edges == expected_edges

    # Test OR logic creates union
    filter1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
        }
    )
    filter2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["C", "D"],
            IGRAPH_DEFS.TARGET: ["D", "A"],
        }
    )
    result = ig_utils._create_universe_edgelist(
        [filter1, filter2],
        UNIVERSE_GATES.OR,
        ["A", "B", "C", "D"],
        is_directed=True,
    )
    assert len(result) == 4  # All unique edges from both filters
    expected_edges = {("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")}
    actual_edges = {
        (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
        for _, row in result.iterrows()
    }
    assert actual_edges == expected_edges

    # Test OR logic removes duplicates
    filter1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
        }
    )
    filter2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
        }
    )
    result = ig_utils._create_universe_edgelist(
        [filter1, filter2],
        UNIVERSE_GATES.OR,
        ["A", "B", "C"],
        is_directed=True,
    )
    assert len(result) == 2  # Duplicates removed
    expected_edges = {("A", "B"), ("B", "C")}
    actual_edges = {
        (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
        for _, row in result.iterrows()
    }
    assert actual_edges == expected_edges

    # Test invalid filter logic raises error (needs multiple filters to trigger validation)
    filter1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A"],
            IGRAPH_DEFS.TARGET: ["B"],
        }
    )
    filter2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["B"],
            IGRAPH_DEFS.TARGET: ["C"],
        }
    )
    with pytest.raises(ValueError, match="Invalid edge_filter_logic"):
        ig_utils._create_universe_edgelist(
            [filter1, filter2], "invalid", ["A", "B", "C"], is_directed=True
        )

    # Test multiple filters with AND logic
    filter1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "C"],
            IGRAPH_DEFS.TARGET: ["B", "C", "D"],
        }
    )
    filter2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
        }
    )
    filter3 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A"],
            IGRAPH_DEFS.TARGET: ["B"],
        }
    )
    result = ig_utils._create_universe_edgelist(
        [filter1, filter2, filter3],
        UNIVERSE_GATES.AND,
        ["A", "B", "C", "D"],
        is_directed=True,
    )
    assert len(result) == 1  # Only (A,B) in all three
    assert result.loc[0, IGRAPH_DEFS.SOURCE] == "A"
    assert result.loc[0, IGRAPH_DEFS.TARGET] == "B"


def test_get_universe_degrees(simple_directed_graph, simple_undirected_graph):
    """Test _get_universe_degrees function."""
    out_deg, in_deg = ig_utils._get_universe_degrees(
        simple_directed_graph, directed=True
    )
    assert np.array_equal(out_deg, [1, 1, 1, 0]) and np.array_equal(
        in_deg, [0, 1, 1, 1]
    )
    out_deg, in_deg = ig_utils._get_universe_degrees(
        simple_undirected_graph, directed=False
    )
    assert np.array_equal(out_deg, in_deg)
