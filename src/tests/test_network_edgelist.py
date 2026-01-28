"""Tests for Edgelist class."""

from __future__ import annotations

import pandas as pd
import pytest

from napistu.network.constants import IGRAPH_DEFS, NAPISTU_GRAPH_EDGES
from napistu.network.edgelist import Edgelist


def test_validate_subset(simple_directed_graph, simple_undirected_graph):
    """Test Edgelist.validate_subset method."""
    # Test valid edgelist passes
    valid_edgelist = pd.DataFrame(
        {IGRAPH_DEFS.SOURCE: ["A", "B"], IGRAPH_DEFS.TARGET: ["B", "C"]}
    )
    el = Edgelist(valid_edgelist)
    el.validate_subset(simple_directed_graph)

    # Test missing required columns raises error
    bad_edgelist = pd.DataFrame({"col1": ["A"], "col2": ["B"]})
    with pytest.raises(ValueError, match="must have either"):
        Edgelist(bad_edgelist)

    # Test invalid vertices raises error
    invalid_vertex_edgelist = pd.DataFrame(
        {IGRAPH_DEFS.SOURCE: ["A", "E"], IGRAPH_DEFS.TARGET: ["B", "C"]}
    )
    el_invalid = Edgelist(invalid_vertex_edgelist)
    with pytest.raises(ValueError, match="vertex\\(s\\) in edgelist not in universe"):
        el_invalid.validate_subset(simple_directed_graph, graph_name="universe")

    # Test invalid edges raises error
    invalid_edge_edgelist = pd.DataFrame(
        {IGRAPH_DEFS.SOURCE: ["A", "C"], IGRAPH_DEFS.TARGET: ["B", "A"]}
    )
    el_invalid_edge = Edgelist(invalid_edge_edgelist)
    with pytest.raises(ValueError, match="edge\\(s\\) in edgelist not in universe"):
        el_invalid_edge.validate_subset(simple_directed_graph, graph_name="universe")

    # Test undirected graph accepts reverse edges
    undirected_edgelist = pd.DataFrame(
        {IGRAPH_DEFS.SOURCE: ["X", "Y"], IGRAPH_DEFS.TARGET: ["Y", "X"]}
    )
    simple_undirected_graph.add_edges([(0, 1)])  # Add edge X-Y
    el_undirected = Edgelist(undirected_edgelist)
    el_undirected.validate_subset(simple_undirected_graph)


def test_standard_merge_by():
    """Test standard_merge_by property."""
    # Test source/target columns return NAME
    df_source_target = pd.DataFrame(
        {IGRAPH_DEFS.SOURCE: ["A", "B"], IGRAPH_DEFS.TARGET: ["B", "C"]}
    )
    el = Edgelist(df_source_target)
    assert el.standard_merge_by == IGRAPH_DEFS.NAME

    # Test from/to columns return INDEX
    df_from_to = pd.DataFrame(
        {NAPISTU_GRAPH_EDGES.FROM: [0, 1], NAPISTU_GRAPH_EDGES.TO: [1, 2]}
    )
    el_from_to = Edgelist(df_from_to)
    assert el_from_to.standard_merge_by == IGRAPH_DEFS.INDEX


def test_merge_edgelists():
    """Test merge_edgelists method."""
    # Test merging source/target edgelists
    df1 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
            "weight": [1.0, 2.0],
        }
    )
    df2 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["B", "C"],
            "score": [0.5, 0.8],
        }
    )
    el1 = Edgelist(df1)
    el2 = Edgelist(df2)
    merged = el1.merge_edgelists(el2)
    assert len(merged) == 2
    assert "weight" in merged.df.columns
    assert "score" in merged.df.columns
    assert merged.df.loc[0, "weight"] == 1.0
    assert merged.df.loc[0, "score"] == 0.5

    # Test merging from/to edgelists
    df3 = pd.DataFrame(
        {
            NAPISTU_GRAPH_EDGES.FROM: [0, 1],
            NAPISTU_GRAPH_EDGES.TO: [1, 2],
            "weight": [1.0, 2.0],
        }
    )
    df4 = pd.DataFrame(
        {
            NAPISTU_GRAPH_EDGES.FROM: [0, 1],
            NAPISTU_GRAPH_EDGES.TO: [1, 2],
            "score": [0.5, 0.8],
        }
    )
    el3 = Edgelist(df3)
    el4 = Edgelist(df4)
    merged_idx = el3.merge_edgelists(el4)
    assert len(merged_idx) == 2
    assert "weight" in merged_idx.df.columns
    assert "score" in merged_idx.df.columns

    # Test merging mismatched conventions raises error
    with pytest.raises(ValueError, match="different merge_by conventions"):
        el1.merge_edgelists(el3)

    # Test merging with DataFrame
    merged_df = el1.merge_edgelists(df2)
    assert len(merged_df) == 2
    assert "weight" in merged_df.df.columns
    assert "score" in merged_df.df.columns

    # Test inner merge (default)
    df5 = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A"],
            IGRAPH_DEFS.TARGET: ["B"],
            "value": [10],
        }
    )
    el5 = Edgelist(df5)
    merged_inner = el1.merge_edgelists(el5, how="inner")
    assert len(merged_inner) == 1  # Only (A, B) in both

    # Test left merge
    merged_left = el1.merge_edgelists(el5, how="left")
    assert len(merged_left) == 2  # All from el1
