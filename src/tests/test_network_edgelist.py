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


def test_has_and_remove_duplicated_edges():
    """Test has_duplicated_edges property and remove_duplicated_edges method."""
    # Test edgelist with duplicates
    df_with_duplicates = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "A", "C"],
            IGRAPH_DEFS.TARGET: ["B", "C", "B", "D"],
            "weight": [1.0, 2.0, 1.5, 3.0],
        }
    )
    el = Edgelist(df_with_duplicates)

    # Test has_duplicated_edges property
    assert el.has_duplicated_edges

    # Test remove_duplicated_edges with keep="first"
    el_cleaned_first = el.remove_duplicated_edges(keep="first")
    assert len(el_cleaned_first) == 3  # A->B, B->C, C->D (first A->B kept)
    assert el_cleaned_first.df.loc[0, "weight"] == 1.0  # First A->B kept

    # Test remove_duplicated_edges with keep="last"
    el_cleaned_last = el.remove_duplicated_edges(keep="last")
    assert len(el_cleaned_last) == 3  # A->B, B->C, C->D (last A->B kept)
    # Find the A->B edge (last one should have weight 1.5)
    a_to_b = el_cleaned_last.df[
        (el_cleaned_last.df[el_cleaned_last.source_col] == "A")
        & (el_cleaned_last.df[el_cleaned_last.target_col] == "B")
    ]
    assert len(a_to_b) == 1
    assert a_to_b.iloc[0]["weight"] == 1.5  # Last A->B kept

    # Test inplace=True
    el_copy = Edgelist(df_with_duplicates)
    result = el_copy.remove_duplicated_edges(keep="first", inplace=True)
    assert result is None
    assert len(el_copy) == 3

    # Test edgelist without duplicates
    df_no_duplicates = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "C"],
            IGRAPH_DEFS.TARGET: ["B", "C", "D"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    el_no_dup = Edgelist(df_no_duplicates)
    assert not el_no_dup.has_duplicated_edges
    el_no_dup_cleaned = el_no_dup.remove_duplicated_edges()
    assert len(el_no_dup_cleaned) == 3  # No change

    # Test with from/to columns
    df_from_to = pd.DataFrame(
        {
            NAPISTU_GRAPH_EDGES.FROM: [0, 1, 0],
            NAPISTU_GRAPH_EDGES.TO: [1, 2, 1],
            "weight": [1.0, 2.0, 1.5],
        }
    )
    el_from_to = Edgelist(df_from_to)
    assert el_from_to.has_duplicated_edges
    el_from_to_cleaned = el_from_to.remove_duplicated_edges(keep="first")
    assert len(el_from_to_cleaned) == 2


def test_has_and_remove_reciprocal_edges():
    """Test has_reciprocal_edges property and remove_reciprocal_edges method."""
    # Test edgelist with reciprocal edges
    df_with_reciprocal = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "C"],
            IGRAPH_DEFS.TARGET: ["B", "A", "D"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    el = Edgelist(df_with_reciprocal)

    # Test has_reciprocal_edges property
    assert el.has_reciprocal_edges

    # Test remove_reciprocal_edges with keep="first"
    el_cleaned_first = el.remove_reciprocal_edges(keep="first")
    assert len(el_cleaned_first) == 2  # One of A->B or B->A removed, C->D kept
    # Check that we have either A->B or B->A, but not both
    edges = set(
        zip(
            el_cleaned_first.df[el_cleaned_first.source_col],
            el_cleaned_first.df[el_cleaned_first.target_col],
        )
    )
    assert ("A", "B") in edges or ("B", "A") in edges
    assert not (("A", "B") in edges and ("B", "A") in edges)
    assert ("C", "D") in edges

    # Test remove_reciprocal_edges with keep="lexicographic"
    el_cleaned_lex = el.remove_reciprocal_edges(keep="lexicographic")
    assert len(el_cleaned_lex) == 2
    # Lexicographic should keep A->B (A < B) and remove B->A
    edges_lex = set(
        zip(
            el_cleaned_lex.df[el_cleaned_lex.source_col],
            el_cleaned_lex.df[el_cleaned_lex.target_col],
        )
    )
    assert ("A", "B") in edges_lex
    assert ("B", "A") not in edges_lex
    assert ("C", "D") in edges_lex

    # Test inplace=True
    el_copy = Edgelist(df_with_reciprocal)
    result = el_copy.remove_reciprocal_edges(keep="first", inplace=True)
    assert result is None
    assert len(el_copy) == 2

    # Test edgelist without reciprocal edges
    df_no_reciprocal = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B", "C"],
            IGRAPH_DEFS.TARGET: ["B", "C", "D"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    el_no_rec = Edgelist(df_no_reciprocal)
    assert not el_no_rec.has_reciprocal_edges
    el_no_rec_cleaned = el_no_rec.remove_reciprocal_edges()
    assert len(el_no_rec_cleaned) == 3  # No change

    # Test with self-loops (A->A) - should not count as reciprocal
    df_with_self_loops = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "A", "B"],
            IGRAPH_DEFS.TARGET: ["A", "B", "A"],
            "weight": [1.0, 2.0, 3.0],
        }
    )
    el_self = Edgelist(df_with_self_loops)
    assert el_self.has_reciprocal_edges  # A->B and B->A are reciprocal
    el_self_cleaned = el_self.remove_reciprocal_edges(keep="lexicographic")
    # Should keep A->B (A < B), remove B->A, and keep A->A (self-loop)
    assert len(el_self_cleaned) == 2
    edges_self = set(
        zip(
            el_self_cleaned.df[el_self_cleaned.source_col],
            el_self_cleaned.df[el_self_cleaned.target_col],
        )
    )
    assert ("A", "A") in edges_self  # Self-loop kept
    assert ("A", "B") in edges_self  # Lexicographic keeps A->B
    assert ("B", "A") not in edges_self  # B->A removed

    # Test edgelist with only self-loops - no reciprocal edges
    df_only_self = pd.DataFrame(
        {
            IGRAPH_DEFS.SOURCE: ["A", "B"],
            IGRAPH_DEFS.TARGET: ["A", "B"],
            "weight": [1.0, 2.0],
        }
    )
    el_only_self = Edgelist(df_only_self)
    assert not el_only_self.has_reciprocal_edges
