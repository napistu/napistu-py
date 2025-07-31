import logging
import pytest

import igraph as ig
import pandas as pd

from napistu.network.ng_core import NapistuGraph

logger = logging.getLogger(__name__)


@pytest.fixture
def test_graph():
    """Create a simple test graph."""
    g = ig.Graph()
    g.add_vertices(3, attributes={"name": ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])
    g.es["r_id"] = ["R1", "R2"]
    return NapistuGraph.from_igraph(g)


def test_remove_isolated_vertices():
    """Test removing isolated vertices from a graph."""

    g = ig.Graph()
    g.add_vertices(5, attributes={"name": ["A", "B", "C", "D", "E"]})
    g.add_edges([(0, 1), (2, 3)])  # A-B, C-D connected; E isolated

    napstu_graph = NapistuGraph.from_igraph(g)
    napstu_graph.remove_isolated_vertices()
    assert napstu_graph.vcount() == 4  # Should have 4 vertices after removing E
    assert "E" not in [v["name"] for v in napstu_graph.vs]  # E should be gone


def test_to_pandas_dfs():
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

    g = NapistuGraph.from_igraph(ig.Graph(graph_data, directed=True))
    vs, es = g.to_pandas_dfs()

    assert all(vs["index"] == list(range(0, 7)))
    assert (
        pd.DataFrame(graph_data)
        .rename({0: "source", 1: "target"}, axis=1)
        .sort_values(["source", "target"])
        .equals(es.sort_values(["source", "target"]))
    )


def test_set_and_get_graph_attrs(test_graph):
    """Test setting and getting graph attributes."""
    attrs = {
        "reactions": {
            "string_wt": {"table": "string", "variable": "score", "trans": "identity"}
        },
        "species": {
            "expression": {"table": "rnaseq", "variable": "fc", "trans": "identity"}
        },
    }

    # Set attributes
    test_graph.set_graph_attrs(attrs)

    # Check that attributes were stored in metadata
    stored_reactions = test_graph.get_metadata("reaction_attrs")
    stored_species = test_graph.get_metadata("species_attrs")

    assert (
        stored_reactions == attrs["reactions"]
    ), f"Expected {attrs['reactions']}, got {stored_reactions}"
    assert (
        stored_species == attrs["species"]
    ), f"Expected {attrs['species']}, got {stored_species}"

    # Get attributes through helper method
    reactions_result = test_graph._get_entity_attrs("reactions")
    species_result = test_graph._get_entity_attrs("species")

    assert (
        reactions_result == attrs["reactions"]
    ), f"Expected {attrs['reactions']}, got {reactions_result}"
    assert (
        species_result == attrs["species"]
    ), f"Expected {attrs['species']}, got {species_result}"

    # Test that method raises ValueError for unknown entity types
    with pytest.raises(ValueError, match="Unknown entity_type: 'nonexistent'"):
        test_graph._get_entity_attrs("nonexistent")

    # Test that method returns None for empty attributes
    test_graph.set_metadata(reaction_attrs={})
    assert test_graph._get_entity_attrs("reactions") is None


def test_compare_and_merge_attrs(test_graph):
    """Test the _compare_and_merge_attrs method directly."""
    new_attrs = {
        "string_wt": {"table": "string", "variable": "score", "trans": "identity"}
    }

    # Test fresh mode
    result = test_graph._compare_and_merge_attrs(
        new_attrs, "reaction_attrs", mode="fresh"
    )
    assert result == new_attrs

    # Test extend mode with no existing attrs
    result = test_graph._compare_and_merge_attrs(
        new_attrs, "reaction_attrs", mode="extend"
    )
    assert result == new_attrs

    # Test extend mode with existing attrs
    existing_attrs = {
        "existing": {"table": "test", "variable": "val", "trans": "identity"}
    }
    test_graph.set_metadata(reaction_attrs=existing_attrs)

    result = test_graph._compare_and_merge_attrs(
        new_attrs, "reaction_attrs", mode="extend"
    )
    expected = {**existing_attrs, **new_attrs}
    assert result == expected


def test_graph_attrs_extend_and_overwrite_protection(test_graph):
    """Test extend mode and overwrite protection."""
    # Set initial attributes
    initial = {
        "reactions": {
            "attr1": {"table": "test", "variable": "val", "trans": "identity"}
        }
    }
    test_graph.set_graph_attrs(initial)

    # Fresh mode should fail with existing data
    with pytest.raises(ValueError, match="Existing reaction_attrs found"):
        test_graph.set_graph_attrs({"reactions": {"attr2": {}}})

    # Extend mode should work
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr2": {"table": "new", "variable": "val2", "trans": "identity"}
            }
        },
        mode="extend",
    )
    reaction_attrs = test_graph.get_metadata("reaction_attrs")
    assert "attr1" in reaction_attrs and "attr2" in reaction_attrs

    # Extend with overlap should fail
    with pytest.raises(ValueError, match="Overlapping keys found"):
        test_graph.set_graph_attrs(
            {
                "reactions": {
                    "attr1": {
                        "table": "conflict",
                        "variable": "val",
                        "trans": "identity",
                    }
                }
            },
            mode="extend",
        )
