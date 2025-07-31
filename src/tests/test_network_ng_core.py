import logging
import pytest

import igraph as ig
import pandas as pd

from napistu.network.ng_core import NapistuGraph
from napistu.network.constants import NAPISTU_GRAPH_EDGES

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


def test_add_edge_data_basic_functionality(test_graph, minimal_valid_sbml_dfs):
    """Test basic add_edge_data functionality with mock reaction data."""
    # Update the test graph to have the correct r_ids that match the SBML data
    test_graph.es["r_id"] = [
        "R00001",
        "R00001",
    ]  # Both edges should map to the same reaction

    # Create mock reaction data for the test reaction
    mock_df = pd.DataFrame(
        {"score_col": [100], "weight_col": [1.5]}, index=["R00001"]
    )  # Use the reaction ID from minimal_valid_sbml_dfs

    # Add mock data to sbml_dfs
    minimal_valid_sbml_dfs.reactions_data["mock_table"] = mock_df

    # Set graph attributes
    reaction_attrs = {
        "score_col": {
            "table": "mock_table",
            "variable": "score_col",
            "trans": "identity",
        },
        "weight_col": {
            "table": "mock_table",
            "variable": "weight_col",
            "trans": "identity",
        },
    }
    test_graph.set_graph_attrs({"reactions": reaction_attrs})

    # Add edge data
    test_graph.add_edge_data(minimal_valid_sbml_dfs)

    # Check that attributes were added
    assert "score_col" in test_graph.es.attributes()
    assert "weight_col" in test_graph.es.attributes()
    # Note: test_graph has 2 edges but only 1 reaction, so values will be filled appropriately
    edge_scores = test_graph.es["score_col"]
    edge_weights = test_graph.es["weight_col"]
    assert any(
        score == 100 for score in edge_scores
    )  # At least one edge should have the value
    assert any(weight == 1.5 for weight in edge_weights)


def test_add_edge_data_mode_and_overwrite(test_graph, minimal_valid_sbml_dfs):
    """Test mode and overwrite behavior for add_edge_data."""
    # Update the test graph to have the correct r_ids that match the SBML data
    test_graph.es["r_id"] = [
        "R00001",
        "R00001",
    ]  # Both edges should map to the same reaction

    # Add initial mock data
    minimal_valid_sbml_dfs.reactions_data["table1"] = pd.DataFrame(
        {"attr1": [10]}, index=["R00001"]
    )
    minimal_valid_sbml_dfs.reactions_data["table2"] = pd.DataFrame(
        {"attr1": [30], "attr2": [50]}, index=["R00001"]
    )

    # Set initial attributes and add
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr1": {"table": "table1", "variable": "attr1", "trans": "identity"}
            }
        }
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs)
    initial_attr1 = test_graph.es["attr1"]

    # Fresh mode should fail without overwrite when setting graph attributes
    with pytest.raises(ValueError, match="Existing reaction_attrs found"):
        test_graph.set_graph_attrs(
            {
                "reactions": {
                    "attr1": {
                        "table": "table2",
                        "variable": "attr1",
                        "trans": "identity",
                    }
                }
            }
        )

    # Fresh mode with overwrite should work for setting graph attributes
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr1": {"table": "table2", "variable": "attr1", "trans": "identity"}
            }
        },
        overwrite=True,
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs, mode="fresh", overwrite=True)
    updated_attr1 = test_graph.es["attr1"]
    assert updated_attr1 != initial_attr1  # Values should be different

    # Extend mode should add new attribute - clear reaction attributes first, then add only attr2
    test_graph.set_metadata(reaction_attrs={})  # Clear existing reaction attributes
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr2": {"table": "table2", "variable": "attr2", "trans": "identity"}
            }
        }
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs, mode="extend")
    assert "attr2" in test_graph.es.attributes()


def test_transform_edges_basic_functionality(test_graph, minimal_valid_sbml_dfs):
    """Test basic edge transformation functionality."""
    # Add mock reaction data with values that will be transformed
    mock_df = pd.DataFrame(
        {"raw_scores": [100, 200]}, index=["R00001", "R00002"]
    )  # Add second reaction for more edges
    minimal_valid_sbml_dfs.reactions_data["test_table"] = mock_df

    # Set reaction attrs with string_inv transformation (1 / (x / 1000))
    reaction_attrs = {
        "raw_scores": {
            "table": "test_table",
            "variable": "raw_scores",
            "trans": "string_inv",
        }
    }
    test_graph.set_graph_attrs({"reactions": reaction_attrs})

    # Add edge data first
    test_graph.add_edge_data(minimal_valid_sbml_dfs)
    original_values = test_graph.es["raw_scores"][:]

    # Transform edges
    test_graph.transform_edges(keep_raw_attributes=True)

    # Check transformation was applied (string_inv: 1/(x/1000))
    transformed_values = test_graph.es["raw_scores"]
    assert transformed_values != original_values

    # Check metadata was updated
    assert (
        "raw_scores" in test_graph.get_metadata("transformations_applied")["reactions"]
    )
    assert (
        test_graph.get_metadata("transformations_applied")["reactions"]["raw_scores"]
        == "string_inv"
    )

    # Check raw attributes were stored
    assert "raw_scores" in test_graph.get_metadata("raw_attributes")["reactions"]


def test_transform_edges_retransformation_behavior(test_graph, minimal_valid_sbml_dfs):
    """Test re-transformation behavior and error handling."""
    # Add mock data
    mock_df = pd.DataFrame({"scores": [500]}, index=["R00001"])
    minimal_valid_sbml_dfs.reactions_data["test_table"] = mock_df

    # Initial transformation
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "scores": {
                    "table": "test_table",
                    "variable": "scores",
                    "trans": "identity",
                }
            }
        }
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs)
    test_graph.transform_edges()  # Don't keep raw attributes

    # Try to change transformation - should fail without raw data
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "scores": {
                    "table": "test_table",
                    "variable": "scores",
                    "trans": "string_inv",
                }
            }
        },
        overwrite=True,
    )
    with pytest.raises(
        ValueError, match="Cannot re-transform attributes without raw data"
    ):
        test_graph.transform_edges()

    # Clear transformation history for second part of test
    test_graph.set_metadata(transformations_applied={"reactions": {}})
    test_graph.set_metadata(raw_attributes={"reactions": {}})

    # Reset with fresh state
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "scores": {
                    "table": "test_table",
                    "variable": "scores",
                    "trans": "identity",
                }
            }
        },
        overwrite=True,
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs, overwrite=True)
    test_graph.transform_edges(
        keep_raw_attributes=True
    )  # This time keep raw attributes
    first_transform = test_graph.es["scores"][:]

    # Now change transformation - should work because we kept raw data
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "scores": {
                    "table": "test_table",
                    "variable": "scores",
                    "trans": "string_inv",
                }
            }
        },
        overwrite=True,
    )
    test_graph.transform_edges()  # Should work now
    second_transform = test_graph.es["scores"][:]

    # Values should be different after re-transformation
    assert first_transform != second_transform
    assert (
        test_graph.get_metadata("transformations_applied")["reactions"]["scores"]
        == "string_inv"
    )


def test_add_degree_attributes(test_graph):
    """Test add_degree_attributes method functionality."""
    # Create a more complex test graph with multiple edges to test degree calculations
    g = ig.Graph()
    g.add_vertices(5, attributes={"name": ["A", "B", "C", "D", "R00001"]})
    g.add_edges(
        [(0, 1), (1, 2), (2, 3), (0, 2), (3, 4)]
    )  # A->B, B->C, C->D, A->C, D->R00001
    g.es["from"] = ["A", "B", "C", "A", "D"]
    g.es["to"] = ["B", "C", "D", "C", "R00001"]
    g.es["r_id"] = ["R1", "R2", "R3", "R4", "R5"]

    napistu_graph = NapistuGraph.from_igraph(g)

    # Add degree attributes
    napistu_graph.add_degree_attributes()

    # Check that degree attributes were added to edges
    assert NAPISTU_GRAPH_EDGES.SC_DEGREE in napistu_graph.es.attributes()
    assert NAPISTU_GRAPH_EDGES.SC_CHILDREN in napistu_graph.es.attributes()
    assert NAPISTU_GRAPH_EDGES.SC_PARENTS in napistu_graph.es.attributes()

    # Get edge data to verify calculations
    edges_df = napistu_graph.get_edge_dataframe()

    # Test degree calculations for specific nodes:
    # Node A: 2 children (B, C), 0 parents -> degree = 2
    # Node B: 1 child (C), 1 parent (A) -> degree = 2
    # Node C: 1 child (D), 2 parents (A, B) -> degree = 3
    # Node D: 1 child (R00001), 1 parent (C) -> degree = 2
    # Node R00001: 0 children, 1 parent (D) -> degree = 1 (but filtered out)

    # Check edge A->B: should have A's degree (2 children, 0 parents = 2)
    edge_a_to_b = edges_df[(edges_df["from"] == "A") & (edges_df["to"] == "B")].iloc[0]
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 2
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 0

    # Check edge B->C: should have B's degree (1 child, 1 parent = 2)
    edge_b_to_c = edges_df[(edges_df["from"] == "B") & (edges_df["to"] == "C")].iloc[0]
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 1

    # Check edge C->D: should have C's degree (1 child, 2 parents = 3)
    edge_c_to_d = edges_df[(edges_df["from"] == "C") & (edges_df["to"] == "D")].iloc[0]
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 3
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 2

    # Check edge D->R00001: should have D's degree (1 child, 1 parent = 2)
    # Note: R00001 is a reaction node, so we use D's degree
    edge_d_to_r = edges_df[
        (edges_df["from"] == "D") & (edges_df["to"] == "R00001")
    ].iloc[0]
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 1

    # Test method chaining
    result = napistu_graph.add_degree_attributes()
    assert result is napistu_graph

    # Test that calling again doesn't change values (idempotent)
    edges_df_after = napistu_graph.get_edge_dataframe()
    pd.testing.assert_frame_equal(edges_df, edges_df_after)


def test_add_degree_attributes_pathological_case(test_graph):
    """Test add_degree_attributes method handles pathological case correctly."""
    # Create a test graph
    g = ig.Graph()
    g.add_vertices(3, attributes={"name": ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])  # A->B, B->C
    g.es["from"] = ["A", "B"]
    g.es["to"] = ["B", "C"]
    g.es["r_id"] = ["R1", "R2"]

    napistu_graph = NapistuGraph.from_igraph(g)

    # Manually add only some degree attributes to create pathological state
    napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_CHILDREN] = [1, 1]
    napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_PARENTS] = [0, 1]
    # Note: sc_degree is missing

    # Test that calling add_degree_attributes raises an error
    with pytest.raises(ValueError, match="Some degree attributes already exist"):
        napistu_graph.add_degree_attributes()

    # Test that the error message includes the specific attributes
    try:
        napistu_graph.add_degree_attributes()
    except ValueError as e:
        error_msg = str(e)
        assert "sc_children" in error_msg
        assert "sc_parents" in error_msg
        assert "sc_degree" in error_msg
        assert "inconsistent state" in error_msg
