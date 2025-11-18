import logging
import os
import tempfile

import igraph as ig
import pandas as pd
import pytest
from fs.errors import ResourceNotFound

from napistu.constants import SBML_DFS, SBOTERM_NAMES
from napistu.network import ng_utils
from napistu.network.constants import (
    DEFAULT_WT_TRANS,
    GRAPH_WIRING_APPROACHES,
    IGRAPH_DEFS,
    NAPISTU_GRAPH,
    NAPISTU_GRAPH_EDGE_DIRECTIONS,
    NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NAPISTU_METADATA_KEYS,
    NAPISTU_WEIGHTING_STRATEGIES,
    WEIGHT_TRANSFORMATIONS,
    WEIGHTING_SPEC,
)
from napistu.network.ng_core import NapistuGraph
from napistu.ontologies.constants import SPECIES_TYPES

logger = logging.getLogger(__name__)


@pytest.fixture
def test_graph():
    """Create a simple test graph."""
    g = ig.Graph()
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])
    g.es[SBML_DFS.R_ID] = ["R1", "R2"]
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]
    return NapistuGraph.from_igraph(g)


@pytest.fixture
def mixed_node_types_graph():
    """Create a graph with both species and reaction nodes for testing node type filtering."""
    g = ig.Graph()
    g.add_vertices(
        6,
        attributes={
            NAPISTU_GRAPH_VERTICES.NAME: [
                "species_A",
                "species_B",
                "reaction_1",
                "reaction_2",
                "species_C",
                "reaction_3",
            ],
            NAPISTU_GRAPH_VERTICES.NODE_TYPE: [
                NAPISTU_GRAPH_NODE_TYPES.SPECIES,
                NAPISTU_GRAPH_NODE_TYPES.SPECIES,
                NAPISTU_GRAPH_NODE_TYPES.REACTION,
                NAPISTU_GRAPH_NODE_TYPES.REACTION,
                NAPISTU_GRAPH_NODE_TYPES.SPECIES,
                NAPISTU_GRAPH_NODE_TYPES.REACTION,
            ],
        },
    )
    g.add_edges([(0, 2), (2, 1)])  # species_A -> reaction_1 -> species_B
    # species_C and reaction_2, reaction_3 are isolated
    return g


def test_remove_vertex_attributes():
    """Test removing vertex attributes from a graph."""
    g = ig.Graph()
    g.add_vertices(
        3,
        attributes={
            NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"],
            "foo": ["protein", "metabolite", "protein"],
            "bar": [1.0, 2.0, 3.0],
        },
    )
    g.add_edges([(0, 1), (1, 2)])

    napistu_graph = NapistuGraph.from_igraph(g)
    napistu_graph.remove_attributes(NAPISTU_GRAPH.VERTICES, ["foo", "bar"])

    assert "foo" not in napistu_graph.vs.attributes()
    assert "bar" not in napistu_graph.vs.attributes()
    assert NAPISTU_GRAPH_VERTICES.NAME in napistu_graph.vs.attributes()


def test_remove_edge_attributes():
    """Test removing edge attributes from a graph."""
    g = ig.Graph()
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])
    g.es["foo"] = [1.0, 2.0]
    g.es["bar"] = [0.8, 0.9]

    napistu_graph = NapistuGraph.from_igraph(g)
    napistu_graph.remove_attributes(NAPISTU_GRAPH.EDGES, ["foo", "bar"])

    assert "foo" not in napistu_graph.es.attributes()
    assert "bar" not in napistu_graph.es.attributes()


@pytest.mark.parametrize(
    "attribute_type,entity_type,data_method,transform_method",
    [
        (
            NAPISTU_GRAPH.VERTICES,
            SBML_DFS.SPECIES,
            "add_vertex_data",
            "transform_vertices",
        ),
        (NAPISTU_GRAPH.EDGES, SBML_DFS.REACTIONS, "add_edge_data", "transform_edges"),
    ],
)
def test_remove_attributes_metadata_cleanup(
    attribute_type, entity_type, data_method, transform_method
):
    """Test that removing attributes cleans up metadata for both vertices and edges."""
    # Create graph
    g = ig.Graph()
    g.add_vertices(
        3,
        attributes={
            NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"],
            SBML_DFS.S_ID: ["A", "B", "C"],  # Required for vertex data merging
        },
    )
    g.add_edges([(0, 1), (1, 2)])
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]  # Required for edge data merging
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]  # Required for edge data merging

    napistu_graph = NapistuGraph.from_igraph(g)

    # Create side-loaded data based on attribute type
    if attribute_type == NAPISTU_GRAPH.VERTICES:
        side_loaded_data = {
            "test_table": pd.DataFrame(
                {
                    SBML_DFS.S_ID: ["A", "B", "C"],
                    "foo": [1.0, 2.0, 3.0],
                    "bar": [10, 20, 30],
                }
            ).set_index(SBML_DFS.S_ID)
        }
    else:  # edges
        side_loaded_data = {
            "test_table": pd.DataFrame(
                {
                    NAPISTU_GRAPH_EDGES.FROM: ["A", "B"],
                    NAPISTU_GRAPH_EDGES.TO: ["B", "C"],
                    "foo": [1.0, 2.0],
                    "bar": [0.8, 0.9],
                }
            ).set_index([NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO])
        }

    # Set up graph attributes for transformation
    graph_attrs = {
        entity_type: {
            "foo": {
                WEIGHTING_SPEC.TABLE: "test_table",
                WEIGHTING_SPEC.VARIABLE: "foo",
                WEIGHTING_SPEC.TRANSFORMATION: WEIGHT_TRANSFORMATIONS.STRING,
            },
            "bar": {
                WEIGHTING_SPEC.TABLE: "test_table",
                WEIGHTING_SPEC.VARIABLE: "bar",
                WEIGHTING_SPEC.TRANSFORMATION: WEIGHT_TRANSFORMATIONS.IDENTITY,
            },
        }
    }
    napistu_graph.set_graph_attrs(graph_attrs)

    # Add data using the appropriate method
    getattr(napistu_graph, data_method)(side_loaded_attributes=side_loaded_data)

    # Apply transformations to create metadata trail
    getattr(napistu_graph, transform_method)(keep_raw_attributes=True)

    # Verify metadata exists
    assert NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES in napistu_graph._metadata
    assert entity_type in napistu_graph._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES]
    assert (
        "foo"
        in napistu_graph._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type]
    )
    assert (
        "bar"
        in napistu_graph._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type]
    )
    assert NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED in napistu_graph._metadata
    assert (
        entity_type
        in napistu_graph._metadata[NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED]
    )

    # Remove one attribute
    napistu_graph.remove_attributes(attribute_type, ["foo"])

    # Verify foo metadata was cleaned up
    assert (
        "foo"
        not in napistu_graph._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][
            entity_type
        ]
    )
    assert (
        "foo"
        not in napistu_graph._metadata[NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED][
            entity_type
        ]
    )
    # But bar should still be there
    assert (
        "bar"
        in napistu_graph._metadata[NAPISTU_METADATA_KEYS.RAW_ATTRIBUTES][entity_type]
    )
    assert (
        "bar"
        in napistu_graph._metadata[NAPISTU_METADATA_KEYS.TRANSFORMATIONS_APPLIED][
            entity_type
        ]
    )


def test_remove_isolated_vertices():
    """Test removing isolated vertices from a graph."""

    g = ig.Graph()
    g.add_vertices(
        5, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C", "D", "E"]}
    )
    g.add_edges([(0, 1), (2, 3)])  # A-B, C-D connected; E isolated

    napstu_graph = NapistuGraph.from_igraph(g)
    napstu_graph.remove_isolated_vertices("all")
    assert napstu_graph.vcount() == 4  # Should have 4 vertices after removing E
    assert "E" not in [
        v[NAPISTU_GRAPH_VERTICES.NAME] for v in napstu_graph.vs
    ]  # E should be gone


def test_remove_isolated_vertices_with_node_types(mixed_node_types_graph):
    """Test removing isolated vertices with node type filtering."""

    # Test default behavior (remove only reactions)
    napstu_graph = NapistuGraph.from_igraph(mixed_node_types_graph)

    # Test default behavior (remove only reactions)
    napstu_graph.remove_isolated_vertices()
    assert napstu_graph.vcount() == 4  # Should remove reaction_2 and reaction_3
    remaining_names = [v[NAPISTU_GRAPH_VERTICES.NAME] for v in napstu_graph.vs]
    assert "species_C" in remaining_names  # species singleton should remain
    assert "reaction_2" not in remaining_names  # reaction singleton should be removed
    assert "reaction_3" not in remaining_names  # reaction singleton should be removed

    # Test removing only species
    napstu_graph2 = NapistuGraph.from_igraph(mixed_node_types_graph)
    napstu_graph2.remove_isolated_vertices(SBML_DFS.SPECIES)
    assert (
        napstu_graph2.vcount() == 5
    )  # Should remove species_C, keep reaction_2 and reaction_3
    remaining_names2 = [v[NAPISTU_GRAPH_VERTICES.NAME] for v in napstu_graph2.vs]
    assert "species_C" not in remaining_names2  # species singleton should be removed
    assert "reaction_2" in remaining_names2  # reaction singleton should remain
    assert "reaction_3" in remaining_names2  # reaction singleton should remain

    # Test removing only reactions
    napstu_graph2_reactions = NapistuGraph.from_igraph(mixed_node_types_graph)
    napstu_graph2_reactions.remove_isolated_vertices(SBML_DFS.REACTIONS)
    assert (
        napstu_graph2_reactions.vcount() == 4
    )  # Should remove reaction_2 and reaction_3, keep species_C
    remaining_names2_reactions = [
        v[NAPISTU_GRAPH_VERTICES.NAME] for v in napstu_graph2_reactions.vs
    ]
    assert "species_C" in remaining_names2_reactions  # species singleton should remain
    assert (
        "reaction_2" not in remaining_names2_reactions
    )  # reaction singleton should be removed
    assert (
        "reaction_3" not in remaining_names2_reactions
    )  # reaction singleton should be removed

    # Test removing all
    napstu_graph3 = NapistuGraph.from_igraph(mixed_node_types_graph)
    napstu_graph3.remove_isolated_vertices("all")
    assert (
        napstu_graph3.vcount() == 3
    )  # Should remove species_C, reaction_2, reaction_3
    remaining_names3 = [v[NAPISTU_GRAPH_VERTICES.NAME] for v in napstu_graph3.vs]
    assert "species_C" not in remaining_names3
    assert "reaction_2" not in remaining_names3
    assert "reaction_3" not in remaining_names3

    # Test that ValueError is raised when node_type attribute is missing
    g4 = ig.Graph()
    g4.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g4.add_edges([(0, 1)])  # A-B connected; C isolated

    napstu_graph4 = NapistuGraph.from_igraph(g4)

    with pytest.raises(ValueError, match="Cannot filter by reactions"):
        napstu_graph4.remove_isolated_vertices(SBML_DFS.REACTIONS)


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


def test_vertex_dataframe_ordering_preservation():
    """Test that get_vertex_dataframe preserves the original vertex ordering."""
    # Create a graph with vertices in a specific order
    g = NapistuGraph(directed=True)

    # Add vertices with specific names in a particular order
    vertex_names = ["C", "A", "B", "D"]
    for name in vertex_names:
        g.add_vertex(
            name=name, node_type=SBML_DFS.SPECIES, species_type=SPECIES_TYPES.METABOLITE
        )

    # Add some edges
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    g.add_edge("C", "D")

    # Get vertex dataframe
    vertex_df = g.get_vertex_dataframe()

    # Verify ordering is preserved (should match igraph's internal order)
    assert len(vertex_df) == 4
    assert list(vertex_df[NAPISTU_GRAPH_VERTICES.NAME]) == vertex_names

    # The index should be named 'index' and contain sequential values
    assert vertex_df.index.name == IGRAPH_DEFS.INDEX
    assert list(vertex_df.index) == list(range(4))  # 0, 1, 2, 3

    # Verify that the index matches the DataFrame row position
    for i, row in vertex_df.iterrows():
        assert i == row.name  # row.name is the index value


def test_edge_dataframe_ordering_preservation():
    """Test that get_edge_dataframe preserves the original edge ordering."""
    # Create a graph with edges in a specific order
    g = NapistuGraph(directed=True)

    # Add vertices
    g.add_vertex(name="A", node_type=SBML_DFS.SPECIES)
    g.add_vertex(name="B", node_type=SBML_DFS.SPECIES)
    g.add_vertex(name="C", node_type=SBML_DFS.SPECIES)
    g.add_vertex(name="D", node_type=SBML_DFS.SPECIES)

    # Add edges in a specific order
    edge_pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
    for source, target in edge_pairs:
        g.add_edge(source, target, weight=1.0)

    # Get edge dataframe
    edge_df = g.get_edge_dataframe()

    # Verify ordering is preserved
    assert len(edge_df) == 4
    assert list(edge_df[IGRAPH_DEFS.SOURCE]) == [0, 1, 2, 0]  # A=0, B=1, C=2, D=3
    assert list(edge_df[IGRAPH_DEFS.TARGET]) == [1, 2, 3, 3]  # B=1, C=2, D=3, D=3

    # Verify that edge attributes are in the same order
    assert list(edge_df[NAPISTU_GRAPH_EDGES.WEIGHT]) == [1.0, 1.0, 1.0, 1.0]


def test_to_pandas_dfs_ordering_consistency():
    """Test that to_pandas_dfs maintains consistent ordering between vertices and edges."""
    # Create a graph with known structure
    g = NapistuGraph(directed=True)

    # Add vertices in specific order
    vertex_names = ["X", "Y", "Z"]
    for name in vertex_names:
        g.add_vertex(name=name, node_type=SBML_DFS.SPECIES, value=len(name))

    # Add edges
    g.add_edge("X", "Y", edge_type="interaction")
    g.add_edge("Y", "Z", edge_type="regulation")

    # Get both dataframes
    vertex_df, edge_df = g.to_pandas_dfs()

    # Verify vertex ordering
    assert list(vertex_df[NAPISTU_GRAPH_VERTICES.NAME]) == vertex_names
    assert list(vertex_df[IGRAPH_DEFS.INDEX]) == list(range(3))

    # Verify edge ordering and that source/target indices are valid
    assert len(edge_df) == 2
    assert list(edge_df[IGRAPH_DEFS.SOURCE]) == [0, 1]  # X=0, Y=1
    assert list(edge_df[IGRAPH_DEFS.TARGET]) == [1, 2]  # Y=1, Z=2
    assert list(edge_df["edge_type"]) == ["interaction", "regulation"]

    # Verify all edge indices are valid vertex indices
    max_vertex_idx = vertex_df[IGRAPH_DEFS.INDEX].max()
    assert edge_df[IGRAPH_DEFS.SOURCE].max() <= max_vertex_idx
    assert edge_df[IGRAPH_DEFS.TARGET].max() <= max_vertex_idx


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
        stored_reactions == attrs[SBML_DFS.REACTIONS]
    ), f"Expected {attrs[SBML_DFS.REACTIONS]}, got {stored_reactions}"
    assert (
        stored_species == attrs[SBML_DFS.SPECIES]
    ), f"Expected {attrs[SBML_DFS.SPECIES]}, got {stored_species}"

    # Get attributes through helper method
    reactions_result = test_graph._get_entity_attrs(SBML_DFS.REACTIONS)
    species_result = test_graph._get_entity_attrs(SBML_DFS.SPECIES)

    assert (
        reactions_result == attrs[SBML_DFS.REACTIONS]
    ), f"Expected {attrs[SBML_DFS.REACTIONS]}, got {reactions_result}"
    assert (
        species_result == attrs[SBML_DFS.SPECIES]
    ), f"Expected {attrs['species']}, got {species_result}"

    # Test that method raises ValueError for unknown entity types
    with pytest.raises(ValueError, match="Unknown entity_type: 'nonexistent'"):
        test_graph._get_entity_attrs("nonexistent")

    # Test that method returns None for empty attributes
    test_graph.set_metadata(reaction_attrs={})
    assert test_graph._get_entity_attrs(SBML_DFS.REACTIONS) is None


def test_compare_and_merge_attrs(test_graph):
    """Test the _compare_and_merge_attrs method directly."""
    new_attrs = {
        "string_wt": {
            WEIGHTING_SPEC.TABLE: "string",
            WEIGHTING_SPEC.VARIABLE: "score",
            WEIGHTING_SPEC.TRANSFORMATION: "identity",
        }
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
        "existing": {
            WEIGHTING_SPEC.TABLE: "test",
            WEIGHTING_SPEC.VARIABLE: "val",
            WEIGHTING_SPEC.TRANSFORMATION: "identity",
        }
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
            "attr1": {
                WEIGHTING_SPEC.TABLE: "test",
                WEIGHTING_SPEC.VARIABLE: "val",
                WEIGHTING_SPEC.TRANSFORMATION: "identity",
            }
        }
    }
    test_graph.set_graph_attrs(initial)

    # Fresh mode should fail with existing data
    with pytest.raises(ValueError, match="Existing reaction_attrs found"):
        test_graph.set_graph_attrs(
            {
                "reactions": {
                    "attr2": {
                        WEIGHTING_SPEC.TABLE: "test",
                        WEIGHTING_SPEC.VARIABLE: "val2",
                        WEIGHTING_SPEC.TRANSFORMATION: "identity",
                    }
                }
            }
        )

    # Extend mode should work
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr2": {
                    WEIGHTING_SPEC.TABLE: "new",
                    WEIGHTING_SPEC.VARIABLE: "val2",
                    WEIGHTING_SPEC.TRANSFORMATION: "identity",
                }
            }
        },
        mode="extend",
    )
    reaction_attrs = test_graph.get_metadata("reaction_attrs")
    assert "attr1" in reaction_attrs and "attr2" in reaction_attrs


def test_add_edge_data_basic_functionality(test_graph, minimal_valid_sbml_dfs):
    """Test basic add_edge_data functionality with mock reaction data."""
    # Update the test graph to have the correct r_ids that match the SBML data
    test_graph.es[SBML_DFS.R_ID] = [
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
    test_graph.es[SBML_DFS.R_ID] = [
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


def test_add_edge_data_extend_mode_overlapping_attrs_bug(
    test_graph, minimal_valid_sbml_dfs
):
    """Test that extend mode should NOT error when there are overlapping attributes.

    This test demonstrates the bug where extend mode incorrectly raises a ValueError
    when there are overlapping attributes, when it should simply add only new attributes.
    """
    # Update the test graph to have the correct r_ids that match the SBML data
    test_graph.es["r_id"] = [
        "R00001",
        "R00001",
    ]  # Both edges should map to the same reaction

    # Create mock data with overlapping attributes
    minimal_valid_sbml_dfs.reactions_data["table1"] = pd.DataFrame(
        {"attr1": [10], "attr2": [20]}, index=["R00001"]
    )
    minimal_valid_sbml_dfs.reactions_data["table2"] = pd.DataFrame(
        {"attr1": [30], "attr3": [50]}, index=["R00001"]  # attr1 overlaps, attr3 is new
    )

    # First, add attr1 and attr2 to the graph
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr1": {"table": "table1", "variable": "attr1", "trans": "identity"},
                "attr2": {"table": "table1", "variable": "attr2", "trans": "identity"},
            }
        }
    )
    test_graph.add_edge_data(minimal_valid_sbml_dfs)

    # Verify initial attributes exist
    assert "attr1" in test_graph.es.attributes()
    assert "attr2" in test_graph.es.attributes()
    initial_attr1_values = test_graph.es["attr1"]
    initial_attr2_values = test_graph.es["attr2"]

    # Now try to extend with overlapping attributes (attr1) and new attribute (attr3)
    # This should NOT raise an error in extend mode - it should only add attr3
    test_graph.set_graph_attrs(
        {
            "reactions": {
                "attr1": {"table": "table2", "variable": "attr1", "trans": "identity"},
                "attr3": {"table": "table2", "variable": "attr3", "trans": "identity"},
            }
        },
        mode="extend",
    )

    # This should now work correctly in extend mode - only add new attributes
    test_graph.add_edge_data(minimal_valid_sbml_dfs, mode="extend")

    # Verify that only attr3 was added, and existing attributes remain unchanged
    assert "attr3" in test_graph.es.attributes()
    assert test_graph.es["attr1"] == initial_attr1_values  # Should be unchanged
    assert test_graph.es["attr2"] == initial_attr2_values  # Should be unchanged

    # Test extend mode with overwrite=True - currently ignored in extend mode
    # The current implementation ignores overwrite in extend mode and only adds new attributes
    test_graph.add_edge_data(minimal_valid_sbml_dfs, mode="extend", overwrite=True)
    # attr1 should remain unchanged since it already exists (overwrite is ignored in extend mode)
    assert test_graph.es["attr1"] == initial_attr1_values
    # attr2 should remain unchanged since it's not in the new data
    assert test_graph.es["attr2"] == initial_attr2_values
    # attr3 should still be there
    assert "attr3" in test_graph.es.attributes()


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
    mock_df.index.name = SBML_DFS.R_ID
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
    g.add_vertices(
        5, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C", "D", "R00001"]}
    )
    g.add_edges(
        [(0, 1), (1, 2), (2, 3), (0, 2), (3, 4)]
    )  # A->B, B->C, C->D, A->C, D->R00001
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B", "C", "A", "D"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C", "D", "C", "R00001"]
    g.es[NAPISTU_GRAPH_EDGES.R_ID] = ["R1", "R2", "R3", "R4", "R5"]

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
    edge_a_to_b = edges_df[
        (edges_df[NAPISTU_GRAPH_EDGES.FROM] == "A")
        & (edges_df[NAPISTU_GRAPH_EDGES.TO] == "B")
    ].iloc[0]
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 2
    assert edge_a_to_b[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 0

    # Check edge B->C: should have B's degree (1 child, 1 parent = 2)
    edge_b_to_c = edges_df[
        (edges_df[NAPISTU_GRAPH_EDGES.FROM] == "B")
        & (edges_df[NAPISTU_GRAPH_EDGES.TO] == "C")
    ].iloc[0]
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_b_to_c[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 1

    # Check edge C->D: should have C's degree (1 child, 2 parents = 3)
    edge_c_to_d = edges_df[
        (edges_df[NAPISTU_GRAPH_EDGES.FROM] == "C")
        & (edges_df[NAPISTU_GRAPH_EDGES.TO] == "D")
    ].iloc[0]
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 3
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_c_to_d[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 2

    # Check edge D->R00001: should have D's degree (1 child, 1 parent = 2)
    # Note: R00001 is a reaction node, so we use D's degree
    edge_d_to_r = edges_df[
        (edges_df[NAPISTU_GRAPH_EDGES.FROM] == "D")
        & (edges_df[NAPISTU_GRAPH_EDGES.TO] == "R00001")
    ].iloc[0]
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_DEGREE] == 2
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == 1
    assert edge_d_to_r[NAPISTU_GRAPH_EDGES.SC_PARENTS] == 1

    # Test method chaining
    result = napistu_graph.add_degree_attributes(inplace=False)

    # Test that calling again doesn't change values (idempotent)
    edges_df_after = result.get_edge_dataframe()
    pd.testing.assert_frame_equal(edges_df, edges_df_after)


def test_add_degree_attributes_pathological_case(test_graph):
    """Test add_degree_attributes method handles pathological case correctly."""
    # Create a test graph
    g = ig.Graph()
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])  # A->B, B->C
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]
    g.es[NAPISTU_GRAPH_EDGES.R_ID] = ["R1", "R2"]

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


def test_reverse_edges():
    """Test the reverse_edges method."""
    # Create test graph with edge attributes
    g = ig.Graph(directed=True)
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])  # A->B->C

    # Add attributes that should be swapped
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]
    g.es[NAPISTU_GRAPH_EDGES.WEIGHT] = [1.0, 2.0]
    g.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] = [0.5, 1.5]
    g.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM] = [1.0, -2.0]
    g.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM] = [-1.0, 2.0]
    g.es[NAPISTU_GRAPH_EDGES.DIRECTION] = [
        NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD,
        NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE,
    ]

    napistu_graph = NapistuGraph.from_igraph(g)
    napistu_graph.add_degree_attributes()

    # Test reversal
    napistu_graph.reverse_edges()

    # Check metadata
    assert napistu_graph.is_reversed is True

    # Check that edge attributes represent reversed graph
    # (FROM/TO attributes are swapped, so conceptually the graph is reversed)
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.FROM] == ["B", "C"]  # was ["A", "B"]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.TO] == ["A", "B"]  # was ["B", "C"]

    # Check other attribute swapping
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] == [0.5, 1.5]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == [1.0, 2.0]

    # Check degree attribute swapping
    # Original: A->B->C
    # After reversal: B->A, C->B
    # The degree attributes should be swapped (SC_PARENTS ↔ SC_CHILDREN)
    # and SC_DEGREE should remain unchanged (total degree)
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_PARENTS] == [
        1,
        1,
    ]  # swapped from SC_CHILDREN
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == [
        0,
        1,
    ]  # swapped from SC_PARENTS
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_DEGREE] == [
        1,
        2,
    ]  # unchanged (total degree)

    # Check special handling - stoichiometries are swapped then negated
    # Original: upstream=[1.0, -2.0], downstream=[-1.0, 2.0]
    # After swap: upstream=[-1.0, 2.0], downstream=[1.0, -2.0]
    # After negate: upstream=[1.0, -2.0], downstream=[-1.0, 2.0]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM] == [1.0, -2.0]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM] == [-1.0, 2.0]
    expected_directions = [
        NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE,  # forward -> reverse
        NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD,  # reverse -> forward
    ]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.DIRECTION] == expected_directions

    # Test double reversal restores original state
    napistu_graph.reverse_edges()
    assert napistu_graph.is_reversed is False

    # Check that attributes are restored to original values
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.FROM] == ["A", "B"]  # restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.TO] == ["B", "C"]  # restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] == [1.0, 2.0]  # restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == [
        0.5,
        1.5,
    ]  # restored
    # Original degree values: A->B (0 parents, 1 child), B->C (1 parent, 1 child)
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_PARENTS] == [0, 1]  # restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_CHILDREN] == [1, 1]  # restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.SC_DEGREE] == [1, 2]  # restored
    # Stoichiometries restored
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM] == [1.0, -2.0]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM] == [-1.0, 2.0]


def test_set_weights():
    """Test the set_weights method."""
    import igraph as ig

    from napistu.network.constants import (
        NAPISTU_GRAPH_EDGES,
        NAPISTU_WEIGHTING_STRATEGIES,
    )

    # Create a simple test graph
    g = ig.Graph(directed=True)
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])  # A->B->C

    # Add basic edge attributes
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]
    # Add required species_type attribute for topology weighting
    g.es[NAPISTU_GRAPH_EDGES.SPECIES_TYPE] = ["protein", "protein"]

    napistu_graph = NapistuGraph.from_igraph(g)
    napistu_graph.add_degree_attributes()

    # Test unweighted strategy
    napistu_graph.set_weights(
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED
    )

    # Check that weights are set to 1
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] == [1, 1]
    assert napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == [1, 1]

    # Test topology strategy
    napistu_graph.set_weights(weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY)

    # Check that topology weights are applied
    assert NAPISTU_GRAPH_EDGES.WEIGHT in napistu_graph.es.attributes()
    assert NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM in napistu_graph.es.attributes()

    # Test invalid strategy
    with pytest.raises(
        ValueError,
        match="weighting_strategy was invalid_strategy and must be one of: mixed, topology, unweighted",
    ):
        napistu_graph.set_weights(weighting_strategy="invalid_strategy")

    # Test with reaction attributes set via set_graph_attrs
    napistu_graph_with_attrs = NapistuGraph.from_igraph(g)
    napistu_graph_with_attrs.add_degree_attributes()

    # Add the string_wt attribute that the mixed strategy expects
    napistu_graph_with_attrs.es["string_wt"] = [0.8, 0.9]

    napistu_graph_with_attrs.set_graph_attrs(
        {
            "reactions": {
                "string_wt": {
                    "table": "string",
                    "variable": "score",
                    "trans": "identity",
                }
            }
        }
    )

    # Test mixed strategy with reaction attributes
    napistu_graph_with_attrs.set_weights(
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.MIXED
    )

    # Check that mixed weights are applied
    assert NAPISTU_GRAPH_EDGES.WEIGHT in napistu_graph_with_attrs.es.attributes()
    assert (
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM in napistu_graph_with_attrs.es.attributes()
    )
    assert "source_wt" in napistu_graph_with_attrs.es.attributes()


def test_get_weight_variables():
    """Test the _get_weight_variables utility method."""
    import igraph as ig

    from napistu.network.constants import NAPISTU_GRAPH_EDGES

    # Create a test graph
    g = ig.Graph(directed=True)
    g.add_vertices(2, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B"]})
    g.add_edges([(0, 1)])
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B"]
    g.es["custom_weight"] = [2.5]

    napistu_graph = NapistuGraph.from_igraph(g)

    # Test with weight_by parameter
    weight_vars = napistu_graph._get_weight_variables(weight_by=["custom_weight"])
    assert "custom_weight" in weight_vars
    assert weight_vars["custom_weight"][WEIGHTING_SPEC.TABLE] == "__edges__"
    assert weight_vars["custom_weight"][WEIGHTING_SPEC.VARIABLE] == "custom_weight"

    # Test with non-existent attribute
    with pytest.raises(ValueError, match="Edge attributes not found in graph"):
        napistu_graph._get_weight_variables(weight_by=["non_existent"])

    # Test with no reaction_attrs and no weight_by
    with pytest.raises(ValueError, match="No reaction_attrs found"):
        napistu_graph._get_weight_variables()

    # Test with reaction_attrs set
    napistu_graph.set_graph_attrs(
        {
            SBML_DFS.REACTIONS: {
                "string_wt": {
                    WEIGHTING_SPEC.TABLE: "string",
                    WEIGHTING_SPEC.VARIABLE: "score",
                    WEIGHTING_SPEC.TRANSFORMATION: "identity",
                }
            }
        }
    )

    weight_vars = napistu_graph._get_weight_variables()
    assert "string_wt" in weight_vars
    assert weight_vars["string_wt"][WEIGHTING_SPEC.TABLE] == "string"


def test_process_napistu_graph_with_reactions_data(sbml_dfs):
    """Test process_napistu_graph with reactions data and graph attributes."""
    import numpy as np
    import pandas as pd

    from napistu.network.constants import NAPISTU_WEIGHTING_STRATEGIES
    from napistu.network.net_create import process_napistu_graph

    # Add reactions_data table called "string" with combined_score variable
    # Only add data for a subset of reactions to test source weights
    reaction_ids = sbml_dfs.reactions.index.tolist()
    # Add string data for only half of the reactions
    subset_size = len(reaction_ids) // 2
    subset_reactions = reaction_ids[:subset_size]

    # Generate random scores in the 150-1000 range for subset of reactions
    combined_scores = np.random.uniform(150, 1000, len(subset_reactions))

    string_data = pd.DataFrame(
        {"combined_score": combined_scores}, index=subset_reactions
    )
    string_data.index.name = SBML_DFS.R_ID

    # Add the reactions data to sbml_dfs
    sbml_dfs.add_reactions_data("string", string_data)

    # Define reaction_graph_attrs matching graph_attrs_spec.yaml
    reaction_graph_attrs = {
        SBML_DFS.REACTIONS: {
            "string_wt": {
                WEIGHTING_SPEC.TABLE: "string",
                WEIGHTING_SPEC.VARIABLE: "combined_score",
                WEIGHTING_SPEC.TRANSFORMATION: "string_inv",
            }
        }
    }

    # Process the napistu graph with the specified parameters
    processed_graph = process_napistu_graph(
        sbml_dfs=sbml_dfs,
        directed=True,
        wiring_approach=GRAPH_WIRING_APPROACHES.BIPARTITE,
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.MIXED,
        reaction_graph_attrs=reaction_graph_attrs,
        verbose=False,
    )

    # Verify the graph was processed correctly
    assert processed_graph is not None
    assert hasattr(processed_graph, "es")
    assert hasattr(processed_graph, "vs")

    # Check that the string_wt attribute was added to edges
    assert "string_wt" in processed_graph.es.attributes()

    # Check that weights were applied
    assert NAPISTU_GRAPH_EDGES.WEIGHT in processed_graph.es.attributes()
    if processed_graph.is_directed():
        assert NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM in processed_graph.es.attributes()

    # Check that source_wt was created (part of mixed strategy)
    assert NAPISTU_GRAPH_EDGES.SOURCE_WT in processed_graph.es.attributes()

    # Verify the graph has the expected metadata
    assert (
        processed_graph.get_metadata(NAPISTU_METADATA_KEYS.WIRING_APPROACH)
        == GRAPH_WIRING_APPROACHES.BIPARTITE
    )
    assert (
        processed_graph.get_metadata(NAPISTU_METADATA_KEYS.WEIGHTING_STRATEGY)
        == NAPISTU_WEIGHTING_STRATEGIES.MIXED
    )

    # Check that transformed string weights are in the correct range (≥1 and ≤6.67 for string_inv)
    string_weights = processed_graph.es["string_wt"]
    non_null_weights = [w for w in string_weights if pd.notna(w)]
    assert len(non_null_weights) > 0
    # Allow for some numerical precision issues
    weight_checks = [bool(0.99 <= w <= 6.68) for w in non_null_weights]
    assert all(weight_checks)

    # Check that source weights are correct:
    # - 10 if string_wt is not None (has string data)
    # - 1 if string_wt is None (no string data)
    source_weights = processed_graph.es[NAPISTU_GRAPH_EDGES.SOURCE_WT]
    for i, (sw, str_wt) in enumerate(zip(source_weights, string_weights)):
        if pd.notna(str_wt):
            assert (
                sw == 10
            ), f"Source weight should be 10 when string_wt exists, got {sw} at edge {i}"
        else:
            assert (
                sw == 1
            ), f"Source weight should be 1 when string_wt is None, got {sw} at edge {i}"

    # Check that final weights are in the correct range (≥1 and <10)
    final_weights = processed_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT]
    assert all(0.49 <= w < 10 for w in final_weights)

    if processed_graph.is_directed():
        upstream_weights = processed_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM]
        assert all(0.49 <= w < 10 for w in upstream_weights)


@pytest.mark.skip_on_windows
def test_to_pickle_and_from_pickle(napistu_graph):
    """Test saving and loading a NapistuGraph via pickle."""
    # Use the existing napistu_graph fixture
    # Add some test metadata to verify it's preserved
    napistu_graph.set_metadata(test_attr="test_value", graph_type="test")

    # Save to pickle
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        pickle_path = tmp_file.name

    try:
        napistu_graph.to_pickle(pickle_path)

        # Load from pickle
        loaded_graph = NapistuGraph.from_pickle(pickle_path)

        # Verify the loaded graph is identical
        assert isinstance(loaded_graph, NapistuGraph)
        assert loaded_graph.vcount() == napistu_graph.vcount()
        assert loaded_graph.ecount() == napistu_graph.ecount()
        assert loaded_graph.is_directed() == napistu_graph.is_directed()
        assert loaded_graph.get_metadata("test_attr") == "test_value"
        assert loaded_graph.get_metadata("graph_type") == "test"

    finally:
        # Clean up
        if os.path.exists(pickle_path):
            os.unlink(pickle_path)


@pytest.mark.skip_on_windows
def test_from_pickle_nonexistent_file():
    """Test that from_pickle raises appropriate error for nonexistent file."""

    # Create a temporary directory and use a path that definitely doesn't exist
    with tempfile.TemporaryDirectory() as temp_dir:
        nonexistent_path = os.path.join(temp_dir, "nonexistent_file.pkl")
        with pytest.raises(ResourceNotFound):
            NapistuGraph.from_pickle(nonexistent_path)


def test_reaction_edge_weighting():
    """Test reaction edge downweighting functionality."""
    # Create a simple test graph: A → R1 → B and C → D (direct)
    ng = NapistuGraph(directed=True)
    ng.add_vertices(
        5, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "R1", "B", "C", "D"]}
    )
    # Set node_types: R1 is a reaction, others are species
    ng.vs[1][
        NAPISTU_GRAPH_VERTICES.NODE_TYPE
    ] = NAPISTU_GRAPH_NODE_TYPES.REACTION  # R1 is a reaction
    ng.add_edges([(0, 1), (1, 2), (3, 4)])  # A→R1, R1→B, C→D

    # Test with default multiplier (0.5)
    ng.set_weights(weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED)

    # Check that reaction edges have reduced weights
    edges_df = ng.get_edge_dataframe()

    # Path A→R1→B should have total cost of 1.0 (0.5 + 0.5)
    # Path C→D should have cost of 1.0
    assert edges_df.loc[0, NAPISTU_GRAPH_EDGES.WEIGHT] == 0.5  # A→R1
    assert edges_df.loc[1, NAPISTU_GRAPH_EDGES.WEIGHT] == 0.5  # R1→B
    assert edges_df.loc[2, NAPISTU_GRAPH_EDGES.WEIGHT] == 1.0  # C→D (no reaction)

    # Check that upstream_weight is also modified for directed graphs
    assert edges_df.loc[0, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == 0.5  # A→R1
    assert edges_df.loc[1, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == 0.5  # R1→B
    assert (
        edges_df.loc[2, NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == 1.0
    )  # C→D (no reaction)

    # Test disabling the feature
    ng.set_weights(
        weighting_strategy=NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
        reaction_edge_multiplier=1.0,
    )
    edges_df = ng.get_edge_dataframe()

    # All edges should have weight 1.0
    assert all(edges_df[NAPISTU_GRAPH_EDGES.WEIGHT] == 1.0)
    assert all(edges_df[NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM] == 1.0)


def test_add_sbml_dfs_summaries(napistu_graph_metabolism, sbml_dfs_metabolism):
    """Test that add_sbml_dfs_summaries adds vertex summary attributes correctly."""

    # Get the expected summary columns
    expected_summaries = ng_utils.get_sbml_dfs_vertex_summaries(sbml_dfs_metabolism)

    # Debug: check if we got a valid DataFrame
    assert expected_summaries is not None, "get_sbml_dfs_vertex_summaries returned None"
    assert isinstance(
        expected_summaries, pd.DataFrame
    ), f"Expected DataFrame, got {type(expected_summaries)}"
    assert len(expected_summaries.columns) > 0, "Summary DataFrame has no columns"

    expected_columns = set(expected_summaries.columns)

    # Test inplace=True (default)
    working_napistu_graph_metabolism = napistu_graph_metabolism.copy()
    assert working_napistu_graph_metabolism is not napistu_graph_metabolism
    result = working_napistu_graph_metabolism.add_sbml_dfs_summaries(
        sbml_dfs_metabolism, inplace=True
    )
    assert result is None

    # Check that all expected columns were added as vertex attributes
    vertex_attrs = set(working_napistu_graph_metabolism.vs.attributes())
    assert expected_columns.issubset(vertex_attrs)

    # Test inplace=False
    new_graph = napistu_graph_metabolism.add_sbml_dfs_summaries(
        sbml_dfs_metabolism, inplace=False
    )
    assert new_graph is not None
    assert new_graph is not napistu_graph_metabolism

    # Check that the new graph has the summary attributes
    new_vertex_attrs = set(new_graph.vs.attributes())
    assert expected_columns.issubset(new_vertex_attrs)


def test_add_vertex_data_basic_functionality(test_graph, minimal_valid_sbml_dfs):
    """Test basic add_vertex_data functionality - mirrors add_edge_data but for vertices."""
    # Set up species data similar to edge data test
    test_graph.vs[SBML_DFS.S_ID] = ["S00001", "S00001", "S00002"]

    mock_df = pd.DataFrame({"score_col": [100, 200]}, index=["S00001", "S00002"])
    mock_df.index.name = SBML_DFS.S_ID
    minimal_valid_sbml_dfs.species_data["mock_table"] = mock_df

    species_attrs = {
        "score_col": {
            WEIGHTING_SPEC.TABLE: "mock_table",
            WEIGHTING_SPEC.VARIABLE: "score_col",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        }
    }
    test_graph.set_graph_attrs({SBML_DFS.SPECIES: species_attrs})

    # Add vertex data
    test_graph.add_vertex_data(minimal_valid_sbml_dfs)

    # Verify attributes were added to vertices (not edges)
    assert "score_col" in test_graph.vs.attributes()
    assert "score_col" not in test_graph.es.attributes()

    # Check values were assigned correctly
    assert test_graph.vs["score_col"][0] == 100  # S00001
    assert test_graph.vs["score_col"][2] == 200  # S00002


def test_add_vertex_data_error_handling(test_graph, minimal_valid_sbml_dfs):
    """Test that add_vertex_data raises appropriate errors like add_edge_data."""
    test_graph.vs[SBML_DFS.S_ID] = ["S00001", "S00001", "S00002"]

    mock_df = pd.DataFrame({"score_col": [100, 200]}, index=["S00001", "S00002"])
    mock_df.index.name = SBML_DFS.S_ID
    minimal_valid_sbml_dfs.species_data["mock_table"] = mock_df

    species_attrs = {
        "score_col": {
            WEIGHTING_SPEC.TABLE: "mock_table",
            WEIGHTING_SPEC.VARIABLE: "score_col",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        }
    }
    test_graph.set_graph_attrs({SBML_DFS.SPECIES: species_attrs})

    # Add data once
    test_graph.add_vertex_data(minimal_valid_sbml_dfs)

    # Test that it raises error for existing attributes (like add_edge_data does)
    with pytest.raises(ValueError, match="Vertex attributes already exist"):
        test_graph.add_vertex_data(minimal_valid_sbml_dfs, mode="fresh")


def test_transform_vertices_basic_functionality(test_graph, minimal_valid_sbml_dfs):
    """Test basic transform_vertices functionality - mirrors transform_edges but for vertices."""
    # Set up species data and add it to vertices
    test_graph.vs[SBML_DFS.S_ID] = ["S00001", "S00001", "S00002"]

    mock_df = pd.DataFrame({"score_col": [100, 200]}, index=["S00001", "S00002"])
    mock_df.index.name = SBML_DFS.S_ID
    minimal_valid_sbml_dfs.species_data["mock_table"] = mock_df

    species_attrs = {
        "score_col": {
            WEIGHTING_SPEC.TABLE: "mock_table",
            WEIGHTING_SPEC.VARIABLE: "score_col",
            WEIGHTING_SPEC.TRANSFORMATION: "square",  # Use a transformation
        }
    }
    # Set up custom transformations for validation
    custom_transformations = {"square": lambda x: x**2}
    test_graph.set_graph_attrs(
        {SBML_DFS.SPECIES: species_attrs}, custom_transformations=custom_transformations
    )

    # Add vertex data first
    test_graph.add_vertex_data(minimal_valid_sbml_dfs)

    # Verify original values
    assert test_graph.vs["score_col"][0] == 100
    assert test_graph.vs["score_col"][2] == 200

    # Transform vertices with custom transformation
    test_graph.transform_vertices(custom_transformations=custom_transformations)

    # Verify transformation was applied (square transformation)
    assert test_graph.vs["score_col"][0] == 10000  # 100^2
    assert test_graph.vs["score_col"][2] == 40000  # 200^2


def test_transform_vertices_validation_behavior(test_graph, minimal_valid_sbml_dfs):
    """Test that set_graph_attrs validates transformations when custom transformations are not provided."""
    # Set up species data
    test_graph.vs[SBML_DFS.S_ID] = ["S00001", "S00001", "S00002"]

    mock_df = pd.DataFrame({"score_col": [100, 200]}, index=["S00001", "S00002"])
    mock_df.index.name = SBML_DFS.S_ID
    minimal_valid_sbml_dfs.species_data["mock_table"] = mock_df

    species_attrs = {
        "score_col": {
            WEIGHTING_SPEC.TABLE: "mock_table",
            WEIGHTING_SPEC.VARIABLE: "score_col",
            WEIGHTING_SPEC.TRANSFORMATION: "square",  # Custom transformation
        }
    }

    # This should raise an error because "square" is not a built-in transformation
    # and no custom transformations are provided
    with pytest.raises(ValueError, match="transformation 'square' was not defined"):
        test_graph.set_graph_attrs({SBML_DFS.SPECIES: species_attrs})


def test_transform_vertices_error_handling(test_graph, minimal_valid_sbml_dfs):
    """Test that transform_vertices handles missing species_attrs like transform_edges."""
    # Test without setting species_attrs - should warn and return early
    test_graph.transform_vertices()

    # No error should be raised, just a warning logged
    # This mirrors the behavior of transform_edges


def test_add_attributes_to_graph_inplace_edges(test_graph):
    """Test _add_attributes_to_graph_inplace with edges using multi-index."""
    # Get a few edges from the test graph
    edge_df = test_graph.get_edge_dataframe()

    num_edges = min(3, len(edge_df))
    selected_edges = edge_df.head(num_edges)

    # Create edge data indexed by (source, target) tuples
    edge_data = pd.DataFrame(
        {
            "edge_weight": [0.5, 0.8, 0.3][:num_edges],
            "edge_confidence": [0.9, 0.7, 0.8][:num_edges],
            "edge_type": ["activation", "inhibition", "binding"][:num_edges],
        },
        index=pd.MultiIndex.from_tuples(
            [
                (row[IGRAPH_DEFS.SOURCE], row[IGRAPH_DEFS.TARGET])
                for _, row in selected_edges.iterrows()
            ],
            names=[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET],
        ),
    )

    # Test adding attributes to edges
    test_graph._add_attributes_df(
        entity_data=edge_data, target_entity=IGRAPH_DEFS.EDGES, overwrite=False
    )

    # Verify attributes were added to edges
    assert "edge_weight" in test_graph.es.attributes()
    assert "edge_confidence" in test_graph.es.attributes()
    assert "edge_type" in test_graph.es.attributes()

    # Verify values were assigned correctly
    edge_weights = test_graph.es["edge_weight"]
    edge_confidences = test_graph.es["edge_confidence"]
    edge_types = test_graph.es["edge_type"]

    # Check that the values match our test data
    for i in range(num_edges):
        assert edge_weights[i] == edge_data["edge_weight"].iloc[i]
        assert edge_confidences[i] == edge_data["edge_confidence"].iloc[i]
        assert edge_types[i] == edge_data["edge_type"].iloc[i]


def test_add_attributes_to_graph_inplace_vertices(test_graph):
    """Test _add_attributes_to_graph_inplace with vertices using single index."""
    # Get a few vertices from the test graph
    vertex_df = test_graph.get_vertex_dataframe()
    num_vertices = min(3, len(vertex_df))
    selected_vertices = vertex_df.head(num_vertices)

    # Create vertex data indexed by vertex names
    vertex_data = pd.DataFrame(
        {
            "vertex_expression": [1.2, 3.4, 5.6][:num_vertices],
            "vertex_confidence": [0.8, 0.9, 0.7][:num_vertices],
            "vertex_location": ["nucleus", "cytoplasm", "membrane"][:num_vertices],
        },
        index=pd.Index(selected_vertices[IGRAPH_DEFS.NAME], name=IGRAPH_DEFS.NAME),
    )

    # Test adding attributes to vertices
    test_graph._add_attributes_df(
        entity_data=vertex_data, target_entity=IGRAPH_DEFS.VERTICES, overwrite=False
    )

    # Verify attributes were added to vertices
    assert "vertex_expression" in test_graph.vs.attributes()
    assert "vertex_confidence" in test_graph.vs.attributes()
    assert "vertex_location" in test_graph.vs.attributes()

    # Verify values were assigned correctly
    vertex_expressions = test_graph.vs["vertex_expression"]
    vertex_confidences = test_graph.vs["vertex_confidence"]
    vertex_locations = test_graph.vs["vertex_location"]

    # Check that the values match our test data
    for i in range(num_vertices):
        assert vertex_expressions[i] == vertex_data["vertex_expression"].iloc[i]
        assert vertex_confidences[i] == vertex_data["vertex_confidence"].iloc[i]
        assert vertex_locations[i] == vertex_data["vertex_location"].iloc[i]


def test_add_attributes_to_graph_inplace_overwrite(test_graph):
    """Test _add_attributes_to_graph_inplace with overwrite=True."""
    # Get a few vertices from the test graph
    vertex_df = test_graph.get_vertex_dataframe()
    num_vertices = min(2, len(vertex_df))
    selected_vertices = vertex_df.head(num_vertices)

    # Create initial vertex data
    initial_data = pd.DataFrame(
        {
            "test_attr": [100, 200][:num_vertices],
        },
        index=pd.Index(selected_vertices[IGRAPH_DEFS.NAME], name=IGRAPH_DEFS.NAME),
    )

    # Add initial attributes
    test_graph._add_attributes_df(
        entity_data=initial_data, target_entity=IGRAPH_DEFS.VERTICES, overwrite=False
    )

    # Verify initial values
    assert test_graph.vs["test_attr"][0] == 100
    if num_vertices > 1:
        assert test_graph.vs["test_attr"][1] == 200

    # Create new data with different values
    new_data = pd.DataFrame(
        {
            "test_attr": [300, 400][:num_vertices],
        },
        index=pd.Index(selected_vertices[IGRAPH_DEFS.NAME], name=IGRAPH_DEFS.NAME),
    )

    # Add new attributes with overwrite=True
    test_graph._add_attributes_df(
        entity_data=new_data, target_entity=IGRAPH_DEFS.VERTICES, overwrite=True
    )

    # Verify values were overwritten
    assert test_graph.vs["test_attr"][0] == 300
    if num_vertices > 1:
        assert test_graph.vs["test_attr"][1] == 400


def test_add_edge_data_with_both_sources(napistu_graph, sbml_dfs_w_data):
    """Test add_edge_data using both sbml_dfs and side_loaded_attributes as data sources."""
    # Get a couple of real edges for side-loaded data
    edge_df = napistu_graph.get_edge_dataframe()
    selected_edges = edge_df.head(2)
    edge_pairs = list(
        zip(
            selected_edges[NAPISTU_GRAPH_EDGES.FROM],
            selected_edges[NAPISTU_GRAPH_EDGES.TO],
        )
    )

    # Create side-loaded data for these edges
    side_loaded_df = pd.DataFrame(
        {
            "external_confidence": [0.95, 0.87],
            "external_source": ["database_A", "database_B"],
        },
        index=pd.MultiIndex.from_tuples(
            edge_pairs, names=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
        ),
    )

    side_loaded_attributes = {"external_db": side_loaded_df}

    # Set up attributes for both data sources
    reaction_attrs = {
        "rxn_score": {
            WEIGHTING_SPEC.TABLE: "rxn_data",
            WEIGHTING_SPEC.VARIABLE: "rxn_attr_float",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
        "confidence": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "external_confidence",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
    }

    napistu_graph.set_graph_attrs({SBML_DFS.REACTIONS: reaction_attrs})

    # Add data from both sources
    napistu_graph.add_edge_data(
        sbml_dfs_w_data, side_loaded_attributes=side_loaded_attributes
    )

    # Verify attributes exist and have some non-None values
    edge_attrs = napistu_graph.es.attributes()
    assert "rxn_score" in edge_attrs
    assert "confidence" in edge_attrs

    # Check that we have some non-None values from each source
    rxn_scores = napistu_graph.es["rxn_score"]
    confidences = napistu_graph.es["confidence"]

    assert any(
        score is not None and not pd.isna(score) for score in rxn_scores
    ), "No valid rxn_score values found"
    assert any(
        conf is not None and not pd.isna(conf) for conf in confidences
    ), "No valid confidence values found"


def test_add_edge_data_side_loaded_only(napistu_graph):
    """Test add_edge_data using only side_loaded_attributes (no sbml_dfs)."""
    # Get a couple of real edges for side-loaded data
    edge_df = napistu_graph.get_edge_dataframe()
    selected_edges = edge_df.head(2)
    edge_pairs = list(
        zip(
            selected_edges[NAPISTU_GRAPH_EDGES.FROM],
            selected_edges[NAPISTU_GRAPH_EDGES.TO],
        )
    )

    # Create side-loaded data for these edges
    side_loaded_df = pd.DataFrame(
        {
            "external_confidence": [0.95, 0.87],
            "external_source": ["database_A", "database_B"],
        },
        index=pd.MultiIndex.from_tuples(
            edge_pairs, names=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
        ),
    )

    side_loaded_attributes = {"external_db": side_loaded_df}

    # Set up attributes for side-loaded data only
    reaction_attrs = {
        "confidence": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "external_confidence",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
        "source_db": {
            WEIGHTING_SPEC.TABLE: "external_db",
            WEIGHTING_SPEC.VARIABLE: "external_source",
            WEIGHTING_SPEC.TRANSFORMATION: DEFAULT_WT_TRANS,
        },
    }

    napistu_graph.set_graph_attrs({SBML_DFS.REACTIONS: reaction_attrs})

    # Add data from side-loaded source only
    napistu_graph.add_edge_data(side_loaded_attributes=side_loaded_attributes)

    # Verify attributes exist and have some non-None values
    edge_attrs = napistu_graph.es.attributes()
    assert "confidence" in edge_attrs
    assert "source_db" in edge_attrs

    # Check that we have some non-None values
    confidences = napistu_graph.es["confidence"]
    source_dbs = napistu_graph.es["source_db"]

    assert any(
        conf is not None and not pd.isna(conf) for conf in confidences
    ), "No valid confidence values found"
    assert any(
        source is not None and not pd.isna(source) for source in source_dbs
    ), "No valid source_db values found"


def test_show_summary(napistu_graph):
    """Test that show_summary method runs without errors."""

    # Call show_summary - should not raise any exceptions
    napistu_graph.show_summary()

    # If we get here, the method ran successfully
    # We can also verify that get_summary works and returns expected structure
    summary_stats = napistu_graph.get_summary()

    # Print the actual summary stats for inspection
    print("\n=== Summary Stats ===")
    print(f"n_vertices: {summary_stats['n_vertices']}")
    print(f"n_edges: {summary_stats['n_edges']}")
    print(f"vertex_node_type_dict: {summary_stats['vertex_node_type_dict']}")
    print(f"vertex_species_type_dict: {summary_stats['vertex_species_type_dict']}")
    print(f"sbo_name_counts_dict: {summary_stats['sbo_name_counts_dict']}")
    print(f"vertex_attributes: {summary_stats['vertex_attributes']}")
    print(f"edge_attributes: {summary_stats['edge_attributes']}")
    print("===================\n")

    # Verify the summary has the expected keys
    expected_keys = [
        "n_vertices",
        "vertex_node_type_dict",
        "vertex_species_type_dict",
        "vertex_attributes",
        "n_edges",
        "sbo_name_counts_dict",
        "edge_attributes",
    ]

    for key in expected_keys:
        assert key in summary_stats, f"Missing key: {key}"

    # Test vertices: type and count
    assert isinstance(summary_stats["n_vertices"], int)
    assert (
        summary_stats["n_vertices"] == 30
    ), f"Expected 30 vertices, got {summary_stats['n_vertices']}"

    # Test edges: type and count
    assert isinstance(summary_stats["n_edges"], int)
    assert (
        summary_stats["n_edges"] == 32
    ), f"Expected 32 edges, got {summary_stats['n_edges']}"

    # Test vertex node type breakdown: type and content
    assert isinstance(summary_stats["vertex_node_type_dict"], dict)
    expected_node_types = {
        NAPISTU_GRAPH_NODE_TYPES.SPECIES: 23,
        NAPISTU_GRAPH_NODE_TYPES.REACTION: 7,
    }
    assert (
        summary_stats["vertex_node_type_dict"] == expected_node_types
    ), f"Expected {expected_node_types}, got {summary_stats['vertex_node_type_dict']}"

    # Test vertex species type breakdown: type and content
    assert isinstance(summary_stats["vertex_species_type_dict"], dict)
    expected_species_types = {"metabolite": 13, "complex": 9, "protein": 1}
    assert (
        summary_stats["vertex_species_type_dict"] == expected_species_types
    ), f"Expected {expected_species_types}, got {summary_stats['vertex_species_type_dict']}"

    # Test SBO term breakdown: type and content
    assert isinstance(summary_stats["sbo_name_counts_dict"], dict)
    expected_sbo_terms = {
        SBOTERM_NAMES.PRODUCT: 13,
        SBOTERM_NAMES.REACTANT: 13,
        SBOTERM_NAMES.CATALYST: 6,
    }
    assert (
        summary_stats["sbo_name_counts_dict"] == expected_sbo_terms
    ), f"Expected {expected_sbo_terms}, got {summary_stats['sbo_name_counts_dict']}"

    # Test vertex attributes: type and content
    assert isinstance(summary_stats["vertex_attributes"], list)
    expected_vertex_attrs = {
        NAPISTU_GRAPH_VERTICES.NAME,
        NAPISTU_GRAPH_VERTICES.NODE_NAME,
        NAPISTU_GRAPH_VERTICES.NODE_TYPE,
        NAPISTU_GRAPH_VERTICES.SPECIES_TYPE,
        SBML_DFS.S_ID,
        SBML_DFS.C_ID,
    }
    actual_vertex_attrs = set(summary_stats["vertex_attributes"])
    assert (
        actual_vertex_attrs == expected_vertex_attrs
    ), f"Expected {expected_vertex_attrs}, got {actual_vertex_attrs}"

    # Test edge attributes: type and content
    assert isinstance(summary_stats["edge_attributes"], list)
    expected_edge_attrs = {
        NAPISTU_GRAPH_EDGES.FROM,
        NAPISTU_GRAPH_EDGES.TO,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
        NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
        NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
        SBML_DFS.R_ID,
        NAPISTU_GRAPH_EDGES.SPECIES_TYPE,
        SBML_DFS.R_ISREVERSIBLE,
        NAPISTU_GRAPH_EDGES.DIRECTION,
        NAPISTU_GRAPH_EDGES.SC_DEGREE,
        NAPISTU_GRAPH_EDGES.SC_CHILDREN,
        NAPISTU_GRAPH_EDGES.SC_PARENTS,
        "topo_weights",
        "upstream_topo_weights",
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    }
    actual_edge_attrs = set(summary_stats["edge_attributes"])
    assert (
        actual_edge_attrs == expected_edge_attrs
    ), f"Expected {expected_edge_attrs}, got {actual_edge_attrs}"


def test_get_vertex_series(test_graph):
    """Test get_vertex_series method."""
    # Test getting vertex names
    vertex_names = test_graph.get_vertex_series(NAPISTU_GRAPH_VERTICES.NAME)
    expected_names = pd.Series(
        ["A", "B", "C"], index=["A", "B", "C"], name=NAPISTU_GRAPH_VERTICES.NAME
    )
    expected_names.index.name = NAPISTU_GRAPH_VERTICES.NAME
    pd.testing.assert_series_equal(vertex_names, expected_names)

    # Test getting non-existent attribute
    with pytest.raises(KeyError, match="Vertex attribute 'nonexistent' not found"):
        test_graph.get_vertex_series("nonexistent")


def test_get_edge_series(test_graph):
    """Test get_edge_series method."""
    # Test getting edge R_ID attribute
    edge_r_ids = test_graph.get_edge_series(NAPISTU_GRAPH_EDGES.R_ID)
    expected_r_ids = pd.Series(
        ["R1", "R2"],
        index=pd.MultiIndex.from_tuples(
            [("A", "B"), ("B", "C")],
            names=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO],
        ),
        name=NAPISTU_GRAPH_EDGES.R_ID,
    )
    pd.testing.assert_series_equal(edge_r_ids, expected_r_ids)

    # Test getting non-existent attribute
    with pytest.raises(KeyError, match="Edge attribute 'nonexistent' not found"):
        test_graph.get_edge_series("nonexistent")


def test_get_edge_endpoint_attributes(napistu_graph):
    """Test get_edge_endpoint_attributes with different attribute combinations."""

    # Test case 1: Pull out species_type only
    species_df = napistu_graph.get_edge_endpoint_attributes(
        NAPISTU_GRAPH_VERTICES.SPECIES_TYPE
    )

    # Verify structure
    assert isinstance(species_df, pd.DataFrame)
    assert species_df.index.names == [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
    assert species_df.columns.names == [
        NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ATTRIBUTE_NAME,
        NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ENDPOINT,
    ]

    # Verify columns
    expected_cols = [
        (NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, IGRAPH_DEFS.SOURCE),
        (NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, IGRAPH_DEFS.TARGET),
    ]
    assert list(species_df.columns) == expected_cols

    # Verify we have data for all edges
    assert len(species_df) == napistu_graph.ecount()

    # Test case 2: Pull out both species_type and node_type
    multi_attrs_df = napistu_graph.get_edge_endpoint_attributes(
        [NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, NAPISTU_GRAPH_VERTICES.NODE_TYPE]
    )

    # Verify structure
    assert isinstance(multi_attrs_df, pd.DataFrame)
    assert multi_attrs_df.index.names == [
        NAPISTU_GRAPH_EDGES.FROM,
        NAPISTU_GRAPH_EDGES.TO,
    ]
    assert multi_attrs_df.columns.names == [
        NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ATTRIBUTE_NAME,
        NAPISTU_GRAPH_EDGE_ENDPOINT_ATTRIBUTES.ENDPOINT,
    ]

    # Verify columns
    expected_cols = [
        (NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, IGRAPH_DEFS.SOURCE),
        (NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, IGRAPH_DEFS.TARGET),
        (NAPISTU_GRAPH_VERTICES.NODE_TYPE, IGRAPH_DEFS.SOURCE),
        (NAPISTU_GRAPH_VERTICES.NODE_TYPE, IGRAPH_DEFS.TARGET),
    ]
    assert list(multi_attrs_df.columns) == expected_cols

    # Verify we have data for all edges
    assert len(multi_attrs_df) == napistu_graph.ecount()

    # Test case 3: Try to pull out a missing attribute (should raise KeyError)
    with pytest.raises(
        KeyError, match="Vertex attribute 'missing_attribute' does not exist"
    ):
        napistu_graph.get_edge_endpoint_attributes("missing_attribute")

    # Also test with list containing missing attribute
    with pytest.raises(
        KeyError, match="Vertex attribute 'missing_attribute' does not exist"
    ):
        napistu_graph.get_edge_endpoint_attributes(
            [NAPISTU_GRAPH_VERTICES.SPECIES_TYPE, "missing_attribute"]
        )


def test_deduplicate_edges():
    """Test deduplicate_edges method with a minimal graph containing duplicate edges."""
    # Create a minimal graph with duplicate edges (same FROM -> TO pairs)
    g = ig.Graph(directed=True)
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})

    # Add edges: A->B appears twice, B->C appears once
    # First A->B edge
    g.add_edge(0, 1)  # A -> B
    g.es[0][NAPISTU_GRAPH_EDGES.FROM] = "A"
    g.es[0][NAPISTU_GRAPH_EDGES.TO] = "B"
    g.es[0][NAPISTU_GRAPH_EDGES.WEIGHT] = 1.0
    g.es[0][SBML_DFS.R_ID] = "R1"

    # Second A->B edge (duplicate)
    g.add_edge(0, 1)  # A -> B (duplicate)
    g.es[1][NAPISTU_GRAPH_EDGES.FROM] = "A"
    g.es[1][NAPISTU_GRAPH_EDGES.TO] = "B"
    g.es[1][NAPISTU_GRAPH_EDGES.WEIGHT] = 2.0  # Different weight
    g.es[1][SBML_DFS.R_ID] = "R2"  # Different reaction ID

    # B->C edge (no duplicate)
    g.add_edge(1, 2)  # B -> C
    g.es[2][NAPISTU_GRAPH_EDGES.FROM] = "B"
    g.es[2][NAPISTU_GRAPH_EDGES.TO] = "C"
    g.es[2][NAPISTU_GRAPH_EDGES.WEIGHT] = 3.0
    g.es[2][SBML_DFS.R_ID] = "R3"

    napistu_graph = NapistuGraph.from_igraph(g)

    # Verify initial state: 3 edges (2 duplicates of A->B, 1 B->C)
    assert napistu_graph.ecount() == 3
    edges_df_before = napistu_graph.get_edge_dataframe()
    assert len(edges_df_before) == 3

    # Deduplicate edges
    napistu_graph.deduplicate_edges()

    # Verify final state: 2 edges (1 A->B, 1 B->C)
    assert napistu_graph.ecount() == 2
    edges_df_after = napistu_graph.get_edge_dataframe()
    assert len(edges_df_after) == 2

    # Verify that only one A->B edge remains (the first one)
    a_to_b_edges = edges_df_after[
        (edges_df_after[NAPISTU_GRAPH_EDGES.FROM] == "A")
        & (edges_df_after[NAPISTU_GRAPH_EDGES.TO] == "B")
    ]
    assert len(a_to_b_edges) == 1
    # The first edge should be kept (with weight 1.0 and R_ID "R1")
    assert a_to_b_edges.iloc[0][NAPISTU_GRAPH_EDGES.WEIGHT] == 1.0
    assert a_to_b_edges.iloc[0][SBML_DFS.R_ID] == "R1"

    # Verify B->C edge is still present
    b_to_c_edges = edges_df_after[
        (edges_df_after[NAPISTU_GRAPH_EDGES.FROM] == "B")
        & (edges_df_after[NAPISTU_GRAPH_EDGES.TO] == "C")
    ]
    assert len(b_to_c_edges) == 1
    assert b_to_c_edges.iloc[0][NAPISTU_GRAPH_EDGES.WEIGHT] == 3.0

    # Verify graph structure is preserved (vertices unchanged)
    assert napistu_graph.vcount() == 3
    vertex_names = [v[NAPISTU_GRAPH_VERTICES.NAME] for v in napistu_graph.vs]
    assert set(vertex_names) == {"A", "B", "C"}


def test_deduplicate_edges_no_duplicates():
    """Test deduplicate_edges method when there are no duplicate edges."""
    # Create a graph with no duplicate edges
    g = ig.Graph(directed=True)
    g.add_vertices(3, attributes={NAPISTU_GRAPH_VERTICES.NAME: ["A", "B", "C"]})
    g.add_edges([(0, 1), (1, 2)])  # A->B, B->C
    g.es[NAPISTU_GRAPH_EDGES.FROM] = ["A", "B"]
    g.es[NAPISTU_GRAPH_EDGES.TO] = ["B", "C"]

    napistu_graph = NapistuGraph.from_igraph(g)
    initial_edge_count = napistu_graph.ecount()

    # Deduplicate edges (should not change anything)
    napistu_graph.deduplicate_edges()

    # Verify graph is unchanged
    assert napistu_graph.ecount() == initial_edge_count
    assert napistu_graph.vcount() == 3
