from __future__ import annotations

import pytest

import igraph as ig
import pandas as pd

from napistu.network import data_handling, net_create
from napistu.network.constants import GRAPH_WIRING_APPROACHES


# Fixtures
@pytest.fixture
def mock_sbml_dfs():
    """Create a mock SBML_dfs object with test data."""

    class MockSBMLDfs:
        def __init__(self):
            self.species_data = {
                "test_table": pd.DataFrame(
                    {
                        "col1": [1, 2, 3],
                        "col2": ["a", "b", "c"],
                        "test_prefix_col": [4, 5, 6],
                    }
                ),
                "another_table": pd.DataFrame(
                    {"col3": [7, 8, 9], "col4": ["d", "e", "f"]}
                ),
            }
            self.reactions_data = {
                "reaction_table": pd.DataFrame(
                    {"rxn_col1": [10, 11, 12], "rxn_col2": ["g", "h", "i"]}
                )
            }

    return MockSBMLDfs()


@pytest.fixture
def test_entity_data():
    """Create a test data table."""
    return pd.DataFrame(
        {
            "attr1": [1, 2, 3],
            "attr2": ["a", "b", "c"],
            "test_prefix_attr": [4, 5, 6],
            "another_attr": [7, 8, 9],
        }
    )


def test_select_sbml_dfs_data_table(mock_sbml_dfs):
    """Test selecting data tables from SBML_dfs object."""
    # Test selecting specific species table
    result = data_handling._select_sbml_dfs_data_table(
        mock_sbml_dfs, "test_table", "species"
    )
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_sbml_dfs.species_data["test_table"])

    # Test selecting reactions table
    result = data_handling._select_sbml_dfs_data_table(
        mock_sbml_dfs, "reaction_table", "reactions"
    )
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_sbml_dfs.reactions_data["reaction_table"])

    # Test error cases
    with pytest.raises(ValueError, match="Invalid table_type"):
        data_handling._select_sbml_dfs_data_table(
            mock_sbml_dfs, table_type="invalid_type"
        )

    with pytest.raises(ValueError, match="Invalid table_name"):
        data_handling._select_sbml_dfs_data_table(
            mock_sbml_dfs, "invalid_table", "species"
        )

    # Test no data case
    mock_sbml_dfs.species_data = {}
    with pytest.raises(ValueError, match="No species data found"):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs)

    # Test multiple tables without specifying name
    mock_sbml_dfs.species_data = {
        "table1": pd.DataFrame({"col1": [1]}),
        "table2": pd.DataFrame({"col2": [2]}),
    }
    with pytest.raises(
        ValueError, match="Expected a single species data table but found 2"
    ):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs)


def test_select_data_table_attrs_basic(test_entity_data):
    """Test basic attribute selection from data table."""
    # Test single attribute as list
    result = data_handling._create_data_table_column_mapping(
        test_entity_data, ["attr1"]
    )
    assert isinstance(result, dict)
    assert result == {"attr1": "attr1"}

    # Test multiple attributes
    result = data_handling._create_data_table_column_mapping(
        test_entity_data, ["attr1", "attr2"]
    )
    assert isinstance(result, dict)
    assert result == {"attr1": "attr1", "attr2": "attr2"}

    # Test invalid attribute
    with pytest.raises(ValueError, match="following attributes were missing"):
        data_handling._create_data_table_column_mapping(
            test_entity_data, ["invalid_attr"]
        )


def test_select_data_table_attrs_advanced(test_entity_data):
    """Test advanced attribute selection features."""
    # Test dictionary renaming
    result = data_handling._create_data_table_column_mapping(
        test_entity_data, {"attr1": "new_name1", "attr2": "new_name2"}
    )
    assert isinstance(result, dict)
    assert result == {"attr1": "new_name1", "attr2": "new_name2"}

    # Test empty dictionary
    with pytest.raises(ValueError, match="No attributes found in the dictionary"):
        data_handling._create_data_table_column_mapping(test_entity_data, {})

    # Test invalid source columns
    with pytest.raises(ValueError, match="following source columns were missing"):
        data_handling._create_data_table_column_mapping(
            test_entity_data, {"invalid_attr": "new_name"}
        )

    # Test conflicting new column names
    with pytest.raises(
        ValueError, match="following new column names conflict with existing columns"
    ):
        data_handling._create_data_table_column_mapping(
            test_entity_data,
            {"attr1": "attr2"},  # trying to rename attr1 to attr2, which already exists
        )

    # Test None returns identity mapping for all columns
    result = data_handling._create_data_table_column_mapping(test_entity_data, None)
    assert isinstance(result, dict)
    expected = {col: col for col in test_entity_data.columns}
    assert result == expected

    # Test string pattern matching
    result = data_handling._create_data_table_column_mapping(
        test_entity_data, "test_prefix_.*"
    )
    assert isinstance(result, dict)
    assert result == {"test_prefix_attr": "test_prefix_attr"}


def test_create_graph_attrs_config():
    """Test creating graph attributes configuration with a single table and transformation."""
    # Test basic case with single table and transformation
    result = data_handling._create_graph_attrs_config(
        column_mapping={"col1": "col1"},
        data_type="species",
        table_name="test_table",
        transformation="identity",
    )

    expected = {
        "species": {
            "col1": {
                "table": "test_table",
                "variable": "col1",
                "trans": "identity",
            }
        }
    }
    assert result == expected

    # Test with column renaming
    result = data_handling._create_graph_attrs_config(
        column_mapping={"original_col": "new_col"},
        data_type="species",
        table_name="test_table",
        transformation="squared",
    )

    expected = {
        "species": {
            "new_col": {
                "table": "test_table",
                "variable": "original_col",
                "trans": "squared",
            }
        }
    }
    assert result == expected

    # Test with multiple columns but same table and transformation
    result = data_handling._create_graph_attrs_config(
        column_mapping={"col1": "col1", "col2": "renamed_col2"},
        data_type="species",
        table_name="test_table",
        transformation="identity",
    )

    expected = {
        "species": {
            "col1": {
                "table": "test_table",
                "variable": "col1",
                "trans": "identity",
            },
            "renamed_col2": {
                "table": "test_table",
                "variable": "col2",
                "trans": "identity",
            },
        }
    }
    assert result == expected


def test_add_results_table_to_graph(sbml_dfs_glucose_metabolism):
    """Test adding results table to graph."""
    # Create a test graph using create_napistu_graph
    graph = net_create.create_napistu_graph(
        sbml_dfs_glucose_metabolism, directed=True, wiring_approach="regulatory"
    )

    # Add some test data to sbml_dfs
    test_data = pd.DataFrame(
        {"test_attr": [1.0, 2.0, 3.0]},
        index=pd.Index(
            list(sbml_dfs_glucose_metabolism.species.index[:3]), name="s_id"
        ),
    )
    sbml_dfs_glucose_metabolism.add_species_data("test_table", test_data)

    # Test basic case - single attribute
    result = data_handling.add_results_table_to_graph(
        napistu_graph=graph,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        attribute_names=["test_attr"],
        table_name="test_table",
        inplace=False,
    )
    assert "test_attr" in result.vs.attributes()

    # Test with transformation
    def square(x):
        return x**2

    result = data_handling.add_results_table_to_graph(
        napistu_graph=graph,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        attribute_names=["test_attr"],
        table_name="test_table",
        transformation="square",
        custom_transformations={"square": square},
        inplace=False,
    )
    assert "test_attr" in result.vs.attributes()

    # Test inplace=True
    original_graph = graph.copy()
    assert (
        "test_attr" not in original_graph.vs.attributes()
    )  # Verify attribute doesn't exist before
    result = data_handling.add_results_table_to_graph(
        napistu_graph=original_graph,
        sbml_dfs=sbml_dfs_glucose_metabolism,
        attribute_names=["test_attr"],
        table_name="test_table",
        inplace=True,
    )
    assert result is None  # Function should return None when inplace=True
    assert (
        "test_attr" in original_graph.vs.attributes()
    )  # Verify original graph was modified

    # Test error cases
    with pytest.raises(ValueError, match="Invalid table_type"):
        data_handling.add_results_table_to_graph(
            napistu_graph=graph,
            sbml_dfs=sbml_dfs_glucose_metabolism,
            table_type="invalid",
        )

    with pytest.raises(NotImplementedError, match="Reactions are not yet supported"):
        data_handling.add_results_table_to_graph(
            napistu_graph=graph,
            sbml_dfs=sbml_dfs_glucose_metabolism,
            table_type="reactions",
        )


def test_add_graph_species_attribute(sbml_dfs_glucose_metabolism):
    """Test adding species attributes to graph."""
    # Create a test graph using create_napistu_graph
    graph = net_create.create_napistu_graph(
        sbml_dfs_glucose_metabolism,
        directed=True,
        wiring_approach=GRAPH_WIRING_APPROACHES.REGULATORY,
    )

    # Add test data to sbml_dfs
    test_data = pd.DataFrame(
        {"test_attr": [1.0, 2.0, 3.0]},
        index=pd.Index(
            list(sbml_dfs_glucose_metabolism.species.index[:3]), name="s_id"
        ),
    )
    sbml_dfs_glucose_metabolism.add_species_data("test_table", test_data)

    # Define custom transformations
    custom_transformations = {"square": lambda x: x**2}

    # Test attempting to overwrite an existing vertex attribute
    collision_attrs = {
        "species": {
            "name": {  # Using 'name' which is a required vertex attribute
                "table": "test_table",
                "variable": "test_attr",
                "trans": "identity",
            }
        }
    }

    with pytest.raises(
        ValueError, match="Attribute 'name' already exists in graph vertices"
    ):
        data_handling._add_graph_species_attribute(
            graph, sbml_dfs_glucose_metabolism, collision_attrs
        )

    # Test basic attribute addition
    species_graph_attrs = {
        "species": {
            "new_attr": {
                "table": "test_table",
                "variable": "test_attr",
                "trans": "identity",
            }
        }
    }

    result = data_handling._add_graph_species_attribute(
        graph, sbml_dfs_glucose_metabolism, species_graph_attrs
    )

    assert "new_attr" in result.vs.attributes()

    # Test with custom transformation
    species_graph_attrs = {
        "species": {
            "squared_attr": {
                "table": "test_table",
                "variable": "test_attr",
                "trans": "square",
            }
        }
    }

    result = data_handling._add_graph_species_attribute(
        graph,
        sbml_dfs_glucose_metabolism,
        species_graph_attrs,
        custom_transformations=custom_transformations,
    )

    assert "squared_attr" in result.vs.attributes()

    # Test error cases
    with pytest.raises(TypeError, match="species_graph_attrs must be a dict"):
        data_handling._add_graph_species_attribute(
            graph, sbml_dfs_glucose_metabolism, species_graph_attrs=[]
        )

    # Test missing required attributes in graph
    bad_graph = ig.Graph()
    bad_graph.add_vertices(3)

    # Use basic transformation to avoid transformation error
    basic_attrs = {
        "species": {
            "new_attr": {
                "table": "test_table",
                "variable": "test_attr",
                "trans": "identity",
            }
        }
    }

    with pytest.raises(ValueError, match="required attributes were missing"):
        data_handling._add_graph_species_attribute(
            bad_graph, sbml_dfs_glucose_metabolism, basic_attrs
        )
