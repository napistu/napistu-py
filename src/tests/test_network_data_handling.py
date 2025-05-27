from __future__ import annotations

import pytest
import pandas as pd
from napistu.network import data_handling
from napistu import sbml_dfs_core
from napistu.constants import ENTITIES_W_DATA

# Fixtures
@pytest.fixture
def mock_sbml_dfs():
    """Create a mock SBML_dfs object with test data."""
    class MockSBMLDfs:
        def __init__(self):
            self.species_data = {
                "test_table": pd.DataFrame({
                    "col1": [1, 2, 3],
                    "col2": ["a", "b", "c"],
                    "test_prefix_col": [4, 5, 6]
                }),
                "another_table": pd.DataFrame({
                    "col3": [7, 8, 9],
                    "col4": ["d", "e", "f"]
                })
            }
            self.reactions_data = {
                "reaction_table": pd.DataFrame({
                    "rxn_col1": [10, 11, 12],
                    "rxn_col2": ["g", "h", "i"]
                })
            }
    return MockSBMLDfs()

@pytest.fixture
def test_data_table():
    """Create a test data table."""
    return pd.DataFrame({
        "attr1": [1, 2, 3],
        "attr2": ["a", "b", "c"],
        "test_prefix_attr": [4, 5, 6],
        "another_attr": [7, 8, 9]
    })

def test_select_sbml_dfs_data_table(mock_sbml_dfs):
    """Test selecting data tables from SBML_dfs object."""
    # Test selecting specific species table
    result = data_handling._select_sbml_dfs_data_table(mock_sbml_dfs, "test_table", "species")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_sbml_dfs.species_data["test_table"])

    # Test selecting reactions table
    result = data_handling._select_sbml_dfs_data_table(mock_sbml_dfs, "reaction_table", "reactions")
    assert isinstance(result, pd.DataFrame)
    assert result.equals(mock_sbml_dfs.reactions_data["reaction_table"])

    # Test error cases
    with pytest.raises(ValueError, match="Invalid table_type"):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs, table_type="invalid_type")

    with pytest.raises(ValueError, match="Invalid table_name"):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs, "invalid_table", "species")

    # Test no data case
    mock_sbml_dfs.species_data = {}
    with pytest.raises(ValueError, match="No species data found"):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs)

    # Test multiple tables without specifying name
    mock_sbml_dfs.species_data = {
        "table1": pd.DataFrame({"col1": [1]}),
        "table2": pd.DataFrame({"col2": [2]})
    }
    with pytest.raises(ValueError, match="Expected a single species data table but found 2"):
        data_handling._select_sbml_dfs_data_table(mock_sbml_dfs)

def test_select_data_table_attrs_basic(test_data_table):
    """Test basic attribute selection from data table."""
    # Test single attribute as list
    result = data_handling._select_sbml_dfs_data_table_attrs(test_data_table, ["attr1"])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["attr1"]
    assert result["attr1"].tolist() == [1, 2, 3]

    # Test multiple attributes
    result = data_handling._select_sbml_dfs_data_table_attrs(test_data_table, ["attr1", "attr2"])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["attr1", "attr2"]
    assert result["attr1"].tolist() == [1, 2, 3]
    assert result["attr2"].tolist() == ["a", "b", "c"]

    # Test invalid attribute
    with pytest.raises(ValueError, match="following attributes were missing"):
        data_handling._select_sbml_dfs_data_table_attrs(test_data_table, ["invalid_attr"])

def test_select_data_table_attrs_advanced(test_data_table):
    """Test advanced attribute selection features."""
    # Test dictionary renaming
    result = data_handling._select_sbml_dfs_data_table_attrs(
        test_data_table, 
        {"attr1": "new_name1", "attr2": "new_name2"}
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["new_name1", "new_name2"]
    assert result["new_name1"].tolist() == [1, 2, 3]
    assert result["new_name2"].tolist() == ["a", "b", "c"]

    # Test empty dictionary
    with pytest.raises(ValueError, match="No attributes found in the dictionary"):
        data_handling._select_sbml_dfs_data_table_attrs(test_data_table, {})

    # Test invalid source columns
    with pytest.raises(ValueError, match="following source columns were missing"):
        data_handling._select_sbml_dfs_data_table_attrs(
            test_data_table, 
            {"invalid_attr": "new_name"}
        )
    
    # Test conflicting new column names
    with pytest.raises(ValueError, match="following new column names conflict with existing columns"):
        data_handling._select_sbml_dfs_data_table_attrs(
            test_data_table,
            {"attr1": "attr2"}  # trying to rename attr1 to attr2, which already exists
        ) 