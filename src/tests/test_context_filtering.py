from __future__ import annotations

import copy
import pytest
import pandas as pd
from napistu import sbml_dfs_utils
from napistu.constants import SBML_DFS
from napistu.context.filtering import (
    filter_species_by_attribute,
    find_species_with_attribute,
    _binarize_species_data,
    filter_reactions_with_disconnected_cspecies,
)


@pytest.fixture
def sbml_dfs_with_test_data(sbml_dfs):
    """Add test data to the sbml_dfs fixture for filtering tests."""
    # Add location data
    location_data = pd.DataFrame(
        index=sbml_dfs.species.index[:5],
        data={
            "compartment": ["nucleus", "cytoplasm", "nucleus", "membrane", "cytoplasm"],
            "confidence": [0.9, 0.8, 0.7, 0.95, 0.85],
        },
    )
    sbml_dfs.add_species_data("location", location_data)

    # Add expression data
    expression_data = pd.DataFrame(
        index=sbml_dfs.species.index[:5],
        data={
            "is_expressed": [True, True, False, True, False],
            "expression_level": [100, 50, 0, 75, 0],
        },
    )
    sbml_dfs.add_species_data("expression", expression_data)

    return sbml_dfs


def test_find_species_to_filter_by_attribute(sbml_dfs_with_test_data):
    """Test the find_species_to_filter_by_attribute function."""
    # Get the first 5 species IDs for reference
    test_species = list(sbml_dfs_with_test_data.species.index[:5])

    # Test filtering by single value
    filtered = find_species_with_attribute(
        sbml_dfs_with_test_data.species_data["location"], "compartment", "nucleus"
    )
    assert len(filtered) == 2
    assert all(s_id in test_species for s_id in filtered)

    # Test filtering by list of values
    filtered = find_species_with_attribute(
        sbml_dfs_with_test_data.species_data["location"],
        "compartment",
        ["nucleus", "cytoplasm"],
    )
    assert len(filtered) == 4
    assert all(s_id in test_species for s_id in filtered)

    # Test filtering with negation
    filtered = find_species_with_attribute(
        sbml_dfs_with_test_data.species_data["location"],
        "compartment",
        "nucleus",
        negate=True,
    )
    assert len(filtered) == 3
    assert all(s_id in test_species for s_id in filtered)

    # Test filtering boolean values
    filtered = find_species_with_attribute(
        sbml_dfs_with_test_data.species_data["expression"], "is_expressed", True
    )
    assert len(filtered) == 3
    assert all(s_id in test_species for s_id in filtered)

    # Test filtering numeric values
    filtered = find_species_with_attribute(
        sbml_dfs_with_test_data.species_data["location"], "confidence", 0.9
    )
    assert len(filtered) == 1
    assert all(s_id in test_species for s_id in filtered)


def test_filter_species_by_attribute(sbml_dfs_with_test_data):
    """Test the filter_species_by_attribute function."""
    # Get the first 5 species IDs for reference
    test_species = list(sbml_dfs_with_test_data.species.index[:5])
    original_species_count = len(sbml_dfs_with_test_data.species)

    # Test filtering in place - should remove species in nucleus
    result = filter_species_by_attribute(
        sbml_dfs_with_test_data, "location", "compartment", "nucleus"
    )
    assert result is None
    # Should have removed the nucleus species from the test set
    assert len(sbml_dfs_with_test_data.species) == original_species_count - 2
    # Check that species in nucleus were removed
    remaining_test_species = [
        s for s in test_species if s in sbml_dfs_with_test_data.species.index
    ]
    assert (
        len(remaining_test_species) == 3
    )  # Should have 3 test species left (cytoplasm, membrane, cytoplasm)

    # Test filtering with new object - should remove expressed species
    sbml_dfs_copy = copy.deepcopy(sbml_dfs_with_test_data)

    # Count how many species are expressed in our test data
    expressed_count = sum(
        sbml_dfs_copy.species_data["expression"]["is_expressed"].iloc[:5]
    )

    filtered_sbml_dfs = filter_species_by_attribute(
        sbml_dfs_copy, "expression", "is_expressed", True, inplace=False
    )
    # Original should be unchanged
    assert len(sbml_dfs_copy.species) == len(sbml_dfs_with_test_data.species)
    # New object should have removed expressed species from our test set
    assert (
        len(filtered_sbml_dfs.species)
        == len(sbml_dfs_with_test_data.species) - expressed_count
    )

    # Test filtering with invalid table name
    with pytest.raises(ValueError, match="species_data_table .* not found"):
        filter_species_by_attribute(
            sbml_dfs_with_test_data, "nonexistent_table", "compartment", "nucleus"
        )

    # Test filtering with invalid attribute name
    with pytest.raises(ValueError, match="attribute_name .* not found"):
        filter_species_by_attribute(
            sbml_dfs_with_test_data, "location", "nonexistent_attribute", "nucleus"
        )

    # Test filtering with list of values and negation
    # Keep only species NOT in nucleus or cytoplasm (just membrane in our test data)

    VALID_COMPARTMENTS = ["nucleus", "cytoplasm"]
    filtered_sbml_dfs = filter_species_by_attribute(
        sbml_dfs_with_test_data,
        "location",
        "compartment",
        VALID_COMPARTMENTS,
        negate=True,
        inplace=False,
    )

    # Get remaining species from our test set
    remaining_test_species = [
        s for s in test_species if s in filtered_sbml_dfs.species.index
    ]

    assert all(filtered_sbml_dfs.species_data["location"].isin(VALID_COMPARTMENTS))


def test_binarize_species_data():
    # Create test data with different column types
    test_data = pd.DataFrame(
        {
            "bool_col": [True, False, True],
            "binary_int": [1, 0, 1],
            "non_binary_int": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Run the binarization
    binary_df = _binarize_species_data(test_data)

    # Check that only boolean and binary columns were kept
    assert set(binary_df.columns) == {"bool_col", "binary_int"}

    # Check that boolean was converted to int
    assert (
        binary_df["bool_col"].dtype == "int32" or binary_df["bool_col"].dtype == "int64"
    )
    assert binary_df["bool_col"].tolist() == [1, 0, 1]

    # Check that binary int remained the same
    assert binary_df["binary_int"].tolist() == [1, 0, 1]

    # Test with only non-binary columns
    non_binary_data = pd.DataFrame(
        {
            "non_binary_int": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
        }
    )

    # Should raise ValueError when no binary columns are found
    with pytest.raises(ValueError, match="No binary or boolean columns found"):
        _binarize_species_data(non_binary_data)

    # Test with empty DataFrame
    empty_data = pd.DataFrame()
    with pytest.raises(ValueError, match="No binary or boolean columns found"):
        _binarize_species_data(empty_data)


def test_filter_reactions_with_disconnected_cspecies(sbml_dfs):
    # 1. Select first few reactions
    first_reactions = list(sbml_dfs.reactions.index[:5])

    # 2. Find defining species in these reactions
    reaction_species = sbml_dfs_utils.add_sbo_role(sbml_dfs.reaction_species)
    defining_species = (
        reaction_species[reaction_species[SBML_DFS.R_ID].isin(first_reactions)]
        .query("sbo_role == 'DEFINING'")
        # at most 1 record for an sc_id in a reaction (generally true anyways)
        .groupby([SBML_DFS.R_ID, SBML_DFS.SC_ID])
        .first()
        .reset_index(drop=False)
        .groupby(SBML_DFS.R_ID)
        .head(2)  # Take 2 defining species per reaction
    )

    # 3. Get species IDs for these compartmentalized species
    species_info = defining_species.merge(
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]],
        left_on=SBML_DFS.SC_ID,
        right_index=True,
    )

    # Filter out reactions that have less than 2 distinct s_ids (transport reactions)
    valid_reactions = (
        species_info.groupby(SBML_DFS.R_ID)[SBML_DFS.S_ID]
        .nunique()
        .pipe(lambda x: x[x >= 2])
        .index
    )
    species_info = species_info[species_info[SBML_DFS.R_ID].isin(valid_reactions)]

    # 4. Create binary occurrence data where DISJOINT_S_ID is in a different comaprtment from the other top species
    # this should result in removing disconnected_reactions from the sbml_dfs
    DISJOINT_S_ID = species_info.value_counts("s_id").index[0]
    disconnected_reactions = set(
        species_info["r_id"][species_info["s_id"] == DISJOINT_S_ID].tolist()
    )

    # mock data
    mock_species_data = pd.DataFrame({SBML_DFS.S_ID: species_info["s_id"].unique()})
    mock_species_data["compartment_A"] = [
        1 if s_id == DISJOINT_S_ID else 0 for s_id in mock_species_data[SBML_DFS.S_ID]
    ]
    mock_species_data["compartment_B"] = [
        0 if s_id == DISJOINT_S_ID else 1 for s_id in mock_species_data[SBML_DFS.S_ID]
    ]
    mock_species_data.set_index(SBML_DFS.S_ID, inplace=True)

    sbml_dfs.add_species_data("test_data", mock_species_data)

    # Run the filter function
    filtered_sbml_dfs = filter_reactions_with_disconnected_cspecies(
        sbml_dfs, "test_data", inplace=False
    )

    filtered_first_reactions = [
        r for r in first_reactions if r not in filtered_sbml_dfs.reactions.index
    ]

    assert set(filtered_first_reactions) == disconnected_reactions
