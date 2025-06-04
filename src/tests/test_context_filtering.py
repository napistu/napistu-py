from __future__ import annotations

import copy
import pytest
import pandas as pd
from napistu.context.filtering import (
    filter_species_by_attribute,
    find_species_with_attribute,
    _binarize_species_data,
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


def test_potential_reaction_contexts_in_evaluate_reaction_context_sufficiency():
    """Test potential reaction contexts in evaluate_reaction_context_sufficiency()"""

    reaction_w_regulators = pd.DataFrame(
        {
            "gene": ["A", "B", "C", "D", "E", "F", "G"],
            "stoi": [-1, -1, 1, 1, 0, 0, 0],
            "role": [
                "substrate",
                "substrate",
                "product",
                "product",
                "catalyst",
                "catalyst",
                "activator",
            ],
        }
    )

    # missing substrates -> inoperable
    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators,
        reaction_w_regulators.iloc[1:],  # Remove first row (slice(-1) equivalent)
    )
    assert result is None

    # missing one enzyme -> operable
    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators,
        reaction_w_regulators.drop(
            reaction_w_regulators.index[4]
        ),  # Remove 5th row (slice(-5) equivalent)
    )
    assert len(result) == 6

    # missing one product -> inoperable
    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators,
        reaction_w_regulators.drop(
            reaction_w_regulators.index[2]
        ),  # Remove 3rd row (slice(-3) equivalent)
    )
    assert result is None

    # missing all enzymes -> inoperable
    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators,
        reaction_w_regulators.iloc[
            0:4
        ],  # Keep only first 4 rows (slice(1:4) equivalent)
    )
    assert result is None

    # missing regulators -> operable
    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators,
        reaction_w_regulators.drop(
            reaction_w_regulators.index[6]
        ),  # Remove 7th row (slice(-7) equivalent)
    )
    assert len(result) == 6

    weird_reaction_w_interactors = pd.DataFrame(
        {
            "gene": ["A", "B", "C"],
            "stoi": [-1, 1, 1],
            "role": ["interactor", "interactor", "interactor"],
        }
    )

    # throw a warning for a weird interaction format
    with pytest.warns(UserWarning, match="interactor"):
        evaluate_reaction_context_sufficiency(
            weird_reaction_w_interactors, weird_reaction_w_interactors
        )

    reaction_w_interactors = pd.DataFrame(
        {"gene": ["A", "B"], "stoi": [-1, 1], "role": ["interactor", "interactor"]}
    )

    result = evaluate_reaction_context_sufficiency(
        reaction_w_regulators, reaction_w_regulators.iloc[0:4]  # Keep only first 4 rows
    )
    assert result is None

    # missing one interactor
    result = evaluate_reaction_context_sufficiency(
        reaction_w_interactors,
        reaction_w_interactors.iloc[1:],  # Remove first row (slice(-1) equivalent)
    )
    assert result is None

    # identify function looks good
    result = evaluate_reaction_context_sufficiency(
        reaction_w_interactors, reaction_w_interactors
    )
    assert len(result) == 2
