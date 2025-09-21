import pandas as pd
import pytest
from pydantic import ValidationError

from napistu.constants import SBML_DFS
from napistu.modify.cofactors import (
    CofactorChebiIDs,
    _filter_one_reactions_cofactors,
    drop_cofactors,
    identify_cofactors,
)
from napistu.modify.constants import COFACTOR_DEFS, COFACTORS


def test_valid_cofactor_mapping():
    """Test that valid cofactor mapping passes validation."""
    valid_mapping = {
        COFACTORS.ATP: [30616, 15422],
        COFACTORS.ADP: [456216, 16761],
        COFACTORS.WATER: [15377, 16234],
    }

    cofactor_validator = CofactorChebiIDs(cofactor_mapping=valid_mapping)

    # Test utility methods
    chebi_map = cofactor_validator.get_chebi_to_cofactor_map()
    assert chebi_map[30616] == COFACTORS.ATP
    assert chebi_map[15377] == COFACTORS.WATER

    all_ids = cofactor_validator.get_all_chebi_ids()
    assert len(all_ids) == 6
    assert 30616 in all_ids


def test_duplicate_chebi_ids_fail():
    """Test that duplicate ChEBI IDs across cofactors fail validation."""
    invalid_mapping = {
        COFACTORS.ATP: [30616, 15422],
        COFACTORS.ADP: [456216, 30616],  # 30616 already used in ATP
        COFACTORS.WATER: [15377],
    }

    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)

    assert "Duplicate ChEBI IDs found" in str(exc_info.value)
    assert "30616" in str(exc_info.value)


def test_empty_chebi_list_fails():
    """Test that empty ChEBI ID lists fail validation."""
    invalid_mapping = {
        COFACTORS.ATP: [],  # Empty list
        COFACTORS.ADP: [456216],
    }

    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)

    assert "cannot be empty" in str(exc_info.value)


def test_negative_chebi_id_fails():
    """Test that negative ChEBI IDs fail validation."""
    invalid_mapping = {
        COFACTORS.ATP: [30616, -123],  # Negative ID
    }

    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)

    assert "must be positive" in str(exc_info.value)


def test_filter_one_reactions_cofactors():
    """Test _filter_one_reactions_cofactors with various filter rule scenarios."""

    # Create sample reaction species data
    def create_reaction_species(cofactors, stoichiometries):
        """Helper to create test reaction species DataFrame."""
        data = []
        for i, (cofactor, stoich) in enumerate(zip(cofactors, stoichiometries)):
            data.append(
                {
                    SBML_DFS.RSC_ID: f"RSC{i:05d}",
                    SBML_DFS.R_ID: "R00001",
                    SBML_DFS.SC_ID: f"SC{i:05d}",
                    SBML_DFS.STOICHIOMETRY: stoich,
                    SBML_DFS.SBO_TERM: "SBO:0000010",
                    COFACTOR_DEFS.COFACTOR: cofactor,
                }
            )
        return pd.DataFrame(data).set_index(SBML_DFS.RSC_ID)

    # Test 1: if_all rule - all required cofactors present
    reaction_species = create_reaction_species(
        [COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4], [-1, 1, 1]
    )
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(
        reaction_species, "ATP hydrolysis", filter_rule
    )
    assert result is not None, "Should find cofactors when all if_all species present"
    assert len(result) == 2, "Should return 2 species (ATP and ADP)"
    assert all(
        result == "ATP hydrolysis"
    ), "All results should have correct filter reason"

    # Test 2: if_all rule - missing required cofactor
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.PO4], [-1, 1])
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(
        reaction_species, "ATP hydrolysis", filter_rule
    )
    assert result is None, "Should return None when required cofactor missing"

    # Test 3: except_any rule - exception present
    reaction_species = create_reaction_species(
        [COFACTORS.ATP, COFACTORS.ADP, COFACTORS.AMP], [-1, 1, 0.5]
    )
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "ATP hydrolysis", filter_rule
    )
    assert result is None, "Should return None when exception species present"

    # Test 4: except_any rule - no exception present
    reaction_species = create_reaction_species(
        [COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4], [-1, 1, 1]
    )
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "ATP hydrolysis", filter_rule
    )
    assert result is not None, "Should find cofactors when no exception present"
    assert len(result) == 2, "Should return 2 species"

    # Test 5: as_substrate rule - required substrate present
    reaction_species = create_reaction_species(
        [COFACTORS.NADH, COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [-1, 1, 1]
    )
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADH, COFACTORS.NAD_PLUS],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "NADH oxidation", filter_rule
    )
    assert result is not None, "Should find cofactors when required substrate present"
    assert len(result) == 2, "Should return 2 species"

    # Test 6: as_substrate rule - required substrate not a substrate
    reaction_species = create_reaction_species(
        [COFACTORS.NADH, COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [1, -1, -1]
    )  # NADH as product
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADH, COFACTORS.NAD_PLUS],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "NADH oxidation", filter_rule
    )
    assert (
        result is None
    ), "Should return None when required substrate is actually a product"

    # Test 7: as_substrate rule - required substrate missing
    reaction_species = create_reaction_species(
        [COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [1, 1]
    )
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NAD_PLUS],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "NADH oxidation", filter_rule
    )
    assert result is None, "Should return None when required substrate not present"

    # Test 8: Complex rule - all conditions met
    reaction_species = create_reaction_species(
        [COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4, COFACTORS.WATER], [-1, 1, 1, -1]
    )
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP],
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.ATP],
    }
    result = _filter_one_reactions_cofactors(
        reaction_species, "ATP hydrolysis", filter_rule
    )
    assert result is not None, "Should find cofactors when all complex conditions met"
    assert len(result) == 2, "Should return 2 species (ATP and ADP)"

    # Test 9: Zero stoichiometry (should be filtered out upstream, but test robustness)
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP], [0, 0])
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(reaction_species, "test", filter_rule)
    assert result is not None, "Should still work with zero stoichiometry species"


def test_identify_cofactors(sbml_dfs):
    """Test identify_cofactors function with sbml_dfs fixture."""

    # Validate initial dimensions
    assert sbml_dfs.reaction_species.shape == (
        32,
        4,
    ), "Initial reaction_species should have 32 rows and 4 columns"
    assert sbml_dfs.species.shape == (
        18,
        3,
    ), "Initial species should have 18 rows and 3 columns"
    assert sbml_dfs.compartmentalized_species.shape == (
        23,
        4,
    ), "Initial compartmentalized_species should have 23 rows and 4 columns"

    # Test identify_cofactors
    cofactor_series = identify_cofactors(sbml_dfs)

    # Validate results
    assert isinstance(
        cofactor_series, pd.Series
    ), "identify_cofactors should return a pandas Series"
    assert cofactor_series.shape == (20,), "Should identify exactly 20 cofactors"
    assert len(cofactor_series) == 20, "Number of cofactors found should be 20"
    assert (
        cofactor_series.name == COFACTOR_DEFS.FILTER_REASON
    ), "Series should have correct name"

    # Validate filter reason counts
    value_counts = cofactor_series.value_counts()
    assert value_counts["CO2"] == 7, "Should identify 7 CO2 cofactors"
    assert value_counts["HCO3-"] == 4, "Should identify 4 HCO3- cofactors"
    assert value_counts["H+"] == 4, "Should identify 4 H+ cofactors"
    assert value_counts["Water"] == 2, "Should identify 2 Water cofactors"
    assert (
        value_counts["NADH H- donation"] == 2
    ), "Should identify 2 NADH H- donation cofactors"
    assert value_counts["O2"] == 1, "Should identify 1 O2 cofactor"

    # Validate structure
    assert all(
        isinstance(reason, str) for reason in cofactor_series.values
    ), "All filter reasons should be strings"
    assert all(
        rsc_id in sbml_dfs.reaction_species.index for rsc_id in cofactor_series.index
    ), "All rsc_ids should exist in reaction_species"

    # Validate sample rsc_ids exist
    expected_sample_ids = [
        "speciesreference_1237038_input_111627",
        "speciesreference_1237038_output_425425",
        "speciesreference_1237042_input_1237009",
        "speciesreference_1237042_output_113528",
        "speciesreference_1237047_input_109276",
    ]
    for rsc_id in expected_sample_ids:
        assert (
            rsc_id in cofactor_series.index
        ), f"Expected rsc_id {rsc_id} should be in cofactor series"


def test_drop_cofactors(sbml_dfs):
    """Test drop_cofactors function with sbml_dfs fixture."""

    # Validate initial dimensions
    assert sbml_dfs.reaction_species.shape == (
        32,
        4,
    ), "Initial reaction_species should have 32 rows and 4 columns"
    assert sbml_dfs.species.shape == (
        18,
        3,
    ), "Initial species should have 18 rows and 3 columns"
    assert sbml_dfs.compartmentalized_species.shape == (
        23,
        4,
    ), "Initial compartmentalized_species should have 23 rows and 4 columns"

    # Test drop_cofactors
    sbml_dfs_filtered = drop_cofactors(sbml_dfs, verbose=False)

    # Validate filtered dimensions
    assert sbml_dfs_filtered.reaction_species.shape == (
        12,
        4,
    ), "Filtered reaction_species should have 12 rows and 4 columns"
    assert sbml_dfs_filtered.species.shape == (
        18 - 7,
        3,
    ), "Filtered species should drop 7 species which were only present as cofactors"
    assert sbml_dfs_filtered.compartmentalized_species.shape == (
        23 - 11,
        4,
    ), "Filtered compartmentalized_species should drop 11 cspecies which only acted as cofactors"

    # Calculate and validate counts
    original_count = sbml_dfs.reaction_species.shape[0]
    filtered_count = sbml_dfs_filtered.reaction_species.shape[0]
    removed_count = original_count - filtered_count

    assert original_count == 32, "Original reaction species count should be 32"
    assert filtered_count == 12, "Filtered reaction species count should be 12"
    assert removed_count == 20, "Removed reaction species count should be exactly 20"

    # Basic validation
    assert isinstance(
        sbml_dfs_filtered, type(sbml_dfs)
    ), "drop_cofactors should return same type as input"

    # Validate that filtered reaction_species is a subset of original
    assert all(
        rsc_id in sbml_dfs.reaction_species.index
        for rsc_id in sbml_dfs_filtered.reaction_species.index
    ), "All remaining rsc_ids should exist in original"

    sbml_dfs_filtered.validate()
