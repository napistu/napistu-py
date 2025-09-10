import pytest
from pydantic import ValidationError

import pandas as pd

from napistu.modify.cofactors import CofactorChebiIDs, _filter_one_reactions_cofactors
from napistu.constants import SBML_DFS
from napistu.modify.constants import (
    COFACTORS,
    COFACTOR_DEFS
)


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
            data.append({
                SBML_DFS.RSC_ID: f"RSC{i:05d}",
                SBML_DFS.R_ID: "R00001",
                SBML_DFS.SC_ID: f"SC{i:05d}",
                SBML_DFS.STOICHIOMETRY: stoich,
                SBML_DFS.SBO_TERM: "SBO:0000010",
                COFACTOR_DEFS.COFACTOR: cofactor
            })
        return pd.DataFrame(data).set_index(SBML_DFS.RSC_ID)
    
    # Test 1: if_all rule - all required cofactors present
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4], [-1, 1, 1])
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(reaction_species, "ATP hydrolysis", filter_rule)
    assert result is not None, "Should find cofactors when all if_all species present"
    assert len(result) == 2, "Should return 2 species (ATP and ADP)"
    assert all(result == "ATP hydrolysis"), "All results should have correct filter reason"
    
    # Test 2: if_all rule - missing required cofactor
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.PO4], [-1, 1])
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(reaction_species, "ATP hydrolysis", filter_rule)
    assert result is None, "Should return None when required cofactor missing"
    
    # Test 3: except_any rule - exception present
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP, COFACTORS.AMP], [-1, 1, 0.5])
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP], 
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "ATP hydrolysis", filter_rule)
    assert result is None, "Should return None when exception species present"
    
    # Test 4: except_any rule - no exception present
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4], [-1, 1, 1])
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP], 
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "ATP hydrolysis", filter_rule)
    assert result is not None, "Should find cofactors when no exception present"
    assert len(result) == 2, "Should return 2 species"
    
    # Test 5: as_substrate rule - required substrate present
    reaction_species = create_reaction_species([COFACTORS.NADH, COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [-1, 1, 1])
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADH, COFACTORS.NAD_PLUS], 
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "NADH oxidation", filter_rule)
    assert result is not None, "Should find cofactors when required substrate present"
    assert len(result) == 2, "Should return 2 species"
    
    # Test 6: as_substrate rule - required substrate not a substrate
    reaction_species = create_reaction_species([COFACTORS.NADH, COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [1, -1, -1])  # NADH as product
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NADH, COFACTORS.NAD_PLUS], 
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "NADH oxidation", filter_rule)
    assert result is None, "Should return None when required substrate is actually a product"
    
    # Test 7: as_substrate rule - required substrate missing
    reaction_species = create_reaction_species([COFACTORS.NAD_PLUS, COFACTORS.H_PLUS], [1, 1])
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.NAD_PLUS], 
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.NADH]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "NADH oxidation", filter_rule)
    assert result is None, "Should return None when required substrate not present"
    
    # Test 8: Complex rule - all conditions met
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP, COFACTORS.PO4, COFACTORS.WATER], [-1, 1, 1, -1])
    filter_rule = {
        COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP], 
        COFACTOR_DEFS.EXCEPT_ANY: [COFACTORS.AMP],
        COFACTOR_DEFS.AS_SUBSTRATE: [COFACTORS.ATP]
    }
    result = _filter_one_reactions_cofactors(reaction_species, "ATP hydrolysis", filter_rule)
    assert result is not None, "Should find cofactors when all complex conditions met"
    assert len(result) == 2, "Should return 2 species (ATP and ADP)"
    
    # Test 9: Zero stoichiometry (should be filtered out upstream, but test robustness)
    reaction_species = create_reaction_species([COFACTORS.ATP, COFACTORS.ADP], [0, 0])
    filter_rule = {COFACTOR_DEFS.IF_ALL: [COFACTORS.ATP, COFACTORS.ADP]}
    result = _filter_one_reactions_cofactors(reaction_species, "test", filter_rule)
    assert result is not None, "Should still work with zero stoichiometry species"