import pytest
from pydantic import ValidationError

from napistu.modify.cofactors import CofactorChebiIDs

def test_valid_cofactor_mapping():
    """Test that valid cofactor mapping passes validation."""
    valid_mapping = {
        "ATP": [30616, 15422],
        "ADP": [456216, 16761],
        "water": [15377, 16234],
    }
    
    cofactor_validator = CofactorChebiIDs(cofactor_mapping=valid_mapping)
    
    # Test utility methods
    chebi_map = cofactor_validator.get_chebi_to_cofactor_map()
    assert chebi_map[30616] == "ATP"
    assert chebi_map[15377] == "water"
    
    all_ids = cofactor_validator.get_all_chebi_ids()
    assert len(all_ids) == 6
    assert 30616 in all_ids


def test_duplicate_chebi_ids_fail():
    """Test that duplicate ChEBI IDs across cofactors fail validation."""
    invalid_mapping = {
        "ATP": [30616, 15422],
        "ADP": [456216, 30616],  # 30616 already used in ATP
        "water": [15377],
    }
    
    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)
    
    assert "Duplicate ChEBI IDs found" in str(exc_info.value)
    assert "30616" in str(exc_info.value)


def test_empty_chebi_list_fails():
    """Test that empty ChEBI ID lists fail validation."""
    invalid_mapping = {
        "ATP": [],  # Empty list
        "ADP": [456216],
    }
    
    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)
    
    assert "cannot be empty" in str(exc_info.value)


def test_negative_chebi_id_fails():
    """Test that negative ChEBI IDs fail validation."""
    invalid_mapping = {
        "ATP": [30616, -123],  # Negative ID
    }
    
    with pytest.raises(ValidationError) as exc_info:
        CofactorChebiIDs(cofactor_mapping=invalid_mapping)
    
    assert "must be positive" in str(exc_info.value)