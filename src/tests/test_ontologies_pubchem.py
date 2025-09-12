import pytest

from napistu.ontologies.constants import PUBCHEM_DEFS
from napistu.ontologies.pubchem import PubChemConnectivityError, map_pubchem_ids


def test_map_pubchem_ids_valid():
    """Test map_pubchem_ids with valid inputs - single batch test."""
    # Single large batch of known valid CIDs
    valid_cids = [
        "2244",
        "5362065",
        "7112",
        "1983",
        "3672",
        "156391",
        "446220",
        "5280656",
    ]  # 8 valid CIDs

    try:
        result = map_pubchem_ids(valid_cids, verbose=False)
    except PubChemConnectivityError as e:
        pytest.skip(f"PubChem connectivity issue: {e}")

    assert len(result) == 8, f"Expected 8 results, got {len(result)}"

    # Check that all have proper structure
    for cid, data in result.items():
        assert PUBCHEM_DEFS.NAME in data, f"Missing name key for CID {cid}"
        assert PUBCHEM_DEFS.SMILES in data, f"Missing smiles key for CID {cid}"
        assert isinstance(
            data[PUBCHEM_DEFS.NAME], str
        ), f"Name should be string for CID {cid}"
        assert isinstance(
            data[PUBCHEM_DEFS.SMILES], str
        ), f"SMILES should be string for CID {cid}"
        assert (
            data[PUBCHEM_DEFS.NAME] != cid
        ), f"CID {cid} should have real name, not just the CID"
        assert data[PUBCHEM_DEFS.SMILES], f"CID {cid} should have SMILES data"

    # Test empty input
    empty_result = map_pubchem_ids([], verbose=False)
    assert empty_result == {}, "Empty input should return empty dict"


def test_map_pubchem_ids_failures():
    """Test map_pubchem_ids with invalid inputs and failure scenarios."""
    # Test 1: Single invalid CID
    invalid_result = map_pubchem_ids(["invalid"], verbose=False, delay=0)
    assert len(invalid_result) == 1, "Should return one result for invalid CID"
    assert (
        invalid_result["invalid"][PUBCHEM_DEFS.NAME] == "invalid"
    ), "Invalid CID should return itself as name"
    assert (
        invalid_result["invalid"][PUBCHEM_DEFS.SMILES] == ""
    ), "Invalid CID should have empty SMILES"

    # Test 2: Mixed valid/invalid (minimal test - just one of each)
    mixed_result = map_pubchem_ids(["2244", "invalid"], verbose=False, delay=0)
    assert len(mixed_result) == 2, "Should return results for both CIDs"
    assert (
        mixed_result["2244"][PUBCHEM_DEFS.NAME] != "2244"
    ), "Valid CID should have real name"
    assert (
        mixed_result["invalid"][PUBCHEM_DEFS.NAME] == "invalid"
    ), "Invalid CID should return itself"
    assert mixed_result["2244"][PUBCHEM_DEFS.SMILES], "Valid CID should have SMILES"
    assert (
        mixed_result["invalid"][PUBCHEM_DEFS.SMILES] == ""
    ), "Invalid CID should have empty SMILES"
