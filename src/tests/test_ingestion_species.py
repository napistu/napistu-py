import pytest

from napistu.ingestion.constants import (
    LATIN_SPECIES_NAMES,
    PSI_MI_INTACT_SPECIES_TO_BASENAME,
)
from napistu.ingestion.species import SpeciesValidator


def test_species_initialization_and_properties():
    """Test species initialization with various inputs and property access."""
    # Test Latin name input
    human_latin = SpeciesValidator(LATIN_SPECIES_NAMES.HOMO_SAPIENS)
    assert human_latin.latin_name == LATIN_SPECIES_NAMES.HOMO_SAPIENS
    assert human_latin.common_name == "human"

    # Test common name input
    mouse_common = SpeciesValidator("mouse")
    assert mouse_common.latin_name == LATIN_SPECIES_NAMES.MUS_MUSCULUS
    assert mouse_common.common_name == "mouse"

    # Test case insensitive input
    human_caps = SpeciesValidator("HUMAN")
    human_mixed = SpeciesValidator("homo sapiens")
    assert (
        human_caps.latin_name
        == human_mixed.latin_name
        == LATIN_SPECIES_NAMES.HOMO_SAPIENS
    )

    # Test string representations
    assert str(human_latin) == "human (Homo sapiens)"
    assert repr(human_latin) == "SpeciesValidator('Homo sapiens')"

    # Test invalid species raises error
    with pytest.raises(ValueError, match="Unknown species"):
        SpeciesValidator("invalid_species")


def test_supported_species_validation():
    """Test validation against lists of supported species."""
    human = SpeciesValidator("human")
    mouse = SpeciesValidator(LATIN_SPECIES_NAMES.MUS_MUSCULUS)
    fly = SpeciesValidator("fly")

    mammal_species = ["human", "mouse", "rat"]
    model_organisms = [
        LATIN_SPECIES_NAMES.HOMO_SAPIENS,
        LATIN_SPECIES_NAMES.MUS_MUSCULUS,
    ]

    # Test positive validation cases
    assert human.validate_against_supported(mammal_species) is True
    assert mouse.validate_against_supported(mammal_species) is True
    assert human.validate_against_supported(model_organisms) is True

    # Test negative validation cases
    assert fly.validate_against_supported(mammal_species) is False
    assert fly.validate_against_supported(model_organisms) is False

    # Test assert_supported success (should not raise)
    human.assert_supported(mammal_species, "genomic_analysis")
    mouse.assert_supported(model_organisms)

    # Test assert_supported failure
    with pytest.raises(ValueError, match="not supported by proteomics"):
        fly.assert_supported(mammal_species, "proteomics")


def test_custom_table_lookup():
    """Test lookup functionality with custom species mapping tables."""
    human = SpeciesValidator("human")
    worm = SpeciesValidator("worm")
    fly = SpeciesValidator("fly")

    # Test lookup with Latin names as keys (default)
    assert (
        human.lookup_custom_value(PSI_MI_INTACT_SPECIES_TO_BASENAME, is_latin=True)
        == "human"
    )
    assert (
        worm.lookup_custom_value(PSI_MI_INTACT_SPECIES_TO_BASENAME, is_latin=True)
        == "caeel"
    )

    # Test lookup with common names as keys
    custom_ids = {"human": "HUMAN_001", "mouse": "MOUSE_001", "yeast": "YEAST_001"}
    human_from_latin = SpeciesValidator(LATIN_SPECIES_NAMES.HOMO_SAPIENS)
    assert (
        human_from_latin.lookup_custom_value(custom_ids, is_latin=False) == "HUMAN_001"
    )

    mouse = SpeciesValidator("mouse")
    assert mouse.lookup_custom_value(custom_ids, is_latin=False) == "MOUSE_001"

    # Test species not found in custom table
    with pytest.raises(ValueError, match="not found in custom table"):
        fly.lookup_custom_value(PSI_MI_INTACT_SPECIES_TO_BASENAME, is_latin=True)

    with pytest.raises(ValueError, match="not found in custom table"):
        fly.lookup_custom_value(custom_ids, is_latin=False)


def test_class_methods_and_utilities():
    """Test class methods and utility functions."""
    # Test get_available_species
    available = SpeciesValidator.get_available_species()

    assert isinstance(available, dict)
    assert "latin_names" in available
    assert "common_names" in available

    # Check that our constants are in the available species
    assert LATIN_SPECIES_NAMES.HOMO_SAPIENS in available["latin_names"]
    assert LATIN_SPECIES_NAMES.MUS_MUSCULUS in available["latin_names"]
    assert "human" in available["common_names"]
    assert "mouse" in available["common_names"]

    # Check that lists have same length (bidirectional mapping)
    assert len(available["latin_names"]) == len(available["common_names"])

    # Verify all species from constants are available
    for latin_name in [
        LATIN_SPECIES_NAMES.HOMO_SAPIENS,
        LATIN_SPECIES_NAMES.MUS_MUSCULUS,
        LATIN_SPECIES_NAMES.SACCHAROMYCES_CEREVISIAE,
    ]:
        assert latin_name in available["latin_names"]
