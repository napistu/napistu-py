import pytest

from napistu.ingestion.constants import (
    LATIN_SPECIES_NAMES,
    PSI_MI_INTACT_SPECIES_TO_BASENAME,
)
from napistu.ingestion.organismal_species import OrganismalSpeciesValidator


def test_species_initialization_and_properties():
    """Test species initialization with various inputs and property access."""
    # Test Latin name input
    human_latin = OrganismalSpeciesValidator(LATIN_SPECIES_NAMES.HOMO_SAPIENS)
    assert human_latin.latin_name == LATIN_SPECIES_NAMES.HOMO_SAPIENS
    assert human_latin.common_name == "human"

    # Test common name input
    mouse_common = OrganismalSpeciesValidator("mouse")
    assert mouse_common.latin_name == LATIN_SPECIES_NAMES.MUS_MUSCULUS
    assert mouse_common.common_name == "mouse"

    # Test case insensitive input
    human_caps = OrganismalSpeciesValidator("HUMAN")
    human_mixed = OrganismalSpeciesValidator("homo sapiens")
    assert (
        human_caps.latin_name
        == human_mixed.latin_name
        == LATIN_SPECIES_NAMES.HOMO_SAPIENS
    )

    # Test string representations
    assert str(human_latin) == "human (Homo sapiens)"
    assert repr(human_latin) == "OrganismalSpeciesValidator('Homo sapiens')"

    # Test invalid species raises error
    with pytest.raises(ValueError, match="Unknown species"):
        OrganismalSpeciesValidator("invalid_species")


def test_supported_species_validation():
    """Test validation against lists of supported species."""
    human = OrganismalSpeciesValidator("human")
    mouse = OrganismalSpeciesValidator(LATIN_SPECIES_NAMES.MUS_MUSCULUS)
    fly = OrganismalSpeciesValidator("fly")

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
    human = OrganismalSpeciesValidator("human")
    worm = OrganismalSpeciesValidator("worm")
    fly = OrganismalSpeciesValidator("fly")

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
    human_from_latin = OrganismalSpeciesValidator(LATIN_SPECIES_NAMES.HOMO_SAPIENS)
    assert (
        human_from_latin.lookup_custom_value(custom_ids, is_latin=False) == "HUMAN_001"
    )

    mouse = OrganismalSpeciesValidator("mouse")
    assert mouse.lookup_custom_value(custom_ids, is_latin=False) == "MOUSE_001"

    # Test species not found in custom table
    with pytest.raises(ValueError, match="not found in custom table"):
        fly.lookup_custom_value(PSI_MI_INTACT_SPECIES_TO_BASENAME, is_latin=True)

    with pytest.raises(ValueError, match="not found in custom table"):
        fly.lookup_custom_value(custom_ids, is_latin=False)


def test_class_methods_and_utilities():
    """Test class methods and utility functions."""
    # Test get_available_species
    available = OrganismalSpeciesValidator.get_available_species()

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


def test_ensure_organismal_species_validator():
    """Test the OrganismalSpeciesValidator.ensure class method."""

    # Test with string input
    validator = OrganismalSpeciesValidator.ensure("human")
    assert isinstance(validator, OrganismalSpeciesValidator)
    assert validator.latin_name == "Homo sapiens"
    assert validator.common_name == "human"

    # Test with existing OrganismalSpeciesValidator
    existing_validator = OrganismalSpeciesValidator("mouse")
    validator = OrganismalSpeciesValidator.ensure(existing_validator)
    assert validator is existing_validator
    assert validator.latin_name == "Mus musculus"

    # Test with invalid input
    with pytest.raises(
        ValueError, match="must be a string or OrganismalSpeciesValidator"
    ):
        OrganismalSpeciesValidator.ensure(123)

    with pytest.raises(
        ValueError, match="must be a string or OrganismalSpeciesValidator"
    ):
        OrganismalSpeciesValidator.ensure(None)
