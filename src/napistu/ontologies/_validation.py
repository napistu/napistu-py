"""Internal validation models for the ontologies subpackage."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class OntologySet(BaseModel):
    """Validate ontology mappings.

    This model ensures that:
    1. All keys are valid ontologies from ONTOLOGIES_LIST
    2. The dict maps strings to sets of strings
    3. Values in the sets do not overlap between different keys
    4. Values in the sets are not also used as keys

    Attributes
    ----------
    ontologies : Dict[str, Set[str]]
        Dictionary mapping ontology names to sets of their aliases
    """

    ontologies: Dict[str, Set[str]]

    @field_validator("ontologies")
    @classmethod
    def validate_ontologies(cls, v: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Validate the ontology mapping structure.

        Parameters
        ----------
        v : Dict[str, Set[str]]
            Dictionary mapping ontology names to sets of their aliases

        Returns
        -------
        Dict[str, Set[str]]
            The validated ontology mapping dictionary

        Raises
        ------
        ValueError
            If any keys are not valid ontologies from ONTOLOGIES_LIST
            If any values overlap between different ontologies
            If any values are also used as ontology keys
        """
        from napistu.constants import ONTOLOGIES_LIST

        # Check that all keys are valid ontologies
        invalid_ontologies = set(v.keys()) - set(ONTOLOGIES_LIST)
        if invalid_ontologies:
            raise ValueError(
                f"Invalid ontologies: {', '.join(invalid_ontologies)}. "
                f"Must be one of: {', '.join(ONTOLOGIES_LIST)}"
            )

        # Check that values don't overlap between keys and aren't used as keys
        all_values = set()
        keys = set(v.keys())
        for key, values in v.items():
            # Check for overlap with other values
            overlap = values & all_values
            if overlap:
                raise ValueError(
                    f"Found overlapping values {overlap} under multiple ontologies"
                )
            # Check for overlap with keys
            key_overlap = values & keys
            if key_overlap:
                raise ValueError(
                    f"Found values {key_overlap} that are also used as ontology keys"
                )
            all_values.update(values)

        return v


class GenodexitoConfig(BaseModel):
    """Configuration for Genodexito with validation.

    Attributes
    ----------
    organismal_species: str
        Species name to use for mapping
    preferred_method: str
        Which mapping method to try first
    allow_fallback: bool
        Whether to allow fallback to other method
    r_paths: Optional[List[str]]
        Optional paths to R libraries
    test_mode: bool
        Whether to limit queries for testing
    """

    organismal_species: str = Field(
        default="Homo sapiens", description="Species name to use"
    )
    preferred_method: str = Field(
        description="Which mapping method to try first",
    )
    allow_fallback: bool = Field(
        default=True, description="Whether to allow fallback to other method"
    )
    r_paths: Optional[List[str]] = Field(
        default=None, description="Optional paths to R libraries"
    )
    test_mode: bool = Field(
        default=False, description="Whether to limit queries for testing"
    )

    @field_validator("preferred_method")
    @classmethod
    def validate_preferred_method(cls, v: str) -> str:
        """Validate that preferred_method is one of the allowed values."""
        from napistu.ontologies.constants import GENODEXITO_DEFS

        if v not in {GENODEXITO_DEFS.BIOCONDUCTOR, GENODEXITO_DEFS.PYTHON}:
            raise ValueError(
                f"Invalid preferred_method: {v}. "
                f"Must be one of: {GENODEXITO_DEFS.BIOCONDUCTOR}, {GENODEXITO_DEFS.PYTHON}"
            )
        return v

    @field_validator("r_paths")
    @classmethod
    def validate_r_paths(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that r_paths contains only strings."""
        if v is not None and not all(isinstance(path, str) for path in v):
            raise ValueError("All elements in r_paths must be strings")
        return v


class SpeciesTypeOntologyMapping(BaseModel):
    """Validate species type to ontology mappings.

    Ensures that:
    1. All ontologies in the values are valid (from ONTOLOGIES)
    2. All ontologies are unique across all species types (no duplicates)
    3. The mapping structure is correct

    Attributes
    ----------
    mappings : Dict[str, List[str]]
        Dictionary mapping species type names to lists of ontologies
    """

    mappings: Dict[str, List[str]]

    @field_validator("mappings")
    @classmethod
    def validate_mappings(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate the species type ontology mappings.

        Parameters
        ----------
        v : Dict[str, List[str]]
            Dictionary mapping species type names to lists of ontologies

        Returns
        -------
        Dict[str, List[str]]
            The validated mapping dictionary

        Raises
        ------
        ValueError
            If any ontologies are invalid or duplicated across species types
        """
        # Get all valid ontologies from the ONTOLOGIES constants
        from napistu.constants import ONTOLOGIES

        valid_ontologies = {
            getattr(ONTOLOGIES, attr)
            for attr in dir(ONTOLOGIES)
            if not attr.startswith("_")
        }

        # Collect all ontologies and check for validity
        all_ontologies = []
        invalid_ontologies = set()

        for ontology_list in v.values():
            for ontology in ontology_list:
                all_ontologies.append(ontology)
                if ontology not in valid_ontologies:
                    invalid_ontologies.add(ontology)

        # Check for invalid ontologies
        if invalid_ontologies:
            raise ValueError(
                f"Invalid ontologies found: {', '.join(invalid_ontologies)}. "
                f"Must be valid ontologies from the ONTOLOGIES constants."
            )

        # Check for duplicates across species types
        seen_ontologies = set()
        duplicate_ontologies = set()

        for ontology_list in v.values():
            for ontology in ontology_list:
                if ontology in seen_ontologies:
                    duplicate_ontologies.add(ontology)
                else:
                    seen_ontologies.add(ontology)

        if duplicate_ontologies:
            # Find which species types have the duplicates
            duplicate_mappings = []
            for dup_ont in duplicate_ontologies:
                species_with_dup = [sp for sp, onts in v.items() if dup_ont in onts]
                duplicate_mappings.append(f"{dup_ont} in {', '.join(species_with_dup)}")

            raise ValueError(
                f"Duplicate ontologies found across species types: {'; '.join(duplicate_mappings)}. "
                "Each ontology can only belong to one species type."
            )

        return v

    def create_ontology_to_species_type_mapping(self) -> Dict[str, str]:
        """Create a flattened mapping from ontology to species type.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping each ontology to its species type
        """
        ontology_to_species_type = {}
        for species_type, ontologies in self.mappings.items():
            for ontology in ontologies:
                ontology_to_species_type[ontology] = species_type
        return ontology_to_species_type
