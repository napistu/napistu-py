"""Module for handling ontology aliases and validation."""
from __future__ import annotations

from typing import Dict, Set
from pydantic import BaseModel, field_validator
from napistu.constants import ONTOLOGY_SPECIES_ALIASES, ONTOLOGIES_LIST


class OntologySet(BaseModel):
    """Validates ontology mappings.
    
    This model ensures that:
    1. All keys are valid ontologies from ONTOLOGIES_LIST
    2. The dict maps strings to sets of strings
    3. Values in the sets do not overlap between different keys
    """
    ontologies: Dict[str, Set[str]]
    
    @field_validator("ontologies")
    @classmethod
    def validate_ontologies(cls, v: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Validate the ontology mapping structure."""
        # Check that all keys are valid ontologies
        invalid_ontologies = set(v.keys()) - set(ONTOLOGIES_LIST)
        if invalid_ontologies:
            raise ValueError(
                f"Invalid ontologies: {', '.join(invalid_ontologies)}. "
                f"Must be one of: {', '.join(ONTOLOGIES_LIST)}"
            )
            
        # Check that values don't overlap between keys
        all_values = set()
        for key, values in v.items():
            overlap = values & all_values
            if overlap:
                raise ValueError(
                    f"Found overlapping values {overlap} under multiple ontologies"
                )
            all_values.update(values)
            
        return v
