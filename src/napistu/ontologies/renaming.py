"""Module for handling ontology aliases and validation."""

from __future__ import annotations

from typing import Dict, Set
import logging
from pydantic import BaseModel, field_validator
from napistu.constants import (
    ONTOLOGY_SPECIES_ALIASES,
    ONTOLOGIES_LIST,
    IDENTIFIERS,
    SBML_DFS,
)
import pandas as pd
from napistu import identifiers, sbml_dfs_core

logger = logging.getLogger(__name__)


def rename_species_ontologies(
    sbml_dfs: sbml_dfs_core.SBML_dfs, aliases=ONTOLOGY_SPECIES_ALIASES
):
    """
    Rename ontologies in the species identifiers table of an SBML_dfs object using provided aliases.

    This function updates the ontology names in the species identifiers of the given SBML_dfs object
    according to the provided alias mapping. It validates the alias mapping, logs which ontologies will be updated,
    and replaces any matching aliases in the species identifiers with their canonical ontology names.

    Parameters
    ----------
    sbml_dfs : napistu.sbml_dfs_core.SBML_dfs
        The SBML_dfs object whose species table will be updated in-place.
    aliases : dict[str, set[str]], optional
        Dictionary mapping canonical ontology names to sets of their aliases. By default, uses ONTOLOGY_SPECIES_ALIASES.
        All keys must be valid ontologies from ONTOLOGIES_LIST. Values must not overlap between keys or with keys themselves.

    Returns
    -------
    None
        The function updates sbml_dfs.species in-place and does not return a value.

    Raises
    ------
    ValueError
        If the alias mapping is invalid (e.g., keys not in ONTOLOGIES_LIST, overlapping values, or values used as keys),
        or if there is no overlap between the provided aliases and the ontologies present in the species identifiers.

    Examples
    --------
    >>> from napistu.ontologies.renaming import rename_species_ontologies
    >>> sbml_dfs = ...  # an SBML_dfs object
    >>> aliases = {"ncbi_entrez_gene": {"ncbigene", "ncbi_gene"}, "uniprot": {"uniprot_id"}}
    >>> rename_species_ontologies(sbml_dfs, aliases)
    """

    species_identifiers = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)

    aliases = OntologySet(ontologies=aliases).ontologies
    alias_mapping = _create_alias_mapping(aliases)

    _log_ontology_updates(alias_mapping, set(species_identifiers[IDENTIFIERS.ONTOLOGY]))

    species_identifiers[IDENTIFIERS.ONTOLOGY] = species_identifiers[
        IDENTIFIERS.ONTOLOGY
    ].map(lambda x: alias_mapping.get(x, x))

    species_identifiers = identifiers.df_to_identifiers(species_identifiers)

    updated_species = sbml_dfs.species.drop(SBML_DFS.S_IDENTIFIERS, axis=1).join(
        pd.DataFrame(species_identifiers)
    )
    # fill missing attributes with empty Identifiers
    updated_species[SBML_DFS.S_IDENTIFIERS] = updated_species[
        SBML_DFS.S_IDENTIFIERS
    ].fillna(identifiers.Identifiers([]))

    setattr(sbml_dfs, "species", updated_species)


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


def _create_alias_mapping(ontology_dict: Dict[str, Set[str]]) -> Dict[str, str]:
    """Create a mapping from aliases to canonical ontology names.

    Only creates mappings for the aliases specified in the input dictionary.
    Does not include mappings for canonical names to themselves.

    Parameters
    ----------
    ontology_dict : Dict[str, Set[str]]
        Dictionary mapping ontologies to their aliases

    Returns
    -------
    Dict[str, str]
        Dictionary mapping each alias to its canonical ontology name
    """
    mapping = {}
    for ontology, aliases in ontology_dict.items():
        # Only map aliases to canonical names
        for alias in aliases:
            mapping[alias] = ontology
    return mapping


def _log_ontology_updates(
    alias_mapping: Dict[str, str], species_ontologies: Set[str]
) -> None:
    """Log which ontology aliases will be updated.

    Parameters
    ----------
    alias_mapping : Dict[str, str]
        Dictionary mapping old ontology names to new ones
    species_ontologies : Set[str]
        Set of ontology names present in the species identifiers

    Raises
    ------
    ValueError
        If there is no overlap between the aliases and species ontologies
    """
    # Find which aliases are present in the species data
    updatable_aliases = set(alias_mapping.keys()) & species_ontologies
    if not updatable_aliases:
        raise ValueError(
            "The set of ontologies in the species identifiers and aliases do not overlap. "
            "Please provide an updated aliases dict."
        )

    # Log which ontologies will be updated
    updates = [f"{old} -> {alias_mapping[old]}" for old in updatable_aliases]
    logger.info(f"Updating the following ontologies: {', '.join(updates)}")
