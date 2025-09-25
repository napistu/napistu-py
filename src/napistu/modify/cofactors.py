"""Module to detect and remove cofactors from a pathway model"""

import logging
from typing import Dict, List, Set

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from napistu import utils
from napistu.constants import (
    BQB,
    IDENTIFIERS,
    ONTOLOGIES,
    SBML_DFS,
)
from napistu.modify.constants import (
    COFACTOR_CHEBI_IDS,
    COFACTOR_DEFS,
    COFACTOR_SCHEMA,
)
from napistu.sbml_dfs_core import SBML_dfs

logger = logging.getLogger(__name__)


def drop_cofactors(
    sbml_dfs: SBML_dfs,
    cofactor_chebi_ids: dict = COFACTOR_CHEBI_IDS,
    cofactor_schema: dict = COFACTOR_SCHEMA,
    verbose: bool = True,
) -> SBML_dfs:
    """
    Drop Cofactors

    Remove reaction species when they are acting as cofactors

    Parameters:
    ----------
    sbml_dfs: SBML_dfs
        A pathway model
    cofactor_chebi_ids: dict
        Dictionary mapping cofactor names to lists of ChEBI IDs
    cofactor_schema: dict
        Dictionary mapping filter names to cofactor filter rules
    verbose: bool
        Whether to print verbose output

    Returns:
    ----------
    sbml_dfs (SBML_dfs):
        A pathway model with some reaction species filtered
    """

    all_cofactors = identify_cofactors(sbml_dfs, cofactor_chebi_ids, cofactor_schema)

    if verbose:
        logger.info(
            f"{all_cofactors.shape[0]} of {sbml_dfs.reaction_species.shape[0]}"
            f" reaction species will be filtered as cofactors"
        )

        styled_df = all_cofactors.value_counts().to_frame()
        utils.show(styled_df)

    # remove reaction species and other impacted entities (e.g., remove water everywhere)
    sbml_dfs_working = sbml_dfs.copy()
    sbml_dfs_working.remove_entities(
        # special behavior where literal cleanup of reactions based on reaction_species is allowed
        # normally, removing substrates/products would remove the reaction
        "cofactors",
        all_cofactors.index.tolist(),
        remove_references=True,
    )

    return sbml_dfs_working


def identify_cofactors(
    sbml_dfs: SBML_dfs,
    cofactor_chebi_ids: dict = COFACTOR_CHEBI_IDS,
    cofactor_schema: dict = COFACTOR_SCHEMA,
) -> pd.Series:
    """
    Identify Cofactors

    Find cofactors which are playing a supporting role in a reaction (e.g., ATP -> ADP or water).

    Parameters:
    ----------
    sbml_dfs: SBML_dfs
        A pathway model

    Returns:
    ----------
    pd.Series with index of rsc_ids and values containing the reason why a reaction species is a cofactor

    """

    # validate inputs up-front
    _ = CofactorChebiIDs(cofactor_mapping=cofactor_chebi_ids)
    _ = CofactorSchema(schema_mapping=cofactor_schema)

    cofactor_species = _find_cofactor_species(sbml_dfs, cofactor_chebi_ids)
    cofactor_rscspecies = _map_cofactors_to_reaction_species(sbml_dfs, cofactor_species)
    return _apply_cofactor_filters(cofactor_rscspecies, cofactor_schema)


# utils


def _apply_cofactor_filters(
    cofactor_rscspecies: pd.DataFrame, cofactor_schema: dict = COFACTOR_SCHEMA
) -> pd.Series:
    """
    Apply Cofactor Filters to Reactions

    Apply cofactor filtering rules to identify reaction species that should be
    removed as cofactors based on the defined schema.

    Parameters
    ----------
    cofactor_rscspecies: pd.DataFrame
        Reaction species DataFrame with cofactor annotations
    cofactor_schema: dict, optional
        Dictionary mapping filter names to cofactor filter rules.
        Defaults to COFACTOR_SCHEMA.

    Returns
    -------
    pd.Series
        Series with index of rsc_ids and values containing the reason why
        a reaction species is a cofactor
    """

    _ = CofactorSchema(schema_mapping=cofactor_schema)

    # Group by reaction and apply filters to each group
    filtered_results = cofactor_rscspecies.groupby(SBML_DFS.R_ID).apply(
        lambda group: _apply_filters_to_reaction(group, cofactor_schema),
        include_groups=False,
    )

    # filtered_results is a Series with MultiIndex (r_id, rsc_id) -> filter_reason
    # Reset index to get rsc_id as the main index
    if len(filtered_results) > 0:
        return filtered_results.reset_index(level=0, drop=True)
    else:
        return pd.Series([], dtype=str, name=COFACTOR_DEFS.FILTER_REASON)


def _apply_filters_to_reaction(
    one_rxns_species: pd.DataFrame, cofactor_schema: dict = COFACTOR_SCHEMA
) -> pd.Series:
    """
    Apply All Filters to Single Reaction

    Apply all cofactor filtering rules to a single reaction's species
    and return the combined results.

    Parameters
    ----------
    one_rxns_species: pd.DataFrame
        Reaction species DataFrame for a single reaction with cofactor annotations
    cofactor_schema: dict, optional
        Dictionary mapping filter names to cofactor filter rules.
        Defaults to COFACTOR_SCHEMA.

    Returns
    -------
    pd.Series
        Series with index of rsc_ids and values containing the reason why
        a reaction species is a cofactor. Empty series if no filters apply.
    """

    filtered_rscs = []

    # Apply each filter rule to this reaction
    for filter_type, cofactor_filter in cofactor_schema.items():
        dropped_species = _filter_one_reactions_cofactors(
            one_rxns_species, filter_type, cofactor_filter
        )
        if dropped_species is not None:
            filtered_rscs.append(dropped_species)

    # Combine results for this reaction
    if filtered_rscs:
        return pd.concat(filtered_rscs)
    else:
        # Return empty Series with proper index/name if no filters applied
        return pd.Series([], dtype=str, name=COFACTOR_DEFS.FILTER_REASON)


def _filter_one_reactions_cofactors(
    one_rxns_species: pd.DataFrame, filter_type: str, cofactor_filter: dict
) -> pd.Series:
    """
    Filter One Reaction's Cofactors

    Apply a cofactor filter to one reaction's species

    Parameters:
    ----------
    one_rxns_species (pd.DataFrame):
        Rows of reactions species containing cofactors
    filter_type: str
        Reason to filter species with this filter
    cofactor_filter: dict
        Species included in filter

    Returns:
    ----------
    pd.Series with index of rsc_ids and values containing the reason why a
    reaction species is a cofactor, or None if filter was not triggered.

    """

    # see if all cofactor species are present
    rsc_labels_set = set(one_rxns_species[COFACTOR_DEFS.COFACTOR].tolist())
    missing_reqs = set(cofactor_filter[COFACTOR_DEFS.IF_ALL]).difference(rsc_labels_set)
    if len(missing_reqs) != 0:
        return None

    # ignore cases involving "except_any" species
    if COFACTOR_DEFS.EXCEPT_ANY in cofactor_filter.keys():
        detected_exceptions = set(
            cofactor_filter[COFACTOR_DEFS.EXCEPT_ANY]
        ).intersection(rsc_labels_set)
        if len(detected_exceptions) != 0:
            return None

    # consider a reaction only if "as_substrate" is a substrate
    if COFACTOR_DEFS.AS_SUBSTRATE in cofactor_filter.keys():
        substrates_set = set(
            one_rxns_species[one_rxns_species[SBML_DFS.STOICHIOMETRY] < 0][
                COFACTOR_DEFS.COFACTOR
            ].tolist()
        )
        substrates_detected = set(
            cofactor_filter[COFACTOR_DEFS.AS_SUBSTRATE]
        ).intersection(substrates_set)

        if len(substrates_detected) == 0:
            return None

    # save the dropped species and filter type (reason for filtering) to a dict
    dropped_species = one_rxns_species[
        one_rxns_species[COFACTOR_DEFS.COFACTOR].isin(
            cofactor_filter[COFACTOR_DEFS.IF_ALL]
        )
    ]

    return dropped_species.assign(filter_reason=filter_type)[
        COFACTOR_DEFS.FILTER_REASON
    ]


def _find_cofactor_species(
    sbml_dfs: SBML_dfs, cofactor_chebi_ids: dict = COFACTOR_CHEBI_IDS
) -> pd.DataFrame:
    """
    Find Cofactor Species

    Identify species in the SBML_dfs that match known cofactor ChEBI IDs.

    Parameters
    ----------
    sbml_dfs: SBML_dfs
        A pathway model
    cofactor_chebi_ids: dict, optional
        Dictionary mapping cofactor names to lists of ChEBI IDs.
        Defaults to COFACTOR_CHEBI_IDS.

    Returns
    -------
    pd.DataFrame
        DataFrame with species information and their corresponding cofactor names,
        with one row per species (deduplicated by s_id)
    """

    # Validate the cofactor mapping structure
    cofactor_validator = CofactorChebiIDs(cofactor_mapping=cofactor_chebi_ids)

    # Get validated mappings
    cofactor_ids_list = cofactor_validator.get_all_chebi_ids()
    chebi_to_cofactor = cofactor_validator.get_chebi_to_cofactor_map()

    if not isinstance(sbml_dfs, SBML_dfs):
        raise TypeError(f"sbml_dfs was type {type(sbml_dfs)} and must be an SBML_dfs")

    # find sbml_dfs species matching possible cofactors
    species_identifiers = sbml_dfs.get_identifiers(SBML_DFS.SPECIES)
    # filter to small molecules ignoring cases where a small molecule is just a part of the species
    species_identifiers = species_identifiers[
        [
            o == ONTOLOGIES.CHEBI and b == BQB.IS
            for o, b in zip(
                species_identifiers[IDENTIFIERS.ONTOLOGY],
                species_identifiers[IDENTIFIERS.BQB],
            )
        ]
    ]

    species_identifiers = species_identifiers.rename(
        columns={IDENTIFIERS.IDENTIFIER: ONTOLOGIES.CHEBI}
    )

    if species_identifiers.shape[0] == 0:
        raise ValueError("No species had ChEBI IDs, cofactors can not be filtered")

    species_identifiers[ONTOLOGIES.CHEBI] = species_identifiers[
        ONTOLOGIES.CHEBI
    ].astype(int)
    species_identifiers = species_identifiers[
        species_identifiers[ONTOLOGIES.CHEBI].isin(cofactor_ids_list)
    ]

    # Add cofactor name mapping
    species_identifiers[COFACTOR_DEFS.COFACTOR] = species_identifiers[
        ONTOLOGIES.CHEBI
    ].map(chebi_to_cofactor)

    # Get unique cofactor species (deduplicated by s_id)
    unique_cofactor_species = species_identifiers.drop_duplicates(
        subset=[SBML_DFS.S_ID]
    ).sort_values(SBML_DFS.S_NAME)

    unique_cofactor_names = sorted(
        unique_cofactor_species[COFACTOR_DEFS.COFACTOR].unique()
    )

    logger.info(
        f"There were {unique_cofactor_species.shape[0]} unique cofactor species found: "
        f"{', '.join(unique_cofactor_names)}"
    )

    # Report cofactors that were not found
    cofactors_found = set(species_identifiers[COFACTOR_DEFS.COFACTOR].unique())
    cofactors_expected = set(cofactor_chebi_ids.keys())
    cofactors_missed = sorted(cofactors_expected - cofactors_found)

    if len(cofactors_missed) != 0:
        logger.warning(
            f"{len(cofactors_missed)} of {len(cofactors_expected)} "
            "cofactors were not located in the pathway model: "
            f"{', '.join(cofactors_missed)}"
        )

    return species_identifiers


def _map_cofactors_to_reaction_species(
    sbml_dfs: SBML_dfs, cofactor_species: pd.DataFrame
) -> pd.DataFrame:
    """
    Map Cofactors to Reaction Species

    Take species with cofactor annotations and map them through compartmentalized
    species to reaction species, filtering to only those with non-zero stoichiometry.

    Parameters
    ----------
    sbml_dfs: SBML_dfs
        A pathway model
    cofactor_species: pd.DataFrame
        DataFrame with species information and cofactor annotations,
        must contain 'cofactor' column and be indexed by s_id

    Returns
    -------
    pd.DataFrame
        Reaction species DataFrame with cofactor annotations, filtered to
        non-zero stoichiometry entries
    """

    # Ensure cofactor_species has the required cofactor column
    if COFACTOR_DEFS.COFACTOR not in cofactor_species.columns:
        raise ValueError("cofactor_species must contain 'cofactor' column")

    # Create species -> cofactor mapping
    species_to_cofactor = cofactor_species.set_index(SBML_DFS.S_ID)[
        COFACTOR_DEFS.COFACTOR
    ]

    # Map species to compartmentalized species
    cofactor_cspecies = sbml_dfs.compartmentalized_species.merge(
        species_to_cofactor, left_on=SBML_DFS.S_ID, right_index=True, how="inner"
    )

    # Map compartmentalized species to reaction species
    cofactor_rscspecies = sbml_dfs.reaction_species.merge(
        cofactor_cspecies[COFACTOR_DEFS.COFACTOR],
        left_on=SBML_DFS.SC_ID,
        right_index=True,
        how="inner",
    )

    # Filter to entries that are actually produced or consumed (non-zero stoichiometry)
    cofactor_rscspecies = cofactor_rscspecies[
        cofactor_rscspecies[SBML_DFS.STOICHIOMETRY] != 0
    ]

    return cofactor_rscspecies


# validators


class CofactorChebiIDs(BaseModel):
    """Validator for COFACTOR_CHEBI_IDS dictionary structure."""

    cofactor_mapping: Dict[str, List[int]] = Field(
        description="Dictionary mapping cofactor names to lists of ChEBI IDs"
    )

    @field_validator("cofactor_mapping")
    @classmethod
    def validate_cofactor_chebi_structure(
        cls, v: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """
        Validate the cofactor ChEBI ID mapping structure.

        Checks:
        1. All keys are non-empty strings
        2. All values are non-empty lists of integers
        3. All ChEBI IDs are positive integers
        4. No ChEBI ID appears in multiple cofactor sets
        """
        if not isinstance(v, dict):
            raise ValueError("cofactor_mapping must be a dictionary")

        if not v:
            raise ValueError("cofactor_mapping cannot be empty")

        all_chebi_ids: Set[int] = set()
        duplicates: List[str] = []

        for cofactor_name, chebi_ids in v.items():
            # Check cofactor name
            if not isinstance(cofactor_name, str) or not cofactor_name.strip():
                raise ValueError(
                    f"Cofactor name must be a non-empty string, got: {cofactor_name}"
                )

            # Check ChEBI IDs list
            if not isinstance(chebi_ids, list):
                raise ValueError(
                    f"ChEBI IDs for '{cofactor_name}' must be a list, got: {type(chebi_ids)}"
                )

            if not chebi_ids:
                raise ValueError(
                    f"ChEBI IDs list for '{cofactor_name}' cannot be empty"
                )

            # Check individual ChEBI IDs
            for chebi_id in chebi_ids:
                if not isinstance(chebi_id, int):
                    raise ValueError(
                        f"ChEBI ID must be an integer, got: {chebi_id} for cofactor '{cofactor_name}'"
                    )

                if chebi_id <= 0:
                    raise ValueError(
                        f"ChEBI ID must be positive, got: {chebi_id} for cofactor '{cofactor_name}'"
                    )

                # Check for duplicates across cofactor sets
                if chebi_id in all_chebi_ids:
                    # Find which cofactor already has this ID
                    for existing_cofactor, existing_ids in v.items():
                        if (
                            existing_cofactor != cofactor_name
                            and chebi_id in existing_ids
                        ):
                            duplicates.append(
                                f"ChEBI ID {chebi_id} appears in both '{existing_cofactor}' and '{cofactor_name}'"
                            )
                            break
                else:
                    all_chebi_ids.add(chebi_id)

        if duplicates:
            raise ValueError(f"Duplicate ChEBI IDs found: {'; '.join(duplicates)}")

        return v

    def get_chebi_to_cofactor_map(self) -> Dict[int, str]:
        """
        Create a mapping from ChEBI IDs to cofactor names.

        Returns
        -------
        Dict[int, str]
            Dictionary mapping ChEBI ID to cofactor name
        """
        chebi_to_cofactor = {}
        for cofactor_name, chebi_ids in self.cofactor_mapping.items():
            for chebi_id in chebi_ids:
                chebi_to_cofactor[chebi_id] = cofactor_name
        return chebi_to_cofactor

    def get_all_chebi_ids(self) -> List[int]:
        """
        Get a flat list of all ChEBI IDs.

        Returns
        -------
        List[int]
            All ChEBI IDs across all cofactors
        """
        all_ids = []
        for chebi_ids in self.cofactor_mapping.values():
            all_ids.extend(chebi_ids)
        return all_ids


class CofactorFilterRule(BaseModel):
    """Validator for a single cofactor filter rule."""

    if_all: List[str] = Field(description="Cofactors that must all be present")
    except_any: List[str] = Field(
        default_factory=list, description="Cofactors that will override the filter"
    )
    as_substrate: List[str] = Field(
        default_factory=list, description="Cofactors that must be present as substrates"
    )

    @field_validator(COFACTOR_DEFS.IF_ALL)
    @classmethod
    def validate_if_all_not_empty(cls, v: List[str]) -> List[str]:
        """Validate that if_all is not empty."""
        if not v:
            raise ValueError("if_all cannot be empty")
        return v

    @field_validator(
        COFACTOR_DEFS.IF_ALL, COFACTOR_DEFS.EXCEPT_ANY, COFACTOR_DEFS.AS_SUBSTRATE
    )
    @classmethod
    def validate_cofactor_names(cls, v: List[str]) -> List[str]:
        """Validate that all cofactor names are non-empty strings."""
        for name in v:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Cofactor name must be a non-empty string, got: {name}"
                )
        return v


class CofactorSchema(BaseModel):
    """Validator for COFACTOR_SCHEMA dictionary structure."""

    schema_mapping: Dict[str, CofactorFilterRule] = Field(
        description="Dictionary mapping filter names to cofactor filter rules"
    )

    @field_validator("schema_mapping")
    @classmethod
    def validate_schema_not_empty(
        cls, v: Dict[str, CofactorFilterRule]
    ) -> Dict[str, CofactorFilterRule]:
        """Validate that schema is not empty."""
        if not v:
            raise ValueError("schema_mapping cannot be empty")
        return v
