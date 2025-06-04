import copy
import logging
from typing import Union, List, Optional, Iterable

import pandas as pd

from napistu import sbml_dfs_core
from napistu.constants import SBML_DFS, MINI_SBO_TO_NAME, SBO_NAME_TO_ROLE

logger = logging.getLogger(__name__)


def find_underspecified_reactions(
    sbml_dfs: sbml_dfs_core.SBML_dfs, sc_ids: Iterable[str]
) -> set[str]:
    """
    Find Underspecified reactions

    Identity reactions which should be removed if a set of molecular species are removed
    from the system.

    Params:
    sbml_dfs (SBML_dfs):
        A pathway representation
    sc_ids (list[str])
        A list of compartmentalized species ids (sc_ids) which will be removed.

    Returns:
    underspecified_reactions (set[str]):
        A list of reactions which should be removed because they will not occur once
        \"sc_ids\" are removed.

    """

    updated_reaction_species = sbml_dfs.reaction_species.copy()
    updated_reaction_species["new"] = ~updated_reaction_species[SBML_DFS.SC_ID].isin(
        sc_ids
    )

    updated_reaction_species = (
        updated_reaction_species.assign(
            sbo_role=updated_reaction_species[SBML_DFS.SBO_TERM]
        )
        .replace({"sbo_role": MINI_SBO_TO_NAME})
        .replace({"sbo_role": SBO_NAME_TO_ROLE})
    )

    reactions_with_lost_defining_members = set(
        updated_reaction_species.query("~new")
        .query("sbo_role == 'DEFINING'")[SBML_DFS.R_ID]
        .tolist()
    )

    N_reactions_with_lost_defining_members = len(reactions_with_lost_defining_members)
    if N_reactions_with_lost_defining_members > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_defining_members} reactions which have lost at least one defining species"
        )

    # for each reaction what are the required sbo_terms?
    reactions_with_requirements = (
        updated_reaction_species.query("sbo_role == 'REQUIRED'")[
            ["r_id", "sbo_term", "new"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # which required members are still present after removing some entries
    reactions_with_lost_requirements = set(
        reactions_with_requirements.query("~new")
        .merge(
            reactions_with_requirements.query("new").rename(
                {"new": "still_present"}, axis=1
            ),
            how="left",
        )
        .fillna(False)[SBML_DFS.R_ID]  # Fill boolean column with False
        .tolist()
    )

    N_reactions_with_lost_requirements = len(reactions_with_lost_requirements)
    if N_reactions_with_lost_requirements > 0:
        logger.info(
            f"Removing {N_reactions_with_lost_requirements} reactions which have lost all required members"
        )

    underspecified_reactions = reactions_with_lost_defining_members.union(
        reactions_with_lost_requirements
    )

    return underspecified_reactions


def filter_species_by_attribute(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_data_table: str,
    attribute_name: str,
    attribute_value: Union[int, bool, str, List[str]],
    negate: bool = False,
    inplace: bool = True,
) -> Optional[sbml_dfs_core.SBML_dfs]:
    """
    Filter species in the SBML_dfs based on an attribute value.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object to filter.
    species_data_table : str
        The name of the species data table to filter.
    attribute_name : str
        The name of the attribute to filter on.
    attribute_value : Union[int, bool, str, List[str]]
        The value of the attribute to filter on. Can be a single value or a list of values.
    negate : bool, optional
        Whether to negate the filter, by default False.
        If True, keeps species with the attribute defined that do NOT match the attribute value.
    inplace : bool, optional
        Whether to filter the SBML_dfs in place, by default True.
        If False, returns a new SBML_dfs object with the filtered species.

    Returns
    -------
    Optional[sbml_dfs_core.SBML_dfs]
        If inplace=True, returns None.
        If inplace=False, returns a new SBML_dfs object with the filtered species.

    Raises
    ------
    ValueError
        If species_data_table is not found in sbml_dfs.species_data
        If attribute_name is not found in the species data table columns
    """

    # If not inplace, make a copy
    if not inplace:
        sbml_dfs = copy.deepcopy(sbml_dfs)

    # Get the species data
    species_data = sbml_dfs.select_species_data(species_data_table)

    # Find species that match the filter criteria (including negation)
    species_to_remove = find_species_with_attribute(
        species_data, attribute_name, attribute_value, negate=negate
    )

    if isinstance(attribute_value, list):
        filter_str = (
            f"{attribute_name} in {attribute_value}"
            if not negate
            else f"{attribute_name} not in {attribute_value}"
        )
    else:
        filter_str = (
            f"{attribute_name}={attribute_value}"
            if not negate
            else f"{attribute_name}!={attribute_value}"
        )
    logger.info(
        f"Removing {len(species_to_remove)} species from {species_data_table} table with filter {filter_str}"
    )

    sbml_dfs._remove_species(species_to_remove)

    return None if inplace else sbml_dfs


def find_species_with_attribute(
    species_data: pd.DataFrame,
    attribute_name: str,
    attribute_value: Union[int, bool, str, List[str]],
    negate: bool = False,
) -> List[str]:
    """
    Find species that match the given attribute filter criteria.

    Parameters
    ----------
    species_data : pd.DataFrame
        The species data table to filter.
    attribute_name : str
        The name of the attribute to filter on.
    attribute_value : Union[int, bool, str, List[str]]
        The value of the attribute to filter on. Can be a single value or a list of values.
    negate : bool, optional
        Whether to negate the filter, by default False.
        If True, returns species that do NOT match the attribute value.

    Returns
    -------
    List[str]
        List of species IDs that match the filter criteria.

    Raises
    ------
    ValueError
        If attribute_name is not found in the species data table columns
    """
    # Check if attribute_name exists in species_data columns
    if attribute_name not in species_data.columns:
        raise ValueError(
            f"attribute_name {attribute_name} not found in species_data.columns. "
            f"Available attributes: {species_data.columns}"
        )

    # First, get the mask for defined values (not NA)
    defined_mask = species_data[attribute_name].notna()

    # Then, get the mask for matching values
    if isinstance(attribute_value, list):
        match_mask = species_data[attribute_name].isin(attribute_value)
    else:
        match_mask = species_data[attribute_name] == attribute_value

    # Apply negation if requested and combine with defined mask
    if negate:
        # When negating, we only want to consider rows where the attribute is defined
        final_mask = defined_mask & ~match_mask
    else:
        final_mask = defined_mask & match_mask

    # Return species that match our criteria
    return species_data[final_mask].index.tolist()


def _binarize_species_data(species_data: pd.DataFrame) -> pd.DataFrame:

    binary_series = []
    for c in species_data.columns:
        if species_data[c].dtype == "bool":
            binary_series.append(species_data[c].astype(int))
        elif species_data[c].dtype == "int64":
            if species_data[c].isin([0, 1]).all():
                binary_series.append(species_data[c])
            else:
                continue
        else:
            continue

    if len(binary_series) == 0:
        raise ValueError("No binary or boolean columns found")

    binary_df = pd.concat(binary_series, axis=1)

    if len(binary_df.columns) != len(species_data.columns):
        left_out = set(species_data.columns) - set(binary_df.columns)
        logger.warning(f"Some columns were not binarized: {', '.join(left_out)}")

    return binary_df
