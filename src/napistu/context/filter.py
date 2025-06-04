import copy
import logging
from typing import Union, List, Optional
import pandas as pd
from napistu import sbml_dfs_core

logger = logging.getLogger(__name__)


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
    # Check if species_data_table exists in sbml_dfs.species_data
    if species_data_table not in sbml_dfs.species_data:
        raise ValueError(
            f"species_data_table {species_data_table} not found in sbml_dfs.species_data. "
            f"Available tables: {sbml_dfs.species_data.keys()}"
        )

    # If not inplace, make a copy
    if not inplace:
        sbml_dfs = copy.deepcopy(sbml_dfs)

    # Get the species data
    species_data = sbml_dfs.species_data[species_data_table]

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
