import copy
import logging
from typing import Union, List, Optional

import pandas as pd

from napistu import sbml_dfs_core
from napistu import sbml_dfs_utils
from napistu import utils
from napistu.constants import SBML_DFS

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


def filter_reactions_with_disconnected_cspecies(
    sbml_dfs: sbml_dfs_core.SBML_dfs, species_data_table: str, inplace: bool = False
) -> Optional[sbml_dfs_core.SBML_dfs]:
    """
    Remove reactions from the SBML_dfs object whose defining compartmentalized species (cspecies) are disconnected
    according to a co-occurrence matrix derived from a species data table.

    This function identifies reactions where any pair of defining cspecies do not co-occur (i.e., are disconnected)
    in the provided species data table, and removes those reactions from the model. The operation can be performed
    in-place or on a copy of the SBML_dfs object.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object to filter reactions from.
    species_data_table : str
        The name of the species data table to use for co-occurrence calculation.
    inplace : bool, optional
        If True, modifies the input SBML_dfs object in-place and returns None. If False (default),
        returns a new SBML_dfs object with the filtered reactions.

    Returns
    -------
    Optional[sbml_dfs_core.SBML_dfs]
        If inplace=True, returns None. If inplace=False, returns a new SBML_dfs object with filtered reactions.

    Warns
    -----
    UserWarning
        If no reactions are pruned based on non-cooccurrence.

    Examples
    --------
    >>> filtered_sbml_dfs = filter_reactions_with_disconnected_cspecies(sbml_dfs, "test_data", inplace=False)
    >>> # To modify in-place:
    >>> filter_reactions_with_disconnected_cspecies(sbml_dfs, "test_data", inplace=True)
    """

    if inplace:
        sbml_dfs = copy.deepcopy(sbml_dfs)

    # find how many conditions a pair of species cooccur in
    cooccurence_edgelist = _create_cooccurence_edgelist(sbml_dfs, species_data_table)

    reactions_to_remove = _find_reactions_with_disconnected_cspecies(
        cooccurence_edgelist, sbml_dfs
    )

    if len(reactions_to_remove) == 0:
        logger.warning("No reactions will be pruned based on non-cooccurrence.")
    else:
        logger.info(
            f"Pruning {len(reactions_to_remove)} reactions based on non-cooccurrence."
        )
        sbml_dfs.remove_reactions(reactions_to_remove)

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


def _find_reactions_with_disconnected_cspecies(
    coccurrence_edgelist: pd.DataFrame,
    sbml_dfs: Optional[sbml_dfs_core.SBML_dfs],
    cooccurence_threshold: int = 0,  #  noqa
) -> set:
    """
    Find reactions with disconnected cspecies.

    This function finds reactions with disconnected cspecies based on the cooccurrence matrix.
    Only cspecies which are DEFINING are considered because these are AND rules for reaction operability.
    It returns the set of reaction ids with disconnected cspecies.

    Parameters
    ----------
    coccurrence_edgelist : pd.DataFrame
        The cooccurrence edgelist.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object.
    cooccurence_threshold : int
        The threshold for cooccurrence. Values equal to or below this threshold are considered disconnected.

    Returns
    -------
    set
        The set of reaction ids with disconnected cspecies.

    """

    utils.match_pd_vars(
        coccurrence_edgelist, {"s_id_1", "s_id_2", "cooccurence"}
    ).assert_present()
    sbml_dfs._validate_table(SBML_DFS.REACTION_SPECIES)
    sbml_dfs._validate_table(SBML_DFS.COMPARTMENTALIZED_SPECIES)

    reaction_species = sbml_dfs_utils.add_sbo_role(sbml_dfs.reaction_species)

    logger.info(
        "Finding disconnected pairs of cspecies based on the zero values in the coccurrence_edgelist"
    )

    # map to cspcies
    disconnected_cspecies = (
        coccurrence_edgelist.query("cooccurence <= @cooccurence_threshold")
        .merge(
            sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
            .reset_index(drop=False)
            .rename(columns={SBML_DFS.S_ID: "s_id_1", SBML_DFS.SC_ID: "sc_id_1"}),
            how="left",
        )
        .merge(
            sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
            .reset_index(drop=False)
            .rename(columns={SBML_DFS.S_ID: "s_id_2", SBML_DFS.SC_ID: "sc_id_2"}),
            how="left",
        )
    )

    # remove defining attributes which don't occur since these are AND rules
    # ignore required attributes since these are OR rules and do not require cooccurrence

    defining_reaction_species = reaction_species.query("sbo_role == 'DEFINING'")[
        [SBML_DFS.R_ID, SBML_DFS.SC_ID]
    ].drop_duplicates()

    logger.info(
        "Finding reactions with disconnected cspecies based on the cooccurrence matrix"
    )
    # since any 2 pairs of cspecies being missing together would stop a reaction from operating, we can convert reaction_species to an edgelist by self-joining on reaction id
    invalid_defining_non_cooccurring = (
        (
            defining_reaction_species.rename(columns={SBML_DFS.SC_ID: "sc_id_1"}).merge(
                defining_reaction_species.rename(columns={SBML_DFS.SC_ID: "sc_id_2"}),
                on=SBML_DFS.R_ID,
                how="left",
            )
        )
        .query("sc_id_1 != sc_id_2")
        .merge(disconnected_cspecies, on=["sc_id_1", "sc_id_2"], how="inner")
    )

    invalid_defining_non_cooccurring_reactions = set(
        invalid_defining_non_cooccurring[SBML_DFS.R_ID].unique()
    )

    return invalid_defining_non_cooccurring_reactions


def _create_cooccurence_edgelist(
    sbml_dfs: sbml_dfs_core.SBML_dfs, species_data_table: str
):
    """
    Create a co-occurrence edgelist for species based on a binary species data table.

    This function computes a co-occurrence matrix for all pairs of species in the given data table,
    where each entry represents the number of conditions in which both species are present (i.e., have value 1).
    The result is returned as an edgelist DataFrame with columns 's_id_1', 's_id_2', and 'cooccurence'.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing the species data table.
    species_data_table : str
        The name of the species data table to use for co-occurrence calculation. The table must contain only binary or boolean columns.

    Returns
    -------
    pd.DataFrame
        Edgelist DataFrame with columns ['s_id_1', 's_id_2', 'cooccurence'], where each row gives the number of conditions in which the two species co-occur.

    Raises
    ------
    ValueError
        If no binary or boolean columns are found in the species data table.
    """
    species_data = sbml_dfs.select_species_data(species_data_table)

    # select all binary columns (results in {0, 1})
    # convert to numpy ndarray
    binary_matrix = _binarize_species_data(species_data).to_numpy()

    # x * t(x)
    cooccurrence_matrix = binary_matrix @ binary_matrix.T
    # convert to a binary matrix

    cooccurence_edgelist = utils.matrix_to_edgelist(
        cooccurrence_matrix,
        row_labels=species_data.index.tolist(),
        col_labels=species_data.index.tolist(),
    ).rename(columns={"row": "s_id_1", "column": "s_id_2", "value": "cooccurence"})

    # calculate the cooccurrence matrix
    return cooccurence_edgelist


def _binarize_species_data(species_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all boolean or binary columns in a species data table to a DataFrame of binary (0/1) values.

    This function selects columns of dtype 'bool' or integer columns containing only 0 and 1, and converts them to a DataFrame of binary values (0/1).
    Columns that are not boolean or binary are ignored. If no such columns are found, a ValueError is raised.

    Parameters
    ----------
    species_data : pd.DataFrame
        The species data table to binarize.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the binarized columns (0/1 values) from the input.

    Raises
    ------
    ValueError
        If no binary or boolean columns are found in the input DataFrame.

    Warns
    -----
    UserWarning
        If some columns in the input were not binarized and left out of the output.
    """
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
