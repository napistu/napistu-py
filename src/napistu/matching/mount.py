import copy
import logging
from typing import Optional, Union, Set, Dict

import pandas as pd

from napistu.constants import SBML_DFS, ONTOLOGIES_LIST
from napistu.matching.constants import (
    FEATURE_ID_VAR_DEFAULT,
    RESOLVE_MATCHES_AGGREGATORS,
    RESOLVE_MATCHES_TMP_WEIGHT_COL,
    BIND_DICT_OF_WIDE_RESULTS_STRATEGIES,
    BIND_DICT_OF_WIDE_RESULTS_STRATEGIES_LIST,
)
from napistu import identifiers, utils
from napistu.matching.species import match_features_to_wide_pathway_species
from napistu import sbml_dfs_core

logger = logging.getLogger(__name__)


def bind_wide_results(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    results_df: pd.DataFrame,
    results_name: str,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    dogmatic: bool = False,
    species_identifiers: Optional[pd.DataFrame] = None,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    numeric_agg: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
    keep_id_col: bool = True,
    verbose: bool = False,
    inplace: bool = True,
) -> Optional[sbml_dfs_core.SBML_dfs]:
    """
    Binds wide results to a sbml_dfs object.

    Take a table with molecular species-level attributes tied to systematic identifiers and match them to an sbml_dfs_model transferring these attributes to species_data

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The sbml_dfs object to bind the results to.
    results_df : pd.DataFrame
        The table containing the results to bind.
    results_name : str
        The name of the results to bind.
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies (these should be entries in ONTOLOGIES_LIST )
        - Dict mapping wide column names to ontology names in the ONTOLOGIES_LIST controlled vocabulary
        - None to automatically detect valid ontology columns based on ONTOLOGIES_LIST
    dogmatic : bool
        Whether to respect differences between genes, transcripts, and proteins (True) or ignore them (False).
    species_identifiers : Optional[pd.DataFrame]
        Systematic identifiers for the molecular species "sbml_dfs". If None this will be generate on-the-fly.
    feature_id_var : str
        The name of the column in the results_df that contains the feature identifiers. If this does not exist it will be created.
    numeric_agg : str
        The aggregation method to use for resolving degeneracy.
    keep_id_col : bool
        Whether to keep the identifier column in the results_df.
    verbose : bool
        Whether to log cases of 1-to-many and many-to-one mapping and to indicate the behavior for resolving degeneracy
    inplace : bool, default=True
        Whether to modify the sbml_dfs object in place. If False, returns a copy.

    Returns
    -------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The sbml_dfs object with the results bound.
    """

    if not inplace:
        sbml_dfs = copy.deepcopy(sbml_dfs)

    species_identifiers = identifiers._prepare_species_identifiers(
        sbml_dfs, dogmatic=dogmatic, species_identifiers=species_identifiers
    )

    # match
    matched_s_ids_from_wide = match_features_to_wide_pathway_species(
        results_df,
        species_identifiers,
        ontologies=ontologies,
        feature_id_var=feature_id_var,
        verbose=verbose,
    )

    disambiguated_matches = resolve_matches(
        matched_data=matched_s_ids_from_wide,
        feature_id_var=feature_id_var,
        numeric_agg=numeric_agg,
        keep_id_col=keep_id_col,
    )

    clean_species_data = utils.drop_extra_cols(
        results_df, disambiguated_matches, always_include=[feature_id_var]
    )

    sbml_dfs.add_species_data(results_name, clean_species_data)

    return None if inplace else sbml_dfs


def bind_dict_of_wide_results(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    results_dict: dict,
    results_name: str,
    strategy: str = BIND_DICT_OF_WIDE_RESULTS_STRATEGIES.CONTATENATE,
    species_identifiers: pd.DataFrame = None,
    ontologies: Optional[Union[str, list]] = None,
    dogmatic: bool = False,
    inplace: bool = True,
    verbose=True,
):
    """
    Bind a dictionary of wide results to an SBML_dfs object.

    This function is used to bind a dictionary of wide results to 1 or more species_data attributes of an SBML_dfs object.
    The dictionary should have keys which are the modality names and values which are the results dataframes.
    The "strategy" argument controls how the results are added to the SBML_dfs object.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        The SBML_dfs object to bind the results to.
    results_dict : dict
        A dictionary of results dataframes with modality names as keys.
    results_name : str
        The name of the species_data attribute to bind the results to.
    strategy : str
        The strategy to use for binding the results.

        Options are:
        - "concatenate" : concatenate the results dataframes and add them as a single attribute.
        - "multiple_keys" : add each modality's results as a separate attribute. The attribute name will be f'{results_name}_{modality}'.
        - "stagger" : add each modality's results as a separate attribute. The attribute name will be f'{attr_name}_{modality}'.

    species_identifiers : pd.DataFrame
        A dataframe with species identifiers.
    ontologies : optional str, list
        The ontology to use for the species identifiers. If not provided, the column names of the results dataframes which match ONTOLOGIES_LIST will be used.
    dogmatic : bool
        Whether to use dogmatic mode. Ignored if species_identifiers is provided.
    verbose : bool
        Whether to print verbose output.
    inplace : bool, default=True
        Whether to modify the sbml_dfs object in place. If False, returns a copy.

    Returns
    -------
    Optional[SBML_dfs]
        If inplace=True, returns None. Otherwise returns the modified copy of sbml_dfs.
    """

    # validate strategy
    if strategy not in BIND_DICT_OF_WIDE_RESULTS_STRATEGIES_LIST:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be one of {BIND_DICT_OF_WIDE_RESULTS_STRATEGIES_LIST}"
        )

    species_identifiers = identifiers._prepare_species_identifiers(
        sbml_dfs, dogmatic=dogmatic, species_identifiers=species_identifiers
    )

    if not inplace:
        sbml_dfs = copy.deepcopy(sbml_dfs)

    if strategy == BIND_DICT_OF_WIDE_RESULTS_STRATEGIES.MULTIPLE_KEYS:
        for modality, results_df in results_dict.items():
            valid_ontologies = _get_wide_results_valid_ontologies(
                results_df, ontologies
            )

            modality_results_name = f"{results_name}_{modality}"

            bind_wide_results(
                sbml_dfs,
                results_df,
                modality_results_name,
                species_identifiers=species_identifiers,
                ontologies=valid_ontologies,
                inplace=True,  # Always use inplace=True here since we handle copying above
                verbose=verbose,
            )

        return None if inplace else sbml_dfs

    # create either a concatenated or staggered results table
    if strategy == BIND_DICT_OF_WIDE_RESULTS_STRATEGIES.CONTATENATE:
        results_df = pd.concat(results_dict.values(), axis=0)
    elif strategy == BIND_DICT_OF_WIDE_RESULTS_STRATEGIES.STAGGER:

        results_dict_copy = results_dict.copy()
        for k, v in results_dict_copy.items():
            valid_ontologies = _get_wide_results_valid_ontologies(v, ontologies)

            if verbose:
                logger.info(
                    f"Modality {k} has ontologies {valid_ontologies}. Other variables will be renamed to {k}_<variable>"
                )

            # rename all the columns besides ontologies names
            for var in v.columns:
                if var not in valid_ontologies:
                    results_dict_copy[k].rename(
                        columns={var: f"{var}_{k}"}, inplace=True
                    )

        results_df = pd.concat(results_dict_copy.values(), axis=1)

    valid_ontologies = _get_wide_results_valid_ontologies(results_df, ontologies)

    bind_wide_results(
        sbml_dfs,
        results_df,
        results_name,
        species_identifiers=species_identifiers,
        ontologies=valid_ontologies,
        inplace=True,  # Always use inplace=True here since we handle copying above
        verbose=verbose,
    )

    return None if inplace else sbml_dfs


def resolve_matches(
    matched_data: pd.DataFrame,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    index_col: str = SBML_DFS.S_ID,
    numeric_agg: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
    keep_id_col: bool = True,
) -> pd.DataFrame:
    """
    Resolve many-to-1 and 1-to-many matches in matched data.

    Parameters
    ----------
    matched_data : pd.DataFrame
        DataFrame containing matched data with columns:
        - feature_id_var: identifier column (e.g. feature_id)
        - index_col: index column (e.g. s_id)
        - other columns: data columns to be aggregated
    feature_id_var : str, default="feature_id"
        Name of the identifier column
    index_col : str, default="s_id"
        Name of the column to use as index
    numeric_agg : str, default="weighted_mean"
        Method to aggregate numeric columns:
        - "weighted_mean": weighted by inverse of feature_id frequency (default)
        - "mean": simple arithmetic mean
        - "first": first value after sorting by feature_id_var (requires feature_id_var)
        - "max": maximum value
    keep_id_col : bool, default=True
        Whether to keep and rollup the feature_id_var in the output.
        If False, feature_id_var will be dropped from the output.

    Returns
    -------
    pd.DataFrame
        DataFrame with resolved matches:
        - Many-to-1: numeric columns are aggregated using specified method
        - 1-to-many: adds a count column showing number of matches
        - Index is set to index_col and named accordingly

    Raises
    ------
    KeyError
        If feature_id_var is not present in the DataFrame
    TypeError
        If DataFrame contains unsupported data types (boolean or datetime)
    """
    # Make a copy to avoid modifying input
    df = matched_data.copy()

    # Check for unsupported data types
    unsupported_dtypes = df.select_dtypes(include=["bool", "datetime64"]).columns
    if not unsupported_dtypes.empty:
        raise TypeError(
            f"Unsupported data types found in columns: {list(unsupported_dtypes)}. "
            "Boolean and datetime columns are not supported."
        )

    # Always require feature_id_var
    if feature_id_var not in df.columns:
        raise KeyError(feature_id_var)

    # Deduplicate by feature_id within each s_id using groupby and first BEFORE any further processing
    df = df.groupby([index_col, feature_id_var], sort=False).first().reset_index()

    # Use a unique temporary column name for weights
    if RESOLVE_MATCHES_TMP_WEIGHT_COL in df.columns:
        raise ValueError(
            f"Temporary weight column name '{RESOLVE_MATCHES_TMP_WEIGHT_COL}' already exists in the input data. Please rename or remove this column and try again."
        )

    # Calculate weights if needed (after deduplication!)
    if numeric_agg == RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN:
        feature_counts = df[feature_id_var].value_counts()
        df[RESOLVE_MATCHES_TMP_WEIGHT_COL] = (
            1 / feature_counts[df[feature_id_var]].values
        )

    # Set index for grouping
    df = df.set_index(index_col)

    # Use utility to split columns
    always_non_numeric = [feature_id_var] if keep_id_col else []
    numeric_cols, non_numeric_cols = _split_numeric_non_numeric_columns(
        df, always_non_numeric=always_non_numeric
    )

    # Get aggregator function
    numeric_aggregator = _get_numeric_aggregator(
        method=numeric_agg, feature_id_var=feature_id_var
    )
    resolved = _aggregate_grouped_columns(
        df,
        numeric_cols,
        non_numeric_cols,
        numeric_aggregator,
        feature_id_var=feature_id_var,
        numeric_agg=numeric_agg,
    )
    # Add count of matches per feature_id
    match_counts = matched_data.groupby(index_col)[feature_id_var].nunique()
    resolved[f"{feature_id_var}_match_count"] = match_counts

    # Drop feature_id_var if not keeping it
    if not keep_id_col and feature_id_var in resolved.columns:
        resolved = resolved.drop(columns=[feature_id_var])

    # Ensure index is named consistently
    resolved.index.name = index_col

    return resolved


def _get_wide_results_valid_ontologies(
    results_df: pd.DataFrame, ontologies: Optional[Union[str, list]] = None
) -> list:
    """
    Get the valid ontologies for a wide results dataframe.

    If ontologies is a string, it will be converted to a list.
    If ontologies is None, the column names of the results dataframe which match ONTOLOGIES_LIST will be used.

    Parameters
    ----------
    results_df : pd.DataFrame
        The results dataframe to get the valid ontologies for.
    ontologies : optional str, list
        The ontology to use for the species identifiers. If not provided, the column names of the results dataframes which match ONTOLOGIES_LIST will be used.

    Returns
    -------
    list
        The valid ontologies for the results dataframe.
    """

    if isinstance(ontologies, str):
        ontologies = [ontologies]  # now, it will be None or list

    if ontologies is None:
        ontologies = [col for col in results_df.columns if col in ONTOLOGIES_LIST]
        if len(ontologies) == 0:
            raise ValueError(
                "No valid ontologies found in results dataframe. Columns are: "
                + str(results_df.columns)
            )

    if isinstance(ontologies, list):
        invalid_ontologies = set(ontologies) - set(ONTOLOGIES_LIST)
        if len(invalid_ontologies) > 0:
            raise ValueError(
                "Invalid ontologies found in ontologies list: "
                + str(invalid_ontologies)
            )

    return ontologies


def _get_numeric_aggregator(
    method: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
) -> callable:
    """
    Get aggregation function for numeric columns with various methods.

    Parameters
    ----------
    method : str, default="weighted_mean"
        Aggregation method to use:
        - "weighted_mean": weighted by inverse of feature_id frequency (default)
        - "mean": simple arithmetic mean
        - "first": first value after sorting by feature_id_var (requires feature_id_var)
        - "max": maximum value
    feature_id_var : str, default="feature_id"
        Name of the column specifying a measured feature - used for sorting and weighting

    Returns
    -------
    callable
        Aggregation function to use with groupby

    Raises
    ------
    ValueError
        If method is not recognized
    """

    def weighted_mean(df: pd.DataFrame) -> float:
        # Get values and weights for this group
        values = df["value"]
        weights = df["weight"]
        # Weights are already normalized globally, just use them directly
        return (values * weights).sum() / weights.sum()

    def first_by_id(df: pd.DataFrame) -> float:
        # Sort by feature_id and take first value
        return df.sort_values(feature_id_var).iloc[0]["value"]

    def simple_mean(series: pd.Series) -> float:
        return series.mean()

    def simple_max(series: pd.Series) -> float:
        return series.max()

    aggregators = {
        RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN: weighted_mean,
        RESOLVE_MATCHES_AGGREGATORS.MEAN: simple_mean,
        RESOLVE_MATCHES_AGGREGATORS.FIRST: first_by_id,
        RESOLVE_MATCHES_AGGREGATORS.MAX: simple_max,
    }

    if method not in aggregators:
        raise ValueError(
            f"Unknown aggregation method: {method}. Must be one of {list(aggregators.keys())}"
        )

    return aggregators[method]


def _split_numeric_non_numeric_columns(df: pd.DataFrame, always_non_numeric=None):
    """
    Utility to split DataFrame columns into numeric and non-numeric, always treating specified columns as non-numeric.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    always_non_numeric : list or set, optional
        Columns to always treat as non-numeric (e.g., ['feature_id']).

    Returns
    -------
    numeric_cols : pd.Index
        Columns considered numeric (int64, float64, and not in always_non_numeric).
    non_numeric_cols : pd.Index
        Columns considered non-numeric (object, string, etc., plus always_non_numeric).
    """
    if always_non_numeric is None:
        always_non_numeric = []
    always_non_numeric = set(always_non_numeric)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(
        always_non_numeric
    )
    non_numeric_cols = df.columns.difference(numeric_cols)
    return numeric_cols, non_numeric_cols


def _aggregate_grouped_columns(
    df: pd.DataFrame,
    numeric_cols,
    non_numeric_cols,
    numeric_aggregator,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    numeric_agg: str = RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
) -> pd.DataFrame:
    """
    Aggregate numeric and non-numeric columns for grouped DataFrame.
    Assumes deduplication by feature_id within each s_id has already been performed.
    Returns the combined DataFrame.
    """
    results = []

    # Handle non-numeric columns
    if len(non_numeric_cols) > 0:
        non_numeric_agg = (
            df[non_numeric_cols]
            .groupby(level=0)
            .agg(lambda x: ",".join(sorted(set(x.astype(str)))))
        )
        results.append(non_numeric_agg)
    # Handle numeric columns
    if len(numeric_cols) > 0:
        numeric_results = {}
        for col in numeric_cols:
            if numeric_agg in [
                RESOLVE_MATCHES_AGGREGATORS.FIRST,
                RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN,
            ]:
                agg_df = pd.DataFrame(
                    {"value": df[col], feature_id_var: df[feature_id_var]}
                )
                if numeric_agg == RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN:
                    agg_df[RESOLVE_MATCHES_TMP_WEIGHT_COL] = df[
                        RESOLVE_MATCHES_TMP_WEIGHT_COL
                    ]
                numeric_results[col] = agg_df.groupby(level=0).apply(
                    lambda x: (
                        numeric_aggregator(x)
                        if numeric_agg != RESOLVE_MATCHES_AGGREGATORS.WEIGHTED_MEAN
                        else numeric_aggregator(
                            x.rename(columns={RESOLVE_MATCHES_TMP_WEIGHT_COL: "weight"})
                        )
                    )
                )
            else:
                numeric_results[col] = df[col].groupby(level=0).agg(numeric_aggregator)
        numeric_agg_df = pd.DataFrame(numeric_results)
        results.append(numeric_agg_df)
    # Combine results
    if results:
        resolved = pd.concat(results, axis=1)
    else:
        resolved = pd.DataFrame(index=df.index)
    return resolved
