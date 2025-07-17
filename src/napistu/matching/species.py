from __future__ import annotations

import logging
from typing import Optional, Union, Set, Dict, List

import numpy as np
import pandas as pd

from napistu import identifiers
from napistu.constants import ONTOLOGIES_LIST, SBML_DFS, IDENTIFIERS
from napistu.matching.constants import FEATURE_ID_VAR_DEFAULT

logger = logging.getLogger(__name__)


def features_to_pathway_species(
    feature_identifiers: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: set,
    feature_identifiers_var: str = IDENTIFIERS.IDENTIFIER,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    expand_identifiers: bool = False,
    identifier_delimiter: str = "/",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Features to Pathway Species

    Match a table of molecular species to their corresponding species in a pathway representation.

    Parameters:
    feature_identifiers: pd.DataFrame
        pd.Dataframe containing a "feature_identifiers_var" variable used to match entries
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from sbml_dfs.get_identifiers("species")
        generally using sbml_dfs.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species
    feature_identifiers_var: str
        Variable in "feature_identifiers" containing identifiers
    expand_identifiers: bool, default=False
        If True, split identifiers in feature_identifiers_var by identifier_delimiter and explode into multiple rows
    identifier_delimiter: str, default="/"
        Delimiter to use for splitting identifiers if expand_identifiers is True
    verbose: bool, default=False
        If True, log mapping statistics at the end of the function

    Returns:
    pathway_species: pd.DataFrame
        species_identifiers joined to feature_identifiers based on shared identifiers
    """

    # Check for identifier column
    if feature_identifiers_var not in feature_identifiers.columns.to_list():
        raise ValueError(
            f"{feature_identifiers_var} must be a variable in 'feature_identifiers', "
            f"possible variables are {', '.join(feature_identifiers.columns.tolist())}"
        )

    # Respect or create feature_id column
    feature_identifiers = _ensure_feature_id_var(feature_identifiers, feature_id_var)

    # Optionally expand identifiers into multiple rows
    if expand_identifiers:
        # Count the number of expansions by counting delimiters
        n_expansions = (
            feature_identifiers[feature_identifiers_var]
            .astype(str)
            .str.count(identifier_delimiter)
            .sum()
        )
        if n_expansions > 0:
            logger.info(
                f"Expanding identifiers: {n_expansions} delimiters found in '{feature_identifiers_var}', will expand to more rows."
            )

        # Split, strip whitespace, and explode
        feature_identifiers = feature_identifiers.copy()
        feature_identifiers[feature_identifiers_var] = (
            feature_identifiers[feature_identifiers_var]
            .astype(str)
            .str.split(identifier_delimiter)
            .apply(lambda lst: [x.strip() for x in lst])
        )
        feature_identifiers = feature_identifiers.explode(
            feature_identifiers_var, ignore_index=True
        )

    # check identifiers table
    identifiers._check_species_identifiers_table(species_identifiers)

    available_ontologies = set(species_identifiers[IDENTIFIERS.ONTOLOGY].tolist())
    unavailable_ontologies = ontologies.difference(available_ontologies)

    # no ontologies present
    if len(unavailable_ontologies) == len(ontologies):
        raise ValueError(
            f"None of the requested ontologies ({', '.join(ontologies)}) "
            "were used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    # 1+ desired ontologies are not present
    if len(unavailable_ontologies) > 0:
        raise ValueError(
            f"Some of the requested ontologies ({', '.join(unavailable_ontologies)}) "
            "were NOT used to annotate pathway species. Available ontologies are: "
            f"{', '.join(available_ontologies)}"
        )

    relevant_identifiers = species_identifiers[
        species_identifiers[IDENTIFIERS.ONTOLOGY].isin(ontologies)
    ]

    # map features to pathway species
    pathway_species = feature_identifiers.merge(
        relevant_identifiers,
        left_on=feature_identifiers_var,
        right_on=IDENTIFIERS.IDENTIFIER,
    )

    if pathway_species.shape[0] == 0:
        logger.warning(
            "None of the provided species identifiers matched entries of the pathway; returning None"
        )
        None

    # report the fraction of unmapped species
    if verbose:
        _log_feature_species_mapping_stats(pathway_species, feature_id_var)

    return pathway_species


def match_features_to_wide_pathway_species(
    wide_df: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: Optional[Union[Set[str], Dict[str, str]]] = None,
    feature_identifiers_var: str = IDENTIFIERS.IDENTIFIER,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert a wide-format DataFrame with multiple ontology columns to long format,
    and match features to pathway species by ontology and identifier.

    Parameters
    ----------
    wide_df : pd.DataFrame
        DataFrame with ontology identifier columns and any number of results columns.
        All non-ontology columns are treated as results.
    species_identifiers : pd.DataFrame
        DataFrame as required by features_to_pathway_species
    ontologies : Optional[Union[Set[str], Dict[str, str]]], default=None
        Either:
        - Set of columns to treat as ontologies (these should be entries in ONTOLOGIES_LIST )
        - Dict mapping wide column names to ontology names in the ONTOLOGIES_LIST controlled vocabulary
        - None to automatically detect valid ontology columns based on ONTOLOGIES_LIST
    feature_identifiers_var : str, default="identifier"
        Name for the identifier column in the long format
    feature_id_var: str, default=FEATURE_ID_VAR_DEFAULT
        Name for the feature id column in the long format
    verbose : bool, default=False
        Whether to print verbose output

    Returns
    -------
    pd.DataFrame
        Output of match_by_ontology_and_identifier

    Examples
    --------
    >>> # Example with auto-detected ontology columns and multiple results
    >>> wide_df = pd.DataFrame({
    ...     'uniprot': ['P12345', 'Q67890'],
    ...     'chebi': ['15377', '16810'],
    ...     'log2fc': [1.0, 2.0],
    ...     'pvalue': [0.01, 0.05]
    ... })
    >>> result = match_features_to_wide_pathway_species(
    ...     wide_df=wide_df,
    ...     species_identifiers=species_identifiers
    ... )

    >>> # Example with custom ontology mapping
    >>> wide_df = pd.DataFrame({
    ...     'protein_id': ['P12345', 'Q67890'],
    ...     'compound_id': ['15377', '16810'],
    ...     'expression': [1.0, 2.0],
    ...     'confidence': [0.8, 0.9]
    ... })
    >>> result = match_features_to_wide_pathway_species(
    ...     wide_df=wide_df,
    ...     species_identifiers=species_identifiers,
    ...     ontologies={'protein_id': 'uniprot', 'compound_id': 'chebi'}
    ... )
    """
    # Make a copy to avoid modifying the input
    wide_df = wide_df.copy()

    # Validate ontologies and get the set of ontology columns
    ontology_cols = _validate_wide_ontologies(wide_df, ontologies)
    melt_cols = list(ontology_cols)

    # Apply renaming if a mapping is provided
    if isinstance(ontologies, dict):
        wide_df = wide_df.rename(columns=ontologies)

    # Ensure feature_id column exists
    wide_df = _ensure_feature_id_var(wide_df, feature_id_var)

    # All non-ontology columns are treated as results
    results_cols = list(set(wide_df.columns) - set(melt_cols))
    if not results_cols:
        raise ValueError("No results columns found in DataFrame")

    logger.info(f"Using columns as results: {results_cols}")

    # Melt ontology columns to long format, keeping all results columns
    long_df = wide_df.melt(
        id_vars=results_cols,
        value_vars=melt_cols,
        var_name=IDENTIFIERS.ONTOLOGY,
        value_name=feature_identifiers_var,
    ).dropna(subset=[feature_identifiers_var])

    logger.debug(f"Final long format shape: {long_df.shape}")

    # Call the matching function with the validated ontologies
    out = match_by_ontology_and_identifier(
        feature_identifiers=long_df,
        species_identifiers=species_identifiers,
        ontologies=ontology_cols,
        feature_identifiers_var=feature_identifiers_var,
    )

    if verbose:
        _log_feature_species_mapping_stats(out, feature_id_var)

    return out


def match_by_ontology_and_identifier(
    feature_identifiers: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: Union[str, Set[str], List[str]],
    feature_identifiers_var: str = IDENTIFIERS.IDENTIFIER,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Match features to pathway species based on both ontology and identifier matches.
    Performs separate matching for each ontology and concatenates the results.

    Parameters
    ----------
    feature_identifiers : pd.DataFrame
        DataFrame containing feature identifiers and results.
        Must have columns [ontology, feature_identifiers_var, results]
    species_identifiers : pd.DataFrame
        DataFrame containing species identifiers from pathway.
        Must have columns [ontology, identifier]
    ontologies : Union[str, Set[str], List[str]]
        Ontologies to match on. Can be:
        - A single ontology string
        - A set of ontology strings
        - A list of ontology strings
    feature_identifiers_var : str, default="identifier"
        Name of the identifier column in feature_identifiers
    verbose : bool, default=False
        Whether to print verbose output

    Returns
    -------
    pd.DataFrame
        Concatenated results of matching for each ontology.
        Contains all columns from features_to_pathway_species()

    Examples
    --------
    >>> # Match using a single ontology
    >>> result = match_by_ontology_and_identifier(
    ...     feature_identifiers=features_df,
    ...     species_identifiers=species_df,
    ...     ontologies="uniprot"
    ... )

    >>> # Match using multiple ontologies
    >>> result = match_by_ontology_and_identifier(
    ...     feature_identifiers=features_df,
    ...     species_identifiers=species_df,
    ...     ontologies={"uniprot", "chebi"}
    ... )
    """
    # Convert string to set for consistent handling
    if isinstance(ontologies, str):
        ontologies = {ontologies}
    elif isinstance(ontologies, list):
        ontologies = set(ontologies)

    # Validate ontologies
    invalid_onts = ontologies - set(ONTOLOGIES_LIST)
    if invalid_onts:
        raise ValueError(
            f"Invalid ontologies specified: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
        )

    # Initialize list to store results
    matched_dfs = []

    # Process each ontology separately
    for ont in ontologies:
        # Filter feature identifiers to current ontology and drop ontology column
        ont_features = (
            feature_identifiers[feature_identifiers[IDENTIFIERS.ONTOLOGY] == ont]
            .drop(columns=[IDENTIFIERS.ONTOLOGY])
            .copy()
        )

        if ont_features.empty:
            logger.warning(f"No features found for ontology: {ont}")
            continue

        # Filter species identifiers to current ontology
        ont_species = species_identifiers[
            species_identifiers[IDENTIFIERS.ONTOLOGY] == ont
        ].copy()

        if ont_species.empty:
            logger.warning(f"No species found for ontology: {ont}")
            continue

        logger.debug(
            f"Matching {len(ont_features)} features to {len(ont_species)} species for ontology {ont}"
        )

        # Match features to species for this ontology
        matched = features_to_pathway_species(
            feature_identifiers=ont_features,
            species_identifiers=ont_species,
            ontologies={ont},
            feature_identifiers_var=feature_identifiers_var,
            verbose=verbose,
        )

        if matched.empty:
            logger.warning(f"No matches found for ontology: {ont}")
            continue

        matched_dfs.append(matched)

    if not matched_dfs:
        logger.warning("No matches found for any ontology")
        return pd.DataFrame()  # Return empty DataFrame with correct columns

    # Combine results from all ontologies
    result = pd.concat(matched_dfs, axis=0, ignore_index=True)

    logger.info(
        f"Found {len(result)} total matches across {len(matched_dfs)} ontologies"
    )

    return result


def _validate_wide_ontologies(
    wide_df: pd.DataFrame,
    ontologies: Optional[Union[str, Set[str], Dict[str, str]]] = None,
) -> Set[str]:
    """
    Validate ontology specifications against the wide DataFrame and ONTOLOGIES_LIST.

    Parameters
    ----------
    wide_df : pd.DataFrame
        DataFrame with one column per ontology and a results column
    ontologies : Optional[Union[str, Set[str], Dict[str, str]]]
        Either:
        - String specifying a single ontology column
        - Set of columns to treat as ontologies
        - Dict mapping wide column names to ontology names
        - None to automatically detect ontology columns based on ONTOLOGIES_LIST

    Returns
    -------
    Set[str]
        Set of validated ontology names. For dictionary mappings, returns the target ontology names.

    Raises
    ------
    ValueError
        If validation fails for any ontology specification or no valid ontologies are found
    """
    # Convert string input to set
    if isinstance(ontologies, str):
        ontologies = {ontologies}

    # Get the set of ontology columns
    if isinstance(ontologies, dict):
        # Check source columns exist in DataFrame
        missing_cols = set(ontologies.keys()) - set(wide_df.columns)
        if missing_cols:
            raise ValueError(f"Source columns not found in DataFrame: {missing_cols}")
        # Validate target ontologies against ONTOLOGIES_LIST
        invalid_onts = set(ontologies.values()) - set(ONTOLOGIES_LIST)
        if invalid_onts:
            raise ValueError(
                f"Invalid ontologies in mapping: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
            )
        # Return target ontology names instead of source column names
        ontology_cols = set(ontologies.values())

    elif isinstance(ontologies, set):
        # Check specified columns exist in DataFrame
        missing_cols = ontologies - set(wide_df.columns)
        if missing_cols:
            raise ValueError(
                f"Specified ontology columns not found in DataFrame: {missing_cols}"
            )
        # Validate specified ontologies against ONTOLOGIES_LIST
        invalid_onts = ontologies - set(ONTOLOGIES_LIST)
        if invalid_onts:
            raise ValueError(
                f"Invalid ontologies in set: {invalid_onts}. Must be one of: {ONTOLOGIES_LIST}"
            )
        ontology_cols = ontologies

    else:
        # Auto-detect ontology columns by matching against ONTOLOGIES_LIST
        ontology_cols = set(wide_df.columns) & set(ONTOLOGIES_LIST)
        if not ontology_cols:
            raise ValueError(
                f"No valid ontology columns found in DataFrame. Column names must match one of: {ONTOLOGIES_LIST}"
            )
        logger.info(f"Auto-detected ontology columns: {ontology_cols}")

    logger.debug(f"Validated ontology columns: {ontology_cols}")
    return ontology_cols


def _log_feature_species_mapping_stats(
    pathway_species: pd.DataFrame, feature_id_var: str = FEATURE_ID_VAR_DEFAULT
):
    """
    Log statistics about the mapping between feature_id and s_id in the pathway_species DataFrame.
    """

    # Percent change in feature_ids
    n_feature_ids = pathway_species[feature_id_var].nunique()
    n_input_feature_ids = (
        pathway_species[feature_id_var].max() + 1
        if feature_id_var in pathway_species.columns
        else 0
    )
    percent_change = (
        100 * (n_feature_ids - n_input_feature_ids) / n_input_feature_ids
        if n_input_feature_ids
        else 0
    )
    logger.info(
        f"{percent_change:+.1f}% change in feature_ids ({n_feature_ids} vs {n_input_feature_ids})"
    )

    # Number of times an s_id maps to 1+ feature_ids (with s_name)
    s_id_counts = pathway_species.groupby(SBML_DFS.S_ID)[feature_id_var].nunique()
    s_id_multi = s_id_counts[s_id_counts > 1]
    logger.info(f"{len(s_id_multi)} s_id(s) map to more than one feature_id.")
    if not s_id_multi.empty:
        examples = pathway_species[
            pathway_species[SBML_DFS.S_ID].isin(s_id_multi.index)
        ][[SBML_DFS.S_ID, SBML_DFS.S_NAME, feature_id_var]]
        logger.info(
            f"Examples of s_id mapping to multiple feature_ids (showing up to 3):\n{examples.groupby([SBML_DFS.S_ID, SBML_DFS.S_NAME])[feature_id_var].apply(list).head(3)}"
        )

    # Number of times a feature_id maps to 1+ s_ids (with s_name)
    feature_id_counts = pathway_species.groupby(feature_id_var)[SBML_DFS.S_ID].nunique()
    feature_id_multi = feature_id_counts[feature_id_counts > 1]
    logger.info(f"{len(feature_id_multi)} feature_id(s) map to more than one s_id.")
    if not feature_id_multi.empty:
        examples = pathway_species[
            pathway_species[feature_id_var].isin(feature_id_multi.index)
        ][[feature_id_var, SBML_DFS.S_ID, SBML_DFS.S_NAME]]
        logger.info(
            f"Examples of feature_id mapping to multiple s_ids (showing up to 3):\n{examples.groupby([feature_id_var])[[SBML_DFS.S_ID, SBML_DFS.S_NAME]].apply(lambda df: list(df.itertuples(index=False, name=None))).head(3)}"
        )


def _ensure_feature_id_var(
    df: pd.DataFrame, feature_id_var: str = FEATURE_ID_VAR_DEFAULT
) -> pd.DataFrame:
    """
    Ensure the DataFrame has a feature_id column, creating one if it doesn't exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check/modify
    feature_id_var : str, default=FEATURE_ID_VAR_DEFAULT
        Name of the feature ID column

    Returns
    -------
    pd.DataFrame
        DataFrame with guaranteed feature_id column
    """
    if feature_id_var not in df.columns:
        logger.warning(f"No {feature_id_var} column found in DataFrame, creating one")
        df = df.copy()
        df[feature_id_var] = np.arange(len(df))
    return df
