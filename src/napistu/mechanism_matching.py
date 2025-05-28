from __future__ import annotations

import logging
from typing import Optional, Union, Set, Dict, List

import igraph as ig
import numpy as np
import pandas as pd

from napistu import identifiers
from napistu import sbml_dfs_core
from napistu import utils
from napistu.constants import SBML_DFS
from napistu.constants import CPR_EDGELIST
from napistu.constants import CPR_EDGELIST_REQ_VARS
from napistu.constants import FEATURE_ID_VAR_DEFAULT
from napistu.constants import RESOLVE_MATCHES_AGGREGATORS
from napistu.constants import RESOLVE_MATCHES_TMP_WEIGHT_COL
from napistu.constants import IDENTIFIERS
from napistu.constants import IDENTIFIER_EDGELIST_REQ_VARS
from napistu.constants import ONTOLOGIES_LIST
from napistu.network.constants import CPR_GRAPH_EDGES
from napistu.network import paths

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
) -> sbml_dfs_core.SBML_dfs:
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

    Returns
    -------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The sbml_dfs object with the results bound.
    """

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

    return sbml_dfs


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
        generally using sbml_dfs_core.export_sbml_dfs()
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


def edgelist_to_pathway_species(
    formatted_edgelist: pd.DataFrame,
    species_identifiers: pd.DataFrame,
    ontologies: set,
    feature_id_var: str = FEATURE_ID_VAR_DEFAULT,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Edgelist to Pathway Species

    Match an edgelist of molecular species pairs to their corresponding species in a pathway representation.

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and "identifier_downstream" variables used to to match entries
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from sbml_dfs.get_identifiers("species") generally using
        sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species
    feature_id_var: str, default=FEATURE_ID_VAR_DEFAULT
        Variable in "formatted_edgelist" containing feature ids
    verbose: bool, default=False
        Whether to print verbose output

    Returns:
    edges_on_pathway: pd.DataFrame
        formatted_edgelist with upstream features mapped
        to "s_id_upstream" and downstream species mapped
        to "s_id_downstream"
    """

    required_vars_distinct_features = {
        CPR_EDGELIST.IDENTIFIER_UPSTREAM,
        CPR_EDGELIST.IDENTIFIER_DOWNSTREAM,
    }
    missing_required_vars_distinct_features = (
        required_vars_distinct_features.difference(
            set(formatted_edgelist.columns.tolist())
        )
    )

    if len(missing_required_vars_distinct_features) > 0:
        raise ValueError(
            f"{len(missing_required_vars_distinct_features)} required variables were "
            "missing from 'formatted_edgelist': "
            f"{', '.join(missing_required_vars_distinct_features)}"
        )

    # define all distinct identifiers in edgelist
    distinct_identifiers = (
        pd.concat(
            [
                formatted_edgelist[CPR_EDGELIST.IDENTIFIER_UPSTREAM],
                formatted_edgelist[CPR_EDGELIST.IDENTIFIER_DOWNSTREAM],
            ]
        )
        .drop_duplicates()
        .reset_index(drop=True)
        .to_frame()
        .rename({0: feature_id_var}, axis=1)
    )

    # merge edgelist identifiers with pathway identifiers to map s_ids to identifiers
    features_on_pathway = features_to_pathway_species(
        feature_identifiers=distinct_identifiers,
        species_identifiers=species_identifiers,
        ontologies=ontologies,
        feature_identifiers_var=feature_id_var,
        verbose=verbose,
    )

    # add s_ids of both upstream and downstream edges to pathway
    edges_on_pathway = formatted_edgelist.merge(
        features_on_pathway[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER]].rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_UPSTREAM,
                IDENTIFIERS.IDENTIFIER: CPR_EDGELIST.IDENTIFIER_UPSTREAM,
            },
            axis=1,
        )
    ).merge(
        features_on_pathway[[SBML_DFS.S_ID, IDENTIFIERS.IDENTIFIER]].rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_DOWNSTREAM,
                IDENTIFIERS.IDENTIFIER: CPR_EDGELIST.IDENTIFIER_DOWNSTREAM,
            },
            axis=1,
        )
    )

    return edges_on_pathway


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


def edgelist_to_scids(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
):
    """

    Edgelist to Compartmentalized Species IDds

    Map an edgelist of possible mechanistic interactions onto a
    pathadex pathway

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and
        "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers produced from
        sbml_dfs.get_identifiers("species") generally using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species

    Returns:
    edgelist_w_scids: pd.DataFrame
        formatted_edgelist with upstream features mapped to "sc_id_upstream" and
        downstream species mapped to "sc_id_downstream"
    """

    identifiers._check_species_identifiers_table(species_identifiers)

    # map edges onto pathway entities based on shared identifiers
    edges_on_pathway = edgelist_to_pathway_species(
        formatted_edgelist=formatted_edgelist,
        species_identifiers=species_identifiers,
        ontologies=ontologies,
    )

    # expand from s_ids to sc_ids
    s_id_pairs = edges_on_pathway[
        [CPR_EDGELIST.S_ID_UPSTREAM, CPR_EDGELIST.S_ID_DOWNSTREAM]
    ].drop_duplicates()
    sc_id_pairs = s_id_pairs.merge(
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
        .reset_index()
        .rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_UPSTREAM,
                SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM,
            },
            axis=1,
        )
    ).merge(
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID]]
        .reset_index()
        .rename(
            {
                SBML_DFS.S_ID: CPR_EDGELIST.S_ID_DOWNSTREAM,
                SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM,
            },
            axis=1,
        )
    )

    # map sc_ids back to edges_on_pathway
    # join lookup table of s_id_upstream, s_id_downstream -> sc_ids
    edgelist_w_scids = edges_on_pathway.merge(sc_id_pairs)

    logger_msg = (
        f"{edgelist_w_scids.shape[0]} interactions mapped "
        "onto pairs of compartmentalized species in the mechanistic model"
    )
    if edgelist_w_scids.shape[0] == 0:
        logger.warning(logger_msg)
    else:
        logger.info(logger_msg)

    return edgelist_w_scids


def filter_to_direct_mechanistic_interactions(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
) -> pd.DataFrame:
    """
    Filter to Direct Mechanistic Interactions

    Filter an edgelist to direct mechanistic interactions

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pd.DataFrame
        A table of molecular species identifiers
        produced from sbml_dfs.get_identifiers("species") generally
        using sbml_dfs_core.export_sbml_dfs()
    ontologies: set
        A set of ontologies used to match features to pathway species

    Returns:
    edgelist_w_direct_mechanistic_interactions: pd.DataFrame
        formatted_edgelist filtered to mechanistic reactions present in the pathway representation
    """

    edgelist_w_scids = _edgelist_to_scids_if_needed(
        formatted_edgelist, sbml_dfs, species_identifiers, ontologies
    )

    # reduce to distinct sc_id pairs
    sc_id_pairs = edgelist_w_scids[list(CPR_EDGELIST_REQ_VARS)].drop_duplicates()

    # define all existing direct regulatory interactions
    pathway_interactions = pd.concat(
        [
            # pair 0 -> <0 # modifiers affect substrates
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] == 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] < 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
            # pair <0 -> >0 # substrates affect products
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] < 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] > 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
            # pair 0 -> >0 # modifiers affect products
            sbml_dfs.reaction_species[
                sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] == 0
            ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]]
            .rename({SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_UPSTREAM}, axis=1)
            .merge(
                sbml_dfs.reaction_species[
                    sbml_dfs.reaction_species[SBML_DFS.STOICHIOMETRY] > 0
                ][[SBML_DFS.R_ID, SBML_DFS.SC_ID]].rename(
                    {SBML_DFS.SC_ID: CPR_EDGELIST.SC_ID_DOWNSTREAM}, axis=1
                )
            ),
        ]
    ).reset_index(drop=True)

    # filter pathway interactions based on matches to sc_id_pairs
    direct_edge_interactions = (
        sc_id_pairs.merge(pathway_interactions)
        .merge(
            sbml_dfs.species[SBML_DFS.S_NAME]
            .to_frame()
            .rename({SBML_DFS.S_NAME: CPR_EDGELIST.S_NAME_UPSTREAM}, axis=1),
            left_on=CPR_EDGELIST.S_ID_UPSTREAM,
            right_index=True,
            # add species metadata for matches
        )
        .merge(
            sbml_dfs.species[SBML_DFS.S_NAME]
            .to_frame()
            .rename({SBML_DFS.S_NAME: CPR_EDGELIST.S_NAME_DOWNSTREAM}, axis=1),
            left_on=CPR_EDGELIST.S_ID_DOWNSTREAM,
            right_index=True,
            # add metadata for reactions where interaction occurs
        )
        .merge(
            sbml_dfs.reactions[SBML_DFS.R_NAME].to_frame(),
            left_on=SBML_DFS.R_ID,
            right_index=True,
        )
    )

    edgelist_w_direct_mechanistic_interactions = edgelist_w_scids.merge(
        direct_edge_interactions[
            [
                CPR_EDGELIST.SC_ID_UPSTREAM,
                CPR_EDGELIST.SC_ID_DOWNSTREAM,
                SBML_DFS.R_ID,
                CPR_EDGELIST.S_NAME_UPSTREAM,
                CPR_EDGELIST.S_NAME_DOWNSTREAM,
                SBML_DFS.R_NAME,
            ]
        ]
    )

    return edgelist_w_direct_mechanistic_interactions


def filter_to_indirect_mechanistic_interactions(
    formatted_edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    cpr_graph: ig.Graph,
    ontologies: set,
    precomputed_distances=None,
    max_path_length=10,
):
    """
    Filter to Indirect Mechanistic Interactions

    Filter an edgelist to indirect mechanistic interactions.
    Indirect relationships are identified by searching a
    network for paths from an upstream species to a downstream species

    Parameters:
    formatted_edgelist: pd.DataFrame
        pd.Dataframe containing a "identifier_upstream" and
        "identifier_downstream" variables used to to match entries
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    species_identifiers: pandas.DataFrame
        A table of molecular species identifiers produced from
        sbml_dfs.get_identifiers("species") generally using sbml_dfs_core.export_sbml_dfs()
    cpr_graph: igraph.Graph
        A network representation of the sbml_dfs model
    ontologies: set
        A set of ontologies used to match features to pathway species
    precomputed_distances: None or a pd.DataFrame containing path lengths and weights
        between pairs of cspecies.
    max_path_length: int
        Maximum number of steps to consider.

    Returns:
    edgelist_w_indirect_mechanistic_interactions: pd.DataFrame
        formatted_edgelist filtered to mechanistic reactions which can be described
        by an indirect mechanism. The mechanism is described by a path weight, length,
        and a vpath and epath list of vertices and edges which were traversed to create the path.
    """

    edgelist_w_scids = _edgelist_to_scids_if_needed(
        formatted_edgelist, sbml_dfs, species_identifiers, ontologies
    )

    if precomputed_distances is not None:
        # rename to match conventions in precomputed_distances
        # filter by these precomputed distances and then restore naming
        edgelist_w_scids = paths._filter_paths_by_precomputed_distances(
            edgelist_w_scids.rename(
                {
                    CPR_EDGELIST.SC_ID_UPSTREAM: CPR_EDGELIST.SC_ID_ORIGIN,
                    CPR_EDGELIST.SC_ID_DOWNSTREAM: CPR_EDGELIST.SC_ID_DEST,
                },
                axis=1,
            ),
            precomputed_distances,
        ).rename(
            {
                CPR_EDGELIST.SC_ID_ORIGIN: CPR_EDGELIST.SC_ID_UPSTREAM,
                CPR_EDGELIST.SC_ID_DEST: CPR_EDGELIST.SC_ID_DOWNSTREAM,
            },
            axis=1,
        )

    # find paths from 1 upstream to all desired downstream sc_ids
    # (this is the convention with igraph)
    indexed_origin_vertices = edgelist_w_scids.set_index(CPR_EDGELIST.SC_ID_UPSTREAM)

    # loop through upstream cspecies and find paths to all downstream species
    global_dict = dict()
    for an_origin_index in indexed_origin_vertices.index.unique():  # type: ignore
        origin_targets = indexed_origin_vertices.loc[
            an_origin_index
        ]  # type: pd.DataFrame

        # if indexing only a single entry pd.DataFrame becomes a pd.Series
        # convert back to DataFrame for consistency
        origin_targets = utils.ensure_pd_df(origin_targets)

        # log entry for debugging
        logger.debug(
            f"finding paths from {an_origin_index} to "
            f"{origin_targets.shape[0]} target vertices"
        )

        # find all paths from indexed_origin to desired destination
        shortest_paths = paths.find_shortest_reaction_paths(
            cpr_graph,
            sbml_dfs,
            origin=an_origin_index,
            # find all unique destinations (as a list for compatibility with igraph dest)
            dest=origin_targets[CPR_EDGELIST.SC_ID_DOWNSTREAM].unique().tolist(),
            weight_var=CPR_GRAPH_EDGES.WEIGHTS,
        )

        if shortest_paths is None:
            continue

        vertices, edges = shortest_paths
        indexed_edges = edges.set_index("path")
        indexed_vertices = vertices.set_index("path")

        paths_list = list()
        for ind in indexed_edges.index.unique():
            one_path = indexed_edges.loc[ind]

            # make sure that we are working with a DF
            if type(one_path) is pd.Series:
                one_path = one_path.to_frame().T

            if one_path.shape[0] > max_path_length:
                continue

            # find the destination node
            # this is annoying because if the graph is undirected
            # its not clear if the from or to edge is the actual destination
            # when taking advantage of the fact that igraph lets you
            # look up multiple destinations at once this information is lost
            ancestor_species = {an_origin_index}
            if one_path.shape[0] > 1:
                penultimate_edge = one_path.iloc[one_path.shape[0] - 2]
                ancestor_species = ancestor_species.union(
                    {
                        penultimate_edge[CPR_GRAPH_EDGES.FROM],
                        penultimate_edge[CPR_GRAPH_EDGES.TO],
                    }
                )

            terminal_edge = one_path.iloc[one_path.shape[0] - 1]
            ending_cspecies = {terminal_edge[CPR_GRAPH_EDGES.FROM], terminal_edge[CPR_GRAPH_EDGES.TO]}.difference(ancestor_species)  # type: ignore

            if len(ending_cspecies) != 1:
                raise ValueError(
                    "The terminal edge could not be determined when summarizing paths"
                )
            ending_cspecies = ending_cspecies.pop()

            path_series = pd.Series(
                {
                    CPR_GRAPH_EDGES.FROM: an_origin_index,
                    CPR_GRAPH_EDGES.TO: ending_cspecies,
                    "weight": sum(one_path[CPR_GRAPH_EDGES.WEIGHTS]),
                    "path_length": one_path.shape[0],
                    "vpath": indexed_vertices.loc[ind],
                    "epath": one_path,
                }  # type: ignore
            )  # type: pd.Series

            paths_list.append(path_series)

        if len(paths_list) > 0:
            origin_paths = pd.DataFrame(paths_list)
            global_dict[an_origin_index] = origin_paths

    if len(global_dict.keys()) == 0:
        logger.warning(
            "None of the provide molecular pairs could be mechanistically linked with a network path"
        )
        return None

    all_shortest_paths = pd.concat(global_dict.values())

    indirect_shortest_paths = edgelist_w_scids.merge(
        all_shortest_paths,
        left_on=[CPR_EDGELIST.SC_ID_UPSTREAM, CPR_EDGELIST.SC_ID_DOWNSTREAM],
        right_on=[CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO],
    )

    return indirect_shortest_paths


def _edgelist_to_scids_if_needed(
    edgelist: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_identifiers: pd.DataFrame,
    ontologies: set,
) -> pd.DataFrame:
    """Map a set of edgelist species to cspecies or skip if cspecies were provided."""

    if utils.match_pd_vars(edgelist, CPR_EDGELIST_REQ_VARS).are_present:
        logger.info(
            f"An edgelist with {', '.join(CPR_EDGELIST_REQ_VARS)} was provided; identifier matching will be skipped"
        )
        return edgelist
    else:
        utils.match_pd_vars(edgelist, IDENTIFIER_EDGELIST_REQ_VARS).assert_present()

        identifiers._check_species_identifiers_table(species_identifiers)

        edgelist_w_scids = edgelist_to_scids(
            edgelist,
            sbml_dfs=sbml_dfs,
            species_identifiers=species_identifiers,
            ontologies=ontologies,
        )

        return edgelist_w_scids


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


def _log_feature_species_mapping_stats(
    pathway_species: pd.DataFrame, feature_id_var: str = FEATURE_ID_VAR_DEFAULT
):
    """
    Log statistics about the mapping between feature_id and s_id in the pathway_species DataFrame.
    """

    # Percent of feature_ids present one or more times in the output
    n_feature_ids = pathway_species[feature_id_var].nunique()
    n_input_feature_ids = (
        pathway_species[feature_id_var].max() + 1
        if feature_id_var in pathway_species.columns
        else 0
    )
    percent_present = (
        100 * n_feature_ids / n_input_feature_ids if n_input_feature_ids else 0
    )
    logger.info(
        f"{percent_present:.1f}% of feature_ids are present one or more times in the output ({n_feature_ids}/{n_input_feature_ids})"
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
