from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from napistu.constants import (
    NAPISTU_EDGELIST,
    SBML_DFS,
)
from napistu.network.constants import (
    DISTANCES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
)
from napistu.network.ig_utils import validate_edge_attributes
from napistu.network.ng_core import NapistuGraph

logger = logging.getLogger(__name__)


def precompute_distances(
    napistu_graph: NapistuGraph,
    max_steps: Optional[int] = None,
    max_score_q: float = float(1),
    partition_size: int = int(1000),
    weight_vars: list[str] = [
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    ],
) -> pd.DataFrame:
    """
    Precompute Distances between all pairs of species in a NapistuGraph network.

    Parameters
    ----------
    napistu_graph: NapistuGraph
        An NapistuGraph network model (subclass of igraph.Graph)
    max_steps: int
        The maximum number of steps between pairs of species to save a distance
    max_score_q: float
        Retain up to the "max_score_q" quantiles of all scores (small scores are better)
    partition_size: int
        The number of species to process together when computing distances. Decreasing this
        value will lower the overall memory footprint of distance calculation.
    weight_vars: list
        One or more variables defining edge weights to use when calculating weighted
        shortest paths. Shortest paths will be separately calculated with each type of
        weights and used to construct path weights named according to 'path_{weight_var}'

    Returns:
    ----------
    A pd.DataFrame containing:
    - sc_id_origin: origin node
    - sc_id_dest: destination node
    - path_length: minimum path length between from and to
    - path_weight*: minimum path weight between from and to (formed by summing the weights of individual edges).
      *One variable will exist for each weight specified in 'weight_vars'

    """

    # validate inputs
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must >= 1, but was {max_steps}")

    if (max_score_q <= 0) or (max_score_q > 1):
        raise ValueError(f"max_score_q must be between 0 and 1 but was {max_score_q}")

    # make sure weight vars exist
    validate_edge_attributes(napistu_graph, weight_vars)

    # assign molecular species to partitions
    vs_to_partition = pd.DataFrame(
        {
            SBML_DFS.SC_ID: napistu_graph.vs[NAPISTU_GRAPH_VERTICES.NAME],
            NAPISTU_GRAPH_VERTICES.NODE_TYPE: napistu_graph.vs[
                NAPISTU_GRAPH_VERTICES.NODE_TYPE
            ],
        }
    ).query(
        f"{NAPISTU_GRAPH_VERTICES.NODE_TYPE} == '{NAPISTU_GRAPH_NODE_TYPES.SPECIES}'"
    )

    n_paritions = math.ceil(vs_to_partition.shape[0] / partition_size)

    vs_to_partition["partition"] = vs_to_partition.index % n_paritions
    vs_to_partition = vs_to_partition.set_index("partition").sort_index()

    # iterate through all partitions of "from" nodes and find their shortest and lowest weighted paths
    unique_partitions = vs_to_partition.index.unique().tolist()

    logger.info(f"Calculating distances for {len(unique_partitions)} partitions")
    precomputed_distances = pd.concat(
        [
            _calculate_distances_subset(
                napistu_graph,
                vs_to_partition,
                vs_to_partition.loc[uq_part],
                weight_vars=weight_vars,
                max_steps=max_steps,
            )
            for uq_part in unique_partitions
        ]
    ).query(f"{DISTANCES.SC_ID_ORIGIN} != {DISTANCES.SC_ID_DEST}")

    # filter by path length and/or weight
    logger.info(
        f"Filtering distances by path length ({max_steps}) and score quantile ({max_score_q})"
    )
    filtered_precomputed_distances = _filter_precomputed_distances(
        precomputed_distances=precomputed_distances,
        max_score_q=max_score_q,
        path_weight_vars=["path_" + w for w in weight_vars],
    ).reset_index(drop=True)

    # validate the precomputed distances
    logger.info("Validating precomputed distances")
    _validate_precomputed_distances(filtered_precomputed_distances)

    return filtered_precomputed_distances


def filter_precomputed_distances_top_n(precomputed_distances, top_n=50):
    """
    Filter precomputed distances to only include the top-n pairs for each distance measure.

    Parameters
    ----------
    precomputed_distances : pd.DataFrame
        Precomputed distances.
    top_n : int, optional
        Top-n pairs to include for each distance measure.

    Returns
    -------
    pd.DataFrame
        Filtered precomputed distances.
    """

    # take the union of top-n for each distance measure; and from origin -> dest and dest -> origin
    distance_vars = set(precomputed_distances.columns) - {
        NAPISTU_EDGELIST.SC_ID_ORIGIN,
        NAPISTU_EDGELIST.SC_ID_DEST,
    }

    valid_pairs = list()
    for distance_var in distance_vars:
        top_n_pairs_by_origin = (
            precomputed_distances.sort_values(by=distance_var, ascending=False)
            .groupby(NAPISTU_EDGELIST.SC_ID_ORIGIN)
            .head(top_n)
        )
        top_n_pairs_by_dest = (
            precomputed_distances.sort_values(by=distance_var, ascending=False)
            .groupby(NAPISTU_EDGELIST.SC_ID_DEST)
            .head(top_n)
        )

        valid_pairs.append(
            top_n_pairs_by_origin[
                [NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST]
            ]
        )
        valid_pairs.append(
            top_n_pairs_by_dest[
                [NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST]
            ]
        )

    all_valid_pairs = pd.concat(valid_pairs).drop_duplicates()

    return precomputed_distances.merge(
        all_valid_pairs,
        on=[NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST],
        how="inner",
    )


def _calculate_distances_subset(
    napistu_graph: NapistuGraph,
    vs_to_partition: pd.DataFrame,
    one_partition: pd.DataFrame,
    weight_vars: list[str] = [
        NAPISTU_GRAPH_EDGES.WEIGHT,
        NAPISTU_GRAPH_EDGES.WEIGHT_UPSTREAM,
    ],
    max_steps: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate shortest path distances from a subset of vertices to all vertices.

    This function computes both unweighted (hop count) and weighted shortest path
    distances from a subset of source vertices to all target vertices in the graph.
    Memory optimization is achieved through early filtering of invalid paths and
    deduplication of identical weight variables.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The network graph containing vertices and weighted edges. Must be a
        subclass of igraph.Graph with edge attributes specified in `weight_vars`.
    vs_to_partition : pd.DataFrame
        DataFrame containing all target vertices in the graph. Must have columns
        matching `SBML_DFS.SC_ID` and `NAPISTU_GRAPH_VERTICES.NODE_TYPE`.
        Represents the full set of potential destination nodes.
    one_partition : pd.DataFrame
        DataFrame containing the subset of source vertices for this calculation.
        Must be a subset of `vs_to_partition` with the same column structure.
        Represents the source nodes for shortest path calculations.
    weight_vars : list of str, default ['weight', 'upstream_weight']
        List of edge attribute names to use for weighted shortest path calculations.
        Each variable will result in a corresponding 'path_{weight_var}' column
        in the output. Identical weight variables are automatically detected and
        deduplicated to avoid redundant calculations.
    max_steps : int, optional
        Maximum number of hops to consider in shortest paths. If specified,
        paths longer than `max_steps` are filtered out during calculation
        rather than after, reducing memory usage. If None, no early filtering
        is applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with shortest path information containing:

        - `sc_id_origin` : str
            Source vertex identifier from `one_partition`
        - `sc_id_dest` : str
            Destination vertex identifier from `vs_to_partition`
        - `path_length` : int
            Minimum number of hops in unweighted shortest path
        - `path_{weight_var}` : float
            Minimum weighted path cost for each weight variable specified.
            One column per entry in `weight_vars`. Values are np.nan for
            unreachable vertex pairs.

    Notes
    -----
    Implementation optimizations:

    1. **Early filtering**: If `max_steps` is provided, only paths ≤ max_steps
       and finite distances are retained, significantly reducing memory usage
       for sparse or filtered networks.

    2. **Weight deduplication**: Identical weight variables (checked via
       `np.array_equal`) are detected automatically. Only unique weight
       calculations are performed, with results copied to duplicate columns.

    3. **Memory efficiency**: Distance matrices are processed immediately
       after calculation and masked arrays are used to avoid storing
       full NxM matrices for large graphs.

    The function assumes that `napistu_graph.distances()` returns finite values
    for connected vertex pairs and `np.inf` for disconnected pairs. Self-loops
    (same origin and destination) are not filtered at this level.

    Examples
    --------
    >>> # Calculate distances from first 100 nodes to all nodes
    >>> partition_0 = vs_to_partition.iloc[:100]
    >>> distances = _calculate_distances_subset(
    ...     graph, vs_to_partition, partition_0,
    ...     weight_vars=['weight', 'upstream_weight'],
    ...     max_steps=5
    ... )
    >>> distances.head()
    """

    # Calculate unweighted distances
    distances_matrix = np.array(
        napistu_graph.distances(
            source=one_partition[SBML_DFS.SC_ID],
            target=vs_to_partition[SBML_DFS.SC_ID],
        )
    )

    if max_steps is not None:
        valid_mask = (distances_matrix <= max_steps) & np.isfinite(distances_matrix)
    else:
        valid_mask = np.isfinite(distances_matrix)

    # Get valid positions
    valid_rows, valid_cols = np.where(valid_mask)

    # Create the unweighted distances DataFrame - only for valid entries
    d_steps = pd.DataFrame(
        {
            NAPISTU_EDGELIST.SC_ID_ORIGIN: one_partition[SBML_DFS.SC_ID]
            .iloc[valid_rows]
            .values,
            NAPISTU_EDGELIST.SC_ID_DEST: vs_to_partition[SBML_DFS.SC_ID]
            .iloc[valid_cols]
            .values,
            DISTANCES.PATH_LENGTH: distances_matrix[valid_rows, valid_cols],
        }
    )

    # in some setups multiple weight variables are identical. Becasue of this only calculate distances
    # with unique variables; then assign the calculated values to all weight variables
    unique_vars_map, representatives = _find_unique_weight_vars(
        napistu_graph, weight_vars
    )

    # Calculate distances only for unique weight variables
    calculated_weights = {}  # Maps representative_var -> calculated values

    for rep_var in representatives.keys():
        weight_matrix = np.array(
            napistu_graph.distances(
                source=one_partition[SBML_DFS.SC_ID],
                target=vs_to_partition[SBML_DFS.SC_ID],
                weights=rep_var,
            )
        )

        # Extract values using the same mask
        weight_values = weight_matrix[valid_rows, valid_cols]
        calculated_weights[rep_var] = np.where(
            np.isfinite(weight_values), weight_values, np.nan
        )

    # Create weight DataFrames for all requested variables (including duplicates)
    d_weights_data = {
        NAPISTU_EDGELIST.SC_ID_ORIGIN: one_partition[SBML_DFS.SC_ID]
        .iloc[valid_rows]
        .values,
        NAPISTU_EDGELIST.SC_ID_DEST: vs_to_partition[SBML_DFS.SC_ID]
        .iloc[valid_cols]
        .values,
    }

    # Assign calculated values to all requested weight variables
    for var in weight_vars:
        rep_var = unique_vars_map[var]
        d_weights_data[f"path_{var}"] = calculated_weights[rep_var]

    d_weights = pd.DataFrame(d_weights_data)

    # Merge shortest path distances by length and by weight
    path_summaries = d_steps.merge(
        d_weights,
        left_on=[NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST],
        right_on=[NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST],
    )

    # return connected species
    return path_summaries


def _filter_precomputed_distances(
    precomputed_distances: pd.DataFrame,
    max_score_q: float = 1,
    path_weight_vars: list[str] = [
        DISTANCES.PATH_WEIGHT,
        DISTANCES.PATH_WEIGHT_UPSTREAM,
    ],
) -> pd.DataFrame:
    """Filter precomputed distances by maximum steps and/or to low scores by quantile."""

    # Create masks for each path_weight_var indicating whether the path is below the cutoff
    masks = []
    for wt_var in path_weight_vars:
        score_q_cutoff = np.quantile(precomputed_distances[wt_var], max_score_q)
        # Mask is True if weight is below or equal to cutoff (keep the row)
        mask = precomputed_distances[wt_var] <= score_q_cutoff
        masks.append(mask)

    # Combine masks: keep rows where at least one weight variable is below cutoff
    # (i.e., keep rows that are True in at least one mask)
    combined_mask = np.logical_or.reduce([mask.values for mask in masks])

    # Filter to rows which have at least one weight variable below the cutoff
    low_weight_precomputed_distances = precomputed_distances[combined_mask].copy()

    n_filtered_by_low_weight = (
        precomputed_distances.shape[0] - low_weight_precomputed_distances.shape[0]
    )

    if n_filtered_by_low_weight > 0:
        logger.info(
            f"filtered {n_filtered_by_low_weight} possible paths with path weights greater"
        )
        logger.info(f"than the {max_score_q} quantile of the path weight distribution")

    return low_weight_precomputed_distances


def _find_unique_weight_vars(
    napistu_graph: NapistuGraph, weight_vars: list[str]
) -> tuple[dict, dict]:
    """
    Find unique weight variables to avoid redundant distance calculations.

    Returns:
        tuple: (unique_vars_map, representatives)
            - unique_vars_map: Maps weight_var -> representative_var for calculation
            - representatives: Maps representative_var -> list of vars it represents
    """
    if len(weight_vars) == 0:
        raise ValueError("weight_vars cannot be empty")

    if len(weight_vars) == 1:
        single_var_map = {var: var for var in weight_vars}
        single_representatives = {var: [var] for var in weight_vars}
        return single_var_map, single_representatives

    # Get edge attributes for comparison
    weight_arrays = {}
    for var in weight_vars:
        weight_arrays[var] = np.array(napistu_graph.es[var])

    # Find which variables are identical
    unique_vars = {}  # Maps var -> representative var to calculate
    representatives = {}  # Maps representative -> list of vars it represents

    for var in weight_vars:
        # Check if this var is identical to any already processed var
        found_match = False
        for rep_var in representatives.keys():
            if np.array_equal(weight_arrays[var], weight_arrays[rep_var]):
                unique_vars[var] = rep_var
                representatives[rep_var].append(var)
                found_match = True
                break

        if not found_match:
            # This is a new unique variable
            unique_vars[var] = var
            representatives[var] = [var]

    # Log the deduplication results
    total_vars = len(weight_vars)
    unique_count = len(representatives)
    if unique_count < total_vars:
        logger.debug(
            f"Weight deduplication: {total_vars} vars -> {unique_count} unique calculations"
        )
        for rep_var, identical_vars in representatives.items():
            if len(identical_vars) > 1:
                logger.debug(f"  '{rep_var}' represents: {identical_vars}")

    return unique_vars, representatives


def _validate_precomputed_distances(precomputed_distances: pd.DataFrame) -> None:
    """
    Validate the precomputed distances DataFrame.

    This function checks the following:
    1. All required variables are present.
    2. All weight variables are numeric.
    3. No missing values are present.
    4. No negative weights are present.
    5. No infinite weights are present.
    """

    REQUIRED_VARS = {NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST}

    missing_vars = REQUIRED_VARS - set(precomputed_distances.columns)
    if missing_vars:
        raise ValueError(
            f"'precomputed_distances' is missing required variables: {missing_vars}"
        )

    # check for non-numeric weights
    weight_vars = list(set(precomputed_distances.columns) - REQUIRED_VARS)
    for var in weight_vars:
        if not pd.api.types.is_numeric_dtype(precomputed_distances[var]):
            raise ValueError(f"Variable '{var}' is not numeric")

    # check for missing values
    rows_with_nans = sum(precomputed_distances.isna().any(axis=1))
    if rows_with_nans > 0:
        raise ValueError(f"{rows_with_nans} rows have missing values")

    # check for negative weights
    negative_weights = precomputed_distances[weight_vars].lt(0).any().any()
    if negative_weights:
        raise ValueError("Negative weights detected")

    # check for infinite weights
    infinite_weights = (
        precomputed_distances[weight_vars].isin([np.inf, -np.inf]).any().any()
    )
    if infinite_weights:
        raise ValueError("Infinite weights detected")

    return None
