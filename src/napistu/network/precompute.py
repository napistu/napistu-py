from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from napistu.network.ng_core import NapistuGraph
from napistu.network.ig_utils import validate_edge_attributes
from napistu.constants import NAPISTU_EDGELIST, SBML_DFS
from napistu.network.constants import (
    NAPISTU_GRAPH_EDGES,
)

logger = logging.getLogger(__name__)


def precompute_distances(
    napistu_graph: NapistuGraph,
    max_steps: int = -1,
    max_score_q: float = float(1),
    partition_size: int = int(5000),
    weights_vars: list[str] = [
        NAPISTU_GRAPH_EDGES.WEIGHTS,
        NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHTS,
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
    weights_vars: list
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
      *One variable will exist for each weight specified in 'weights_vars'

    """

    if max_steps == -1:
        max_steps = int(100000)

    # validate inputs
    if max_steps < 1:
        raise ValueError(f"max_steps must >= 1, but was {max_steps}")

    if (max_score_q < 0) or (max_score_q > 1):
        raise ValueError(f"max_score_q must be between 0 and 1 but was {max_score_q}")

    # make sure weight vars exist
    validate_edge_attributes(napistu_graph, weights_vars)

    # assign molecular species to partitions
    vs_to_partition = pd.DataFrame(
        {"sc_id": napistu_graph.vs["name"], "node_type": napistu_graph.vs["node_type"]}
    ).query("node_type == 'species'")

    n_paritions = math.ceil(vs_to_partition.shape[0] / partition_size)

    vs_to_partition["partition"] = vs_to_partition.index % n_paritions
    vs_to_partition = vs_to_partition.set_index("partition").sort_index()

    # interate through all partitions of "from" nodes and find their shortest and lowest weighted paths
    unique_partitions = vs_to_partition.index.unique().tolist()

    logger.info(f"Calculating distances for {len(unique_partitions)} partitions")
    precomputed_distances = pd.concat(
        [
            _calculate_distances_subset(
                napistu_graph,
                vs_to_partition,
                vs_to_partition.loc[uq_part],
                weights_vars=weights_vars,
            )
            for uq_part in unique_partitions
        ]
    ).query("sc_id_origin != sc_id_dest")

    # filter by path length and/or weight

    logger.info(
        f"Filtering distances by path length ({max_steps}) and score quantile ({max_score_q})"
    )
    filtered_precomputed_distances = _filter_precomputed_distances(
        precomputed_distances=precomputed_distances,
        max_steps=max_steps,
        max_score_q=max_score_q,
        path_weights_vars=["path_" + w for w in weights_vars],
    ).reset_index(drop=True)

    return filtered_precomputed_distances


def _calculate_distances_subset(
    napistu_graph: NapistuGraph,
    vs_to_partition: pd.DataFrame,
    one_partition: pd.DataFrame,
    weights_vars: list[str] = [
        NAPISTU_GRAPH_EDGES.WEIGHTS,
        NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHTS,
    ],
) -> pd.DataFrame:
    """Calculate distances from a subset of vertices to all vertices."""

    d_steps = (
        pd.DataFrame(
            np.array(
                napistu_graph.distances(
                    source=one_partition[SBML_DFS.SC_ID],
                    target=vs_to_partition[SBML_DFS.SC_ID],
                )
            ),
            index=one_partition[SBML_DFS.SC_ID].rename(NAPISTU_EDGELIST.SC_ID_ORIGIN),
            columns=vs_to_partition[SBML_DFS.SC_ID].rename(NAPISTU_EDGELIST.SC_ID_DEST),
        )
        .reset_index()
        .melt(NAPISTU_EDGELIST.SC_ID_ORIGIN, value_name="path_length")
        .replace([np.inf, -np.inf], np.nan, inplace=False)
        .dropna()
    )

    d_weights_list = list()
    for weight_type in weights_vars:
        d_weights_subset = (
            pd.DataFrame(
                np.array(
                    napistu_graph.distances(
                        source=one_partition[SBML_DFS.SC_ID],
                        target=vs_to_partition[SBML_DFS.SC_ID],
                        weights=weight_type,
                    )
                ),
                index=one_partition[SBML_DFS.SC_ID].rename(
                    NAPISTU_EDGELIST.SC_ID_ORIGIN
                ),
                columns=vs_to_partition[SBML_DFS.SC_ID].rename(
                    NAPISTU_EDGELIST.SC_ID_DEST
                ),
            )
            .reset_index()
            .melt(NAPISTU_EDGELIST.SC_ID_ORIGIN, value_name=f"path_{weight_type}")
            .replace([np.inf, -np.inf], np.nan, inplace=False)
            .dropna()
        )

        d_weights_list.append(d_weights_subset)

    d_weights = d_weights_list.pop()
    while len(d_weights_list) > 0:
        d_weights = d_weights.merge(d_weights_list.pop())

    # merge shortest path distances by length and by weight
    # note: these may be different paths! e.g., a longer path may have a lower weight than a short one
    path_summaries = d_steps.merge(
        d_weights,
        left_on=[NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST],
        right_on=[NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST],
    )

    # return connected species
    return path_summaries


def _filter_precomputed_distances(
    precomputed_distances: pd.DataFrame,
    max_steps: float | int = np.inf,
    max_score_q: float = 1,
    path_weights_vars: list[str] = ["path_weights", "path_upstream_weights"],
) -> pd.DataFrame:
    """Filter precomputed distances by maximum steps and/or to low scores by quantile."""

    # filter by path lengths
    short_precomputed_distances = precomputed_distances[
        precomputed_distances["path_length"] <= max_steps
    ]
    n_filtered_by_path_length = (
        precomputed_distances.shape[0] - short_precomputed_distances.shape[0]
    )
    if n_filtered_by_path_length > 0:
        logger.info(
            f"filtered {n_filtered_by_path_length} possible paths with length > {max_steps}"
        )

    # filter by path weights
    for wt_var in path_weights_vars:
        score_q_cutoff = np.quantile(short_precomputed_distances[wt_var], max_score_q)

        short_precomputed_distances.loc[
            short_precomputed_distances[wt_var] > score_q_cutoff, wt_var
        ] = np.nan

    valid_weights = short_precomputed_distances[path_weights_vars].dropna(how="all")

    low_weight_precomputed_distances = short_precomputed_distances[
        short_precomputed_distances.index.isin(valid_weights.index.tolist())
    ]

    n_filtered_by_low_weight = (
        short_precomputed_distances.shape[0] - low_weight_precomputed_distances.shape[0]
    )

    if n_filtered_by_low_weight > 0:
        logger.info(
            f"filtered {n_filtered_by_low_weight} possible paths with path weights greater"
        )
        logger.info(f"than the {max_score_q} quantile of the path weight distribution")

    weight_nan_summary = valid_weights.isnull().sum()
    if any(weight_nan_summary != 0):
        nan_summary = " and ".join(
            [
                f"{k} has {v} np.nan values"
                for k, v in weight_nan_summary.to_dict().items()
            ]
        )
        logger.info(nan_summary)

    return low_weight_precomputed_distances
