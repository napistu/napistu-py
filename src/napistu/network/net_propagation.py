"""
Network propagation with null distribution testing.

This module provides functions for propagating vertex attributes over a network
(e.g. personalized PageRank) and testing observed scores against null distributions.
Supported null strategies include vertex permutation, edge permutation, parametric,
and uniform, enabling network-based significance testing for gene prioritization
and related analyses.

Public Functions
----------------
melt_propagation_results(propagation_results, index_name, attribute_name)
    Melt results from network_propagation_with_null into a tall format.
network_propagation_with_null(graph, attributes, null_strategy, ...)
    Apply network propagation and compare observed scores to a null distribution.
net_propagate_attributes(graph, attributes, propagation_method, ...)
    Propagate multiple attributes over a network using a propagation method.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import igraph as ig
import numpy as np
import pandas as pd
import scipy.stats as stats

from napistu.network.constants import (
    MASK_KEYWORDS,
    NAPISTU_GRAPH_VERTICES,
    NET_PROPAGATION_DEFS,
    NET_PROPAGATION_METRICS,
    NULL_STRATEGIES,
    PARAMETRIC_NULL_DEFAULT_DISTRIBUTION,
    VALID_NULL_STRATEGIES,
)
from napistu.network.ig_utils import (
    _ensure_valid_attribute,
    _get_attribute_masks,
    _parse_mask_input,
)
from napistu.statistics.quantiles import calculate_quantiles
from napistu.utils.pd_utils import downcast_float_dataframe

logger = logging.getLogger(__name__)


@dataclass
class PropagationMethod:
    method: callable
    non_negative: bool


def melt_propagation_results(
    propagation_results: pd.DataFrame,
    index_name: Optional[str] = None,
    attribute_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Melt results from network_propagation_with_null into a tall format.

    Parameters
    ----------
    propagation_results : pd.DataFrame
        DataFrame with a 2-level MultiIndex on columns (metric, attribute) as
        returned by network_propagation_with_null.
    index_name : str, optional
        Name of the index to use for the feature_id column. If None, the existing index name
        will be used and an error will be raised if propagation_results does not have an index name.
    attribute_name : str, optional
        Name of the attribute column. If None, the existing attribute column will be used.

    Returns
    -------
    pd.DataFrame
        Tall DataFrame with columns:
        - feature_id: node identifier (from index)
        - attribute: attribute name
        - observed: raw propagated score
        - log2_enrichment: log2(observed / mean_null)
        - quantile: proportion of null values <= observed (only present if in input)
    """

    if not isinstance(propagation_results.columns, pd.MultiIndex):
        raise ValueError(
            "Input must have a MultiIndex column structure as returned by network_propagation_with_null"
        )
    if index_name is None:
        if propagation_results.index.name is None:
            raise ValueError(
                "Input must have an index name as returned by network_propagation_with_null"
            )
        index_name = propagation_results.index.name
    if attribute_name is None:
        if propagation_results.columns.names[1] is None:
            raise ValueError(
                "Input must have an attribute name as returned by network_propagation_with_null"
            )
        attribute_name = propagation_results.columns.names[1]

    metrics = propagation_results.columns.get_level_values(0).unique().tolist()
    expected_metrics = {
        NET_PROPAGATION_METRICS.OBSERVED,
        NET_PROPAGATION_METRICS.QUANTILE,
        NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
    }
    unexpected = set(metrics) - expected_metrics
    if unexpected:
        raise ValueError(f"Unexpected metrics in columns: {unexpected}")

    # Stack attribute level into rows, reset index to get feature_id
    tall = (
        propagation_results.stack(level=1, future_stack=True)
        .rename_axis(index=[index_name, attribute_name])
        .reset_index()
    )

    # Enforce column order with quantile conditional on presence
    col_order = [
        index_name,
        attribute_name,
        NET_PROPAGATION_METRICS.OBSERVED,
        NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
    ]
    if NET_PROPAGATION_METRICS.QUANTILE in metrics:
        col_order.append(NET_PROPAGATION_METRICS.QUANTILE)

    return tall[col_order]


def network_propagation_with_null(
    graph: ig.Graph,
    attributes: List[str],
    null_strategy: str = NULL_STRATEGIES.VERTEX_PERMUTATION,
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    n_samples: int = 100,
    verbose: bool = False,
    **null_kwargs,
) -> pd.DataFrame:
    """
    Apply network propagation to attributes and compare against null distributions.

    This is the main orchestrator function that:
    1. Calculates observed propagated scores
    2. Generates null distribution using specified strategy
    3. Returns a MultiIndex DataFrame with observed scores, quantiles, and log2 enrichment

    Null Strategy Selection
    ----------------------
    Two main approaches are used in network biology:

    **Vertex permutation** ('vertex_permutation'): Permutes node labels/attributes while
    preserving network topology. This tests whether individual nodes are significant
    given the network structure. Standard approach for gene prioritization and
    network-based gene set enrichment analysis.
    Reference: Schulte-Sasse et al. (2019) BMC Bioinformatics 20:587

    **Edge permutation** ('edge_permutation'): Rewires network edges while preserving
    degree distribution. This tests whether network topology itself is significant.
    Used when testing subnetwork patterns or connectivity significance.
    Reference: Leiserson et al. (2015) Nature Genetics (HotNet2 methodology)

    For vertex-level significance testing (gene prioritization), node permutation
    is the appropriate null model as it preserves network structure while
    randomizing signal assignment.

    Other supported null strategies:

    **Uniform ('uniform'):** A quick, qualitative readout. Generates a uniform null
    distribution over masked nodes and takes the ratio of observed network propagation
    score.

    **Parametric ('parametric'):** Similar to vertex permutation but rather than sampling
    observed values, samples are drawn from a distribution fit to the observed values.
    First fits a parametric distribution to the observed scores and then samples
    `n_samples` null samples for each vertex to compare observed to null quantiles.

    **Pooled vertex permutation** ('pooled_vertex_permutation'): Constructs a single
    empirical universe by merging all masked values across attributes (including zeros)
    and draws null reset vectors from it. The propagated null is broadcast across all
    attribute columns, so every attribute is compared against the same shared null.
    This reduces the propagation cost from `n_samples * n_attributes` to `n_samples`,
    but assumes attributes are exchangeable in both magnitude and sparsity. Only
    supported when all attributes share an identical mask. When attributes differ
    meaningfully in sparsity, this strategy can introduce systematic shifts in log2
    enrichment because propagation methods like personalized PageRank respond
    nonlinearly to reset concentration; in that case, prefer
    'attr_pooled_vertex_permutation'.

    **Attribute-pooled vertex permutation** ('attr_pooled_vertex_permutation'): Each
    null sample is generated by selecting one attribute, permuting its masked values
    without replacement, propagating, and broadcasting the result across all attribute
    columns. Total samples are distributed across attributes as evenly as possible
    (with the first `n_samples % n_attributes` attributes receiving one extra sample).
    This preserves each attribute's own sparsity and magnitude profile in the null
    while still pooling propagated outputs across attributes for variance reduction.
    Total propagation cost is `n_samples`, matching pooled vertex permutation. Only
    supported when all attributes share an identical mask. Prefer this over
    'pooled_vertex_permutation' when attributes share a measurable subgraph but
    differ in sparsity or magnitude profile.

    Creating Masks
    --------------
    Most null strategies benefit from including a mask which indicates which nodes are being tested.
    For vertex permutation the parametric null only masked nodes will be considered for sampling.
    Using a mask with uniform null strategy means that numeric reset probabilities will be compared to
    constant ones by default. Masking is an important consideration for mitigating ascertainment bias.
    If we are only sampling a subset of vertices like metabolites, we'll only consider those as sources
    of signals in the null.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to propagate and test.
    null_strategy : str
        Null distribution strategy. One of: 'uniform', 'parametric',
        'vertex_permutation', 'pooled_vertex_permutation',
        'attr_pooled_vertex_permutation', 'edge_permutation'.
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    n_samples : int
        Number of null samples to generate (ignored for uniform null).
    verbose : bool, optional
        Extra reporting. Default is False.
    **null_kwargs
        Additional arguments to pass to the null generator (e.g., mask, burn_in_ratio, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with a 2-level MultiIndex on columns (metric, attribute) where
        metric is one of ['observed', 'quantile', 'log2_enrichment'].

        - 'observed': raw propagated scores for each attribute
        - 'quantile': proportion of null values <= observed values (NaN for uniform null)
        - 'log2_enrichment': log2(observed / mean_null). For vertex permutation null this
            is enrichment relative to a topology-matched baseline; for uniform null this is
            enrichment relative to a flat baseline.

        Example access:
            result["observed"]["gene_score"]
            result["quantile"]["gene_score"]
            result["log2_enrichment"]["gene_score"]

    Examples
    --------
    >>> # Node permutation test with custom mask
    >>> result = network_propagation_with_null(
    ...     graph, ['gene_score'],
    ...     null_strategy='vertex_permutation',
    ...     n_samples=1000,
    ...     mask='measured_genes'
    ... )

    >>> # Edge permutation test
    >>> result = network_propagation_with_null(
    ...     graph, ['pathway_score'],
    ...     null_strategy='edge_permutation',
    ...     n_samples=100,
    ...     burn_in_ratio=10,
    ...     sampling_ratio=0.1
    ... )

    >>> # Attribute-pooled null for many patient-level signals with varying sparsity
    >>> result = network_propagation_with_null(
    ...     graph, patient_attrs,
    ...     null_strategy='attr_pooled_vertex_permutation',
    ...     n_samples=1000,
    ...     mask='measured_features'
    ... )
    """
    # 1. Calculate observed propagated scores
    observed_scores = net_propagate_attributes(
        graph, attributes, propagation_method, additional_propagation_args
    )

    # 2. Get null generator function
    null_generator = _get_null_generator(null_strategy)

    # 3. Generate null distribution
    if null_strategy == NULL_STRATEGIES.UNIFORM:
        null_distribution = null_generator(
            graph=graph,
            attributes=attributes,
            propagation_method=propagation_method,
            additional_propagation_args=additional_propagation_args,
            **null_kwargs,
        )

        # 4a. Uniform null: log2 enrichment vs flat baseline; quantile is not defined
        quantiles = pd.DataFrame(
            np.nan, index=observed_scores.index, columns=observed_scores.columns
        )
        log2_enrichment = _compute_log2_enrichment(observed_scores, null_distribution)

    else:
        null_distribution = null_generator(
            graph=graph,
            attributes=attributes,
            propagation_method=propagation_method,
            additional_propagation_args=additional_propagation_args,
            n_samples=n_samples,
            verbose=verbose,
            **null_kwargs,
        )
        null_distribution = downcast_float_dataframe(null_distribution)

        # 4b. Sampled nulls: both quantile and log2 enrichment vs topology-matched baseline
        quantiles = calculate_quantiles(observed_scores, null_distribution)
        log2_enrichment = _compute_log2_enrichment(observed_scores, null_distribution)

    # 5. Combine into MultiIndex DataFrame with metric as outer level
    results = {
        NET_PROPAGATION_METRICS.OBSERVED: observed_scores,
        NET_PROPAGATION_METRICS.LOG2_ENRICHMENT: log2_enrichment,
    }
    if null_strategy != NULL_STRATEGIES.UNIFORM:
        results[NET_PROPAGATION_METRICS.QUANTILE] = quantiles

    key_order = [
        NET_PROPAGATION_METRICS.OBSERVED,
        NET_PROPAGATION_METRICS.QUANTILE,
        NET_PROPAGATION_METRICS.LOG2_ENRICHMENT,
    ]
    return pd.concat(results, axis=1).reindex(
        [k for k in key_order if k in results], axis=1, level=0
    )


def net_propagate_attributes(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Propagate multiple attributes over a network using a network propagation method.

    Parameters
    ----------
    graph : ig.Graph
        The graph to propagate attributes over.
    attributes : List[str]
        List of attribute names to propagate.
    propagation_method : str
        The network propagation method to use (e.g., 'personalized_pagerank').
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.

    Returns
    -------
    pd.DataFrame
        DataFrame with node names as index and attributes as columns,
        containing the propagated attribute values.
    """

    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    if additional_propagation_args is None:
        additional_propagation_args = {}

    results = []
    for attr in attributes:
        # Validate attributes
        attr_data = _ensure_valid_attribute(
            graph, attr, non_negative=propagation_method.non_negative
        )
        # apply the propagation method
        pr_attr = propagation_method.method(
            graph, attr_data, **additional_propagation_args
        )

        results.append(pr_attr)

    # Get node names once
    names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    return pd.DataFrame(np.column_stack(results), index=names, columns=attributes)


# setup propagation methods


def _pagerank_wrapper(graph: ig.Graph, attr_data: np.ndarray, **kwargs):
    return graph.personalized_pagerank(reset=attr_data.tolist(), **kwargs)


_pagerank_method = PropagationMethod(method=_pagerank_wrapper, non_negative=True)
NET_PROPAGATION_METHODS: dict[str, PropagationMethod] = {
    NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK: _pagerank_method
}
VALID_NET_PROPAGATION_METHODS = NET_PROPAGATION_METHODS.keys()


def _ensure_propagation_method(
    propagation_method: Union[str, PropagationMethod],
) -> PropagationMethod:
    if isinstance(propagation_method, str):
        if propagation_method not in VALID_NET_PROPAGATION_METHODS:
            raise ValueError(f"Invalid propagation method: {propagation_method}")
        return NET_PROPAGATION_METHODS[propagation_method]
    return propagation_method


# other private methods


def _allocate_samples_across_attributes(n_samples: int, n_attributes: int) -> List[int]:
    """Distribute n_samples across n_attributes as evenly as possible.

    Each attribute receives base = n_samples // n_attributes samples. The
    first (n_samples % n_attributes) attributes receive one additional sample,
    so the total returned matches n_samples exactly.

    If n_samples < n_attributes, a warning is issued and only the first
    n_samples attributes contribute one sample each. The remainder receive
    zero samples.

    Parameters
    ----------
    n_samples : int
        Total samples to distribute.
    n_attributes : int
        Number of attributes to distribute across.

    Returns
    -------
    List[int]
        Per-attribute sample counts, summing to min(n_samples, n_attributes)
        in the underflow case or n_samples otherwise.
    """
    if n_samples < n_attributes:
        logger.warning(
            f"n_samples ({n_samples}) is less than n_attributes ({n_attributes}). "
            f"Only the first {n_samples} attributes will contribute to the null. "
            f"Consider increasing n_samples or using vertex_permutation_null."
        )
        return [1 if i < n_samples else 0 for i in range(n_attributes)]

    base = n_samples // n_attributes
    remainder = n_samples % n_attributes
    return [base + (1 if i < remainder else 0) for i in range(n_attributes)]


def _attr_pooled_vertex_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = MASK_KEYWORDS.ATTR,
    n_samples: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate null distribution by permuting each attribute's values within its
    mask, propagating, and pooling the resulting propagated vectors across
    attributes into a shared null distribution.

    Each null sample is generated by selecting one attribute, permuting its
    masked values without replacement, propagating the resulting reset vector,
    and broadcasting the propagated result across all attribute columns. This
    preserves per-attribute sparsity and magnitude profile (each null draw
    inherits the exact non-zero pattern of its source attribute) while still
    pooling propagated outputs across attributes for variance reduction in the
    null estimate.

    The total number of propagation calls is n_samples, matching the cost of
    pooled_vertex_permutation_null and avoiding the n_samples * n_attributes
    cost of plain vertex_permutation_null.

    Sample allocation
    -----------------
    n_samples is distributed across attributes as evenly as possible:
    base = n_samples // n_attributes samples per attribute, with the first
    (n_samples % n_attributes) attributes receiving one additional sample.
    The total number of samples returned exactly equals n_samples, which is
    important for p-value resolution downstream.

    If n_samples < n_attributes, a warning is issued and only the first
    n_samples attributes contribute one sample each. Consider increasing
    n_samples or switching to vertex_permutation_null in this regime.

    Assumptions
    -----------
    All attributes must share an identical mask. Attributes are assumed to
    occupy the same subgraph (same set of measurable vertices), so pooling
    propagated outputs across attributes produces a meaningful shared null.
    Unlike pooled_vertex_permutation_null, this method does not assume
    attributes are exchangeable in magnitude or sparsity — each null draw
    preserves the source attribute's own profile.

    When to prefer this over pooled_vertex_permutation_null
    -------------------------------------------------------
    When attributes share a mask (same measurable subgraph) but differ in
    sparsity or magnitude profile. pooled_vertex_permutation_null mixes values
    across attributes indiscriminately, which causes systematic shifts in
    log2 enrichment for atypical attributes when propagation is nonlinear in
    reset concentration (as with personalized PageRank). This method
    eliminates that bias by keeping each null draw's reset profile faithful
    to a real attribute, while still benefiting from cross-attribute pooling
    at the propagated-output stage.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for. All must share the same mask.
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own
        mask). All attributes must resolve to the same mask.
    n_samples : int
        Total number of null samples to generate, distributed across
        attributes.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with shape (n_samples * n_nodes, n_attributes).
        Each block of n_nodes rows corresponds to one null sample, with the
        same propagated vector broadcast across all attribute columns.
    """
    propagation_method, shared_mask, node_names, null_graph = (
        _setup_vertex_permutation_null(
            graph,
            attributes,
            propagation_method,
            mask,
            verbose,
            require_shared_mask=True,
        )
    )

    masked_indices = np.where(shared_mask)[0]
    n_attributes = len(attributes)

    samples_per_attr = _allocate_samples_across_attributes(n_samples, n_attributes)

    original_values = {attr: np.array(graph.vs[attr]) for attr in attributes}
    all_results = []

    for attr, n_attr_samples in zip(attributes, samples_per_attr):
        if n_attr_samples == 0:
            continue

        attr_values = original_values[attr]
        masked_values = attr_values[masked_indices]

        for _ in range(n_attr_samples):
            permuted_values = np.random.permutation(masked_values)

            null_attr_values = attr_values.copy()
            null_attr_values[masked_indices] = permuted_values
            null_graph.vs[attr] = null_attr_values.tolist()

            broadcast = _propagate_and_broadcast(
                null_graph,
                attr,
                n_attributes,
                propagation_method,
                additional_propagation_args,
            )
            all_results.append(broadcast)

    full_index = node_names * n_samples
    all_data = np.vstack(all_results)
    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def _build_pooled_universe(
    graph: ig.Graph, attributes: List[str], mask: np.ndarray
) -> np.ndarray:
    """Collect all masked values across all attributes into a single array.

    Includes zeros so that the pooled universe reflects the joint distribution of
    magnitude and sparsity across attributes. Sampling from this universe yields
    null reset vectors whose expected sparsity matches the pooled population
    average, rather than forcing every masked position to be non-zero.
    """
    universe_parts = []
    for attr in attributes:
        attr_values = np.array(graph.vs[attr])
        masked_values = attr_values[mask]
        universe_parts.append(masked_values)

    universe = np.concatenate(universe_parts)

    if not np.any(universe > 0):
        raise ValueError(
            "All masked values are zero across every attribute. "
            "Cannot construct a meaningful pooled null."
        )

    return universe


def _compute_log2_enrichment(
    observed: pd.DataFrame,
    null_df: pd.DataFrame,
    epsilon: float = 1e-10,
) -> pd.DataFrame:
    """
    Compute log2 enrichment of observed scores relative to the mean null distribution.

    Parameters
    ----------
    observed : pd.DataFrame
        DataFrame with features as index and attributes as columns containing
        observed propagated scores.
    null_df : pd.DataFrame
        Stacked null samples with features as index (multiple rows per feature)
        and attributes as columns. Same format as output of null generator functions.
    epsilon : float
        Small value added to null mean to avoid division by zero. Default 1e-10.

    Returns
    -------
    pd.DataFrame
        DataFrame with same structure as observed containing log2(observed / mean_null).
        Positive values indicate observed score exceeds the mean null; negative values
        indicate observed score is below mean null.

    Notes
    -----
    The interpretation of log2_enrichment depends on the null strategy used:
    - Vertex permutation null: enrichment relative to topology-matched baseline;
        a value of 1.0 means the observed score is 2x the mean null score for a
        vertex with the same network position but randomized signal assignment.
    - Uniform null: enrichment relative to a flat baseline; more sensitive to
        topological biases since the null does not account for network structure.
    """
    if not observed.columns.equals(null_df.columns):
        raise ValueError("Column names must match between observed and null data")

    missing_features = set(observed.index) - set(null_df.index)
    if missing_features:
        raise ValueError(f"Missing features in null data: {missing_features}")

    if observed.isna().any().any():
        raise ValueError("NaN values found in observed data")
    if null_df.isna().any().any():
        raise ValueError("NaN values found in null data")

    null_mean = null_df.groupby(level=0).mean()

    # Align to observed index order since groupby may sort differently
    null_mean = null_mean.reindex(observed.index)

    log2_enrichment = np.log2(observed / (null_mean + epsilon))

    return pd.DataFrame(log2_enrichment, index=observed.index, columns=observed.columns)


def _edge_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    burn_in_ratio: float = 10,
    sampling_ratio: float = 0.1,
    n_samples: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate null distribution by edge rewiring and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to use (values unchanged by rewiring).
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    burn_in_ratio : float
        Multiplier for initial rewiring.
    sampling_ratio : float
        Proportion of edges to rewire between samples.
    n_samples : int
        Number of null samples to generate.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null samples from rewired network.
        Shape: (n_samples * n_nodes, n_attributes)
    """

    # Validate attributes
    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    # Setup rewired graph
    null_graph = graph.copy()
    n_edges = len(null_graph.es)

    # Initial burn-in
    null_graph.rewire(n=burn_in_ratio * n_edges)

    # Get node names
    node_names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Pre-allocate for results
    all_results = []

    # Generate samples
    for _ in range(n_samples):
        # Incremental rewiring
        null_graph.rewire(n=int(sampling_ratio * n_edges))

        # Apply propagation method to rewired graph (attributes unchanged)
        result = net_propagate_attributes(
            null_graph, attributes, propagation_method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def _fit_distribution_parameters(
    graph: ig.Graph,
    attributes: List[str],
    masks: Dict[str, np.ndarray],
    distribution: Any,
    fit_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Fit distribution parameters for each attribute using masked data."""
    params = {}

    for attr in attributes:
        attr_mask = masks[attr]
        attr_values = np.array(graph.vs[attr])
        masked_values = attr_values[attr_mask]
        masked_nonzero = masked_values[masked_values > 0]

        if len(masked_nonzero) == 0:
            raise ValueError(f"No nonzero values in mask for attribute '{attr}'")

        try:
            # Let SciPy handle parameter estimation and validation
            fitted_params = distribution.fit(masked_nonzero, **fit_kwargs)

            params[attr] = {
                "fitted_params": fitted_params,
                "mask": attr_mask,
                "distribution": distribution,
            }

        except Exception as e:
            dist_name = (
                distribution.name
                if hasattr(distribution, "name")
                else str(distribution)
            )
            raise ValueError(
                f"Failed to fit {dist_name} distribution to attribute '{attr}': {str(e)}"
            )

    return params


def _generate_parametric_null_sample(
    null_graph: ig.Graph,
    attributes: List[str],
    params: Dict[str, Dict[str, Any]],
    ensure_nonnegative: bool,
) -> None:
    """Generate one null sample by modifying graph attributes in-place."""
    for attr in attributes:
        attr_mask = params[attr]["mask"]
        fitted_params = params[attr]["fitted_params"]
        distribution = params[attr]["distribution"]

        # Generate values for masked nodes using fitted distribution
        null_attr_values = np.zeros(null_graph.vcount())
        n_masked = attr_mask.sum()

        # Sample from fitted distribution
        sampled_values = distribution.rvs(*fitted_params, size=n_masked)

        # Ensure non-negative if requested (common for PageRank)
        if ensure_nonnegative:
            # warning if there are negative samples since this suggests that the wrong
            # distribution is being used
            if np.any(sampled_values < 0):
                logger.warning(
                    f"Negative samples for attribute '{attr}' suggest that the wrong distribution is being used"
                )
            sampled_values = np.maximum(sampled_values, 0)

        null_attr_values[attr_mask] = sampled_values
        null_graph.vs[attr] = null_attr_values.tolist()


def _get_distribution_object(distribution: Union[str, Any]) -> Any:
    """Get scipy.stats distribution object from string name or object."""
    if isinstance(distribution, str):
        try:
            return getattr(stats, distribution)
        except AttributeError:
            raise ValueError(
                f"Unknown distribution: '{distribution}'. "
                f"Must be a valid scipy.stats distribution name."
            )
    return distribution


def _parametric_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    distribution: Union[str, Any] = PARAMETRIC_NULL_DEFAULT_DISTRIBUTION,
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = MASK_KEYWORDS.ATTR,
    n_samples: int = 100,
    fit_kwargs: Optional[dict] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate parametric null distribution by fitting scipy.stats distribution to observed values.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for.
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    distribution : str or scipy.stats distribution
        Distribution to fit. Can be:
        - String name (e.g., 'norm', 'gamma', 'beta', 'expon', 'lognorm')
        - SciPy stats distribution object (e.g., stats.gamma, stats.beta)
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
    n_samples : int
        Number of null samples to generate.
    fit_kwargs : dict, optional
        Additional arguments passed to distribution.fit() method.
        Common examples:
        - For gamma: {'floc': 0} to fix location at 0
        - For beta: {'floc': 0, 'fscale': 1} to fix support to [0,1]
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with specified parametric distribution over masked nodes.
        Shape: (n_samples * n_nodes, n_attributes)

    Examples
    --------
    >>> # Gaussian null (default)
    >>> result = parametric_null(graph, ['gene_expression'])

    >>> # Gamma null for positive-valued data
    >>> result = parametric_null(graph, ['gene_expression'],
    ...                         distribution='gamma',
    ...                         fit_kwargs={'floc': 0})

    >>> # Beta null for data in [0,1]
    >>> result = parametric_null(graph, ['probabilities'],
    ...                         distribution='beta')

    >>> # Custom scipy distribution
    >>> result = parametric_null(graph, ['counts'],
    ...                         distribution=stats.poisson)
    """
    # Setup
    dist = _get_distribution_object(distribution)
    if fit_kwargs is None:
        fit_kwargs = {}

    # Validate attributes
    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    # Parse mask input and get masks
    mask_specs = _parse_mask_input(mask, attributes, verbose=verbose)
    masks = _get_attribute_masks(graph, mask_specs)

    # Fit distribution parameters for each attribute
    params = _fit_distribution_parameters(graph, attributes, masks, dist, fit_kwargs)

    # Get node names for output
    node_names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Create null graph once (will overwrite attributes in each sample)
    null_graph = graph.copy()
    all_results = []

    # Generate samples
    for i in range(n_samples):
        # Generate null sample (modifies null_graph in-place)
        _generate_parametric_null_sample(
            null_graph,
            attributes,
            params,
            ensure_nonnegative=propagation_method.non_negative,
        )

        # Apply propagation method to null graph
        result = net_propagate_attributes(
            null_graph, attributes, propagation_method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def _pooled_vertex_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = MASK_KEYWORDS.ATTR,
    n_samples: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate null distribution by sampling from a pooled empirical universe across
    all attributes and applying the propagation method.

    Rather than permuting each attribute independently (as in vertex_permutation_null),
    this strategy constructs a single empirical universe by merging all masked values
    (including zeros) across all attributes. Each null sample draws n_masked values
    from this universe with replacement, assigns them to masked vertices, and
    propagates the resulting synthetic attribute vector. The propagated result is
    broadcast across all attribute columns so that every attribute is compared
    against the same shared null distribution.

    This reduces propagation calls from n_attributes * n_samples to n_samples,
    making it suitable for large attribute sets (e.g. 200+ patient-level summaries).

    Assumptions
    -----------
    All attributes must share an identical mask. Attributes are assumed to be
    exchangeable in both magnitude and sparsity — the pooled universe mixes values
    across attributes indiscriminately, so the null implicitly treats any
    (value, zero/nonzero) pattern as equally likely for any attribute. This is a
    caller contract, validated only for mask equality.

    When this assumption is appropriate
    -----------------------------------
    Use this strategy when attributes have similar magnitude profiles and similar
    sparsity (similar fraction of non-zero values per attribute). In that regime,
    pooled sampling provides variance reduction in the null estimate without
    introducing systematic bias.

    When to prefer an alternative
    -----------------------------
    If attributes differ meaningfully in sparsity, the pooled null's expected
    sparsity is the population average and will not match attributes that are much
    sparser or denser than typical. For personalized PageRank specifically, this
    causes systematic shifts in log2 enrichment for atypical attributes (sparser
    attributes shift left, denser shift right) because PPR responds nonlinearly to
    reset concentration. In that case, prefer attr_pooled_vertex_permutation,
    which preserves per-attribute sparsity while still pooling propagated outputs
    across attributes for variance reduction.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for. All must share the same mask.
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
        All attributes must resolve to the same mask.
    n_samples : int
        Number of null samples to generate.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with shape (n_samples * n_nodes, n_attributes).
        Each block of n_nodes rows corresponds to one null sample, with the same
        propagated vector broadcast across all attribute columns.
    """
    propagation_method, shared_mask, node_names, null_graph = (
        _setup_vertex_permutation_null(
            graph,
            attributes,
            propagation_method,
            mask,
            verbose,
            require_shared_mask=True,
        )
    )

    universe = _build_pooled_universe(graph, attributes, shared_mask)
    n_masked = shared_mask.sum()
    n_attributes = len(attributes)
    _POOLED_ATTR = "__pooled_null__"

    all_results = []

    for _ in range(n_samples):
        sampled_values = np.random.choice(universe, size=n_masked, replace=True)

        null_attr_values = np.zeros(graph.vcount())
        null_attr_values[shared_mask] = sampled_values
        null_graph.vs[_POOLED_ATTR] = null_attr_values.tolist()

        broadcast = _propagate_and_broadcast(
            null_graph,
            _POOLED_ATTR,
            n_attributes,
            propagation_method,
            additional_propagation_args,
        )
        all_results.append(broadcast)

    full_index = node_names * n_samples
    all_data = np.vstack(all_results)
    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def _propagate_and_broadcast(
    null_graph: ig.Graph,
    source_attr: str,
    n_attributes: int,
    propagation_method: PropagationMethod,
    additional_propagation_args: Optional[dict],
) -> np.ndarray:
    """Propagate a single attribute on the null graph and broadcast across columns.

    Used by null generators that produce one propagated vector per sample and
    share it across all attribute columns of the output null tensor.

    Parameters
    ----------
    null_graph : ig.Graph
        Graph with the source attribute already set to the null reset values.
    source_attr : str
        Name of the attribute to propagate.
    n_attributes : int
        Number of columns to broadcast across in the output.
    propagation_method : PropagationMethod
        Normalized propagation method.
    additional_propagation_args : dict, optional
        Forwarded to net_propagate_attributes.

    Returns
    -------
    np.ndarray
        Array of shape (n_nodes, n_attributes) where every column is a copy
        of the propagated vector.
    """
    result = net_propagate_attributes(
        null_graph, [source_attr], propagation_method, additional_propagation_args
    )
    propagated = result[source_attr].values
    return np.column_stack([propagated] * n_attributes)


def _setup_vertex_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[str, PropagationMethod],
    mask: Optional[Union[str, np.ndarray, List, Dict]],
    verbose: bool,
    require_shared_mask: bool,
) -> tuple:
    """Shared setup for vertex-permutation-style null generators.

    Validates the propagation method and attributes, parses masks, copies the
    graph, and resolves node names. Optionally enforces that all attributes
    share an identical mask, which is required by null generators that pool
    propagation outputs across attributes.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to validate and resolve masks for.
    propagation_method : str or PropagationMethod
        Propagation method, normalized via _ensure_propagation_method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification, parsed via _parse_mask_input.
    verbose : bool
        Forwarded to _parse_mask_input.
    require_shared_mask : bool
        If True, validate that all attribute masks are identical and return
        a single shared mask. If False, return the full per-attribute mask
        dictionary.

    Returns
    -------
    propagation_method : PropagationMethod
        Normalized propagation method.
    masks : np.ndarray or Dict[str, np.ndarray]
        If require_shared_mask is True, a single boolean array. Otherwise a
        dict mapping attribute names to their boolean masks.
    node_names : list
        Vertex names from the graph, falling back to integer indices if the
        name attribute is absent.
    null_graph : ig.Graph
        A fresh copy of the input graph for the caller to mutate.
    """
    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    mask_specs = _parse_mask_input(mask, attributes, verbose=verbose)
    masks = _get_attribute_masks(graph, mask_specs)

    if require_shared_mask:
        _validate_masks_identical(masks, attributes)
        masks = masks[attributes[0]]

    node_names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    null_graph = graph.copy()

    return propagation_method, masks, node_names, null_graph


def _uniform_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = MASK_KEYWORDS.ATTR,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate uniform null distribution over masked nodes and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for.
    propagation_method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null sample with uniform distribution over masked nodes.
        Shape: (n_nodes, n_attributes)
    """

    # Validate attributes
    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    # Parse mask input
    mask_specs = _parse_mask_input(mask, attributes, verbose=verbose)
    masks = _get_attribute_masks(graph, mask_specs)

    # Create null graph with uniform attributes
    # we'll use these updated attributes when calling net_propagate_attributes() below
    null_graph = graph.copy()

    for _, attr in enumerate(attributes):
        attr_mask = masks[attr]
        n_masked = attr_mask.sum()

        if n_masked == 0:
            raise ValueError(f"No nodes in mask for attribute '{attr}'")

        # Check for constant attribute values when mask is the same as attribute
        if isinstance(mask_specs[attr], str) and mask_specs[attr] == attr:
            attr_values = np.array(graph.vs[attr])
            nonzero_values = attr_values[attr_values > 0]
            if len(np.unique(nonzero_values)) == 1:
                logger.warning(
                    f"Attribute '{attr}' has constant non-zero values, uniform null may not be meaningful."
                )

        # Set uniform values for masked nodes
        null_attr_values = np.zeros(graph.vcount())
        null_attr_values[attr_mask] = 1.0 / n_masked
        null_graph.vs[attr] = null_attr_values.tolist()

    # Apply propagation method to null graph
    return net_propagate_attributes(
        null_graph, attributes, propagation_method, additional_propagation_args
    )


def _validate_masks_identical(
    masks: Dict[str, np.ndarray], attributes: List[str]
) -> None:
    """Validate that all attribute masks are identical."""
    reference = masks[attributes[0]]
    for attr in attributes[1:]:
        if not np.array_equal(reference, masks[attr]):
            raise ValueError(
                f"Attribute '{attr}' has a different mask than '{attributes[0]}'. "
                "pooled_vertex_permutation_null requires all attributes to share "
                "an identical mask. Consider running vertex_permutation_null "
                "separately per mask group."
            )


def _validate_vertex_attributes(
    graph: ig.Graph, attributes: List[str], propagation_method: str
) -> None:
    """Validate vertex attributes for propagation method."""

    propagation_method = _ensure_propagation_method(propagation_method)

    # check that the attributes are numeric and non-negative if required
    for attr in attributes:
        _ = _ensure_valid_attribute(
            graph, attr, non_negative=propagation_method.non_negative
        )

    return None


def _vertex_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    propagation_method: Union[
        str, PropagationMethod
    ] = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = MASK_KEYWORDS.ATTR,
    replace: bool = False,
    n_samples: int = 100,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generate null distribution by permuting vertex attribute values and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to permute.
    propagation_method : str or PropagationMethod
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
    replace : bool
        Whether to sample with replacement.
    n_samples : int
        Number of null samples to generate.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with permuted attribute values.
        Shape: (n_samples * n_nodes, n_attributes)
    """
    propagation_method, masks, node_names, null_graph = _setup_vertex_permutation_null(
        graph,
        attributes,
        propagation_method,
        mask,
        verbose,
        require_shared_mask=False,
    )

    original_values = {attr: np.array(graph.vs[attr]) for attr in attributes}
    all_results = []

    for _ in range(n_samples):
        for attr in attributes:
            attr_mask = masks[attr]
            masked_indices = np.where(attr_mask)[0]
            masked_values = original_values[attr][masked_indices]

            null_attr_values = original_values[attr].copy()

            if replace:
                permuted_values = np.random.choice(
                    masked_values, size=len(masked_values), replace=True
                )
            else:
                permuted_values = np.random.permutation(masked_values)

            null_attr_values[masked_indices] = permuted_values
            null_graph.vs[attr] = null_attr_values.tolist()

        result = net_propagate_attributes(
            null_graph, attributes, propagation_method, additional_propagation_args
        )
        all_results.append(result)

    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])
    return pd.DataFrame(all_data, index=full_index, columns=attributes)


# Null generator registry
NULL_GENERATORS = {
    NULL_STRATEGIES.ATTR_POOLED_VERTEX_PERMUTATION: _attr_pooled_vertex_permutation_null,
    NULL_STRATEGIES.EDGE_PERMUTATION: _edge_permutation_null,
    NULL_STRATEGIES.PARAMETRIC: _parametric_null,
    NULL_STRATEGIES.POOLED_VERTEX_PERMUTATION: _pooled_vertex_permutation_null,
    NULL_STRATEGIES.UNIFORM: _uniform_null,
    NULL_STRATEGIES.VERTEX_PERMUTATION: _vertex_permutation_null,
}


def _get_null_generator(strategy: str):
    """Get null generator function by name."""
    if strategy not in VALID_NULL_STRATEGIES:
        raise ValueError(
            f"Unknown null strategy: {strategy}. Available: {VALID_NULL_STRATEGIES}"
        )
    return NULL_GENERATORS[strategy]
