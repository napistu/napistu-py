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

logger = logging.getLogger(__name__)


@dataclass
class PropagationMethod:
    method: callable
    non_negative: bool


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
    3. Compares observed vs null using quantiles (for sampled nulls) or ratios (for uniform)

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

    **Uniform ('uniform'):** A quick, qualitative readout. Generates a uniform null distribution over masked nodes and
    takes the ratio of observed network propagation score.

    **Parametric ('parametric'):** Similar to node permutation but rather than sampling observed values sample
    draws from a distribution fit to the observed values. First fits a parametric distribution to the observed scores
    and then samples `n_samples` null samples for each vertex to compare observed to null quantiles.

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
        Null distribution strategy. One of: 'uniform', 'parametric', 'vertex_permutation', 'edge_permutation'.
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
        DataFrame with same structure as observed scores containing:
        - For uniform null: observed/uniform ratios
        - For other nulls: quantiles (proportion of null values <= observed values)

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
    """
    # 1. Calculate observed propagated scores
    observed_scores = net_propagate_attributes(
        graph, attributes, propagation_method, additional_propagation_args
    )

    # 2. Get null generator function
    null_generator = get_null_generator(null_strategy)

    # 3. Generate null distribution
    if null_strategy == NULL_STRATEGIES.UNIFORM:
        # Uniform null doesn't take n_samples
        null_distribution = null_generator(
            graph=graph,
            attributes=attributes,
            propagation_method=propagation_method,
            additional_propagation_args=additional_propagation_args,
            **null_kwargs,
        )

        # 4a. For uniform null: calculate observed/uniform ratios
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        ratios = observed_scores / (null_distribution + epsilon)
        return ratios

    else:
        # Other nulls take n_samples
        null_distribution = null_generator(
            graph=graph,
            attributes=attributes,
            propagation_method=propagation_method,
            additional_propagation_args=additional_propagation_args,
            n_samples=n_samples,
            verbose=verbose,
            **null_kwargs,
        )

        # 4b. For sampled nulls: calculate quantiles
        return calculate_quantiles(observed_scores, null_distribution)


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


def uniform_null(
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


def parametric_null(
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


def vertex_permutation_null(
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
    # Validate attributes
    propagation_method = _ensure_propagation_method(propagation_method)
    _validate_vertex_attributes(graph, attributes, propagation_method)

    # Parse mask input
    mask_specs = _parse_mask_input(mask, attributes, verbose=verbose)
    masks = _get_attribute_masks(graph, mask_specs)

    # Get original attribute values
    original_values = {}
    for attr in attributes:
        original_values[attr] = np.array(graph.vs[attr])

    # Get node names
    node_names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Pre-allocate for results
    all_results = []

    # Generate samples
    # we'll only do this once and overwrite the attributes in each sample
    null_graph = graph.copy()

    for _ in range(n_samples):

        # Permute values among masked nodes for each attribute
        for _, attr in enumerate(attributes):
            attr_mask = masks[attr]
            masked_indices = np.where(attr_mask)[0]
            masked_values = original_values[attr][masked_indices]

            # Start with original values
            null_attr_values = original_values[attr].copy()

            if replace:
                # Sample with replacement
                permuted_values = np.random.choice(
                    masked_values, size=len(masked_values), replace=True
                )
            else:
                # Permute without replacement
                permuted_values = np.random.permutation(masked_values)

            null_attr_values[masked_indices] = permuted_values
            null_graph.vs[attr] = null_attr_values.tolist()

        # Apply propagation method to null graph
        result = net_propagate_attributes(
            null_graph, attributes, propagation_method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def edge_permutation_null(
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


# Null generator registry
NULL_GENERATORS = {
    NULL_STRATEGIES.UNIFORM: uniform_null,
    NULL_STRATEGIES.PARAMETRIC: parametric_null,
    NULL_STRATEGIES.VERTEX_PERMUTATION: vertex_permutation_null,
    NULL_STRATEGIES.EDGE_PERMUTATION: edge_permutation_null,
}


def get_null_generator(strategy: str):
    """Get null generator function by name."""
    if strategy not in VALID_NULL_STRATEGIES:
        raise ValueError(
            f"Unknown null strategy: {strategy}. Available: {VALID_NULL_STRATEGIES}"
        )
    return NULL_GENERATORS[strategy]


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
