import inspect
import logging
from typing import Optional, Union, List, Dict

import pandas as pd
import numpy as np
import igraph as ig

from napistu.network.ig_utils import (
    _parse_mask_input,
    _get_attribute_masks,
    _ensure_nonnegative_vertex_attribute,
)
from napistu.statistics.quantiles import calculate_quantiles
from napistu.network.constants import (
    NAPISTU_GRAPH_VERTICES,
    NET_PROPAGATION_DEFS,
    VALID_NET_PROPAGATION_METHODS,
)

logger = logging.getLogger(__name__)


def network_propagation_with_null(
    graph: ig.Graph,
    attributes: List[str],
    null_strategy: str = "node_permutation",
    method: str = "personalized_pagerank",
    additional_propagation_args: Optional[dict] = None,
    n_samples: int = 100,
    **null_kwargs,
) -> pd.DataFrame:
    """
    Apply network propagation to attributes and compare against null distributions.

    This is the main orchestrator function that:
    1. Calculates observed propagated scores
    2. Generates null distribution using specified strategy
    3. Compares observed vs null using quantiles (for sampled nulls) or ratios (for uniform)

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to propagate and test.
    null_strategy : str
        Null distribution strategy. One of: 'uniform', 'gaussian', 'node_permutation', 'edge_permutation'.
    method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    n_samples : int
        Number of null samples to generate (ignored for uniform null).
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
    ...     null_strategy='node_permutation',
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
        graph, attributes, method, additional_propagation_args
    )

    # 2. Get null generator function
    null_generator = get_null_generator(null_strategy)

    # 3. Generate null distribution
    if null_strategy == "uniform":
        # Uniform null doesn't take n_samples
        null_distribution = null_generator(
            graph, attributes, method, additional_propagation_args, **null_kwargs
        )

        # 4a. For uniform null: calculate observed/uniform ratios
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-10
        ratios = observed_scores / (null_distribution + epsilon)
        return ratios

    else:
        # Other nulls take n_samples
        null_distribution = null_generator(
            graph,
            attributes,
            method,
            additional_propagation_args,
            n_samples=n_samples,
            **null_kwargs,
        )

        # 4b. For sampled nulls: calculate quantiles
        return calculate_quantiles(observed_scores, null_distribution)


def net_propagate_attributes(
    graph: ig.Graph,
    attributes: List[str],
    method: str = NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK,
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
    method : str
        The network propagation method to use (e.g., 'personalized_pagerank').
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.

    Returns
    -------
    pd.DataFrame
        DataFrame with node names as index and attributes as columns,
        containing the propagated attribute values.
    """
    results = []
    for attr in attributes:
        attr_data = _ensure_nonnegative_vertex_attribute(graph, attr)
        additional_propagation_args = _validate_additional_propagation_args(
            method, additional_propagation_args
        )

        if method == NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK:
            pr_attr = graph.personalized_pagerank(
                reset=attr_data.tolist(), **additional_propagation_args
            )
        else:
            raise ValueError(
                f"Invalid method: {method}, valid methods are {VALID_NET_PROPAGATION_METHODS}"
            )

        results.append(pr_attr)

    # Get node names once
    names = (
        graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
        if NAPISTU_GRAPH_VERTICES.NAME in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    return pd.DataFrame(np.column_stack(results), index=names, columns=attributes)


def _validate_additional_propagation_args(
    propagation_method: str, additional_propagation_args: dict
) -> dict:
    """
    Validate additional arguments for a network propagation method.

    Parameters
    ----------
    propagation_method : str
        The network propagation method to validate.
    additional_propagation_args : dict
        The additional arguments to validate.

    Returns
    -------
    dict
        The validated additional arguments.

    Raises
    ------
    ValueError
        If the propagation method is invalid or if the additional arguments are invalid.
    """

    NET_PROPAGATION_FUNCTIONS = {
        NET_PROPAGATION_DEFS.PERSONALIZED_PAGERANK: ig.Graph.personalized_pagerank
    }

    if propagation_method not in VALID_NET_PROPAGATION_METHODS:
        raise ValueError(
            f"Invalid method: {propagation_method}, valid methods are {VALID_NET_PROPAGATION_METHODS}"
        )

    # Validate additional_propagation_args
    if additional_propagation_args is None:
        return {}
    else:
        valid_args = set(
            inspect.signature(
                NET_PROPAGATION_FUNCTIONS[propagation_method]
            ).parameters.keys()
        )
        for k in additional_propagation_args:
            if k not in valid_args:
                raise ValueError(f"Invalid argument for personalized_pagerank: {k}")

        return additional_propagation_args


def uniform_null(
    graph: ig.Graph,
    attributes: List[str],
    method: str = "personalized_pagerank",
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = "attr",
) -> pd.DataFrame:
    """
    Generate uniform null distribution over masked nodes and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for.
    method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).

    Returns
    -------
    pd.DataFrame
        Propagated null sample with uniform distribution over masked nodes.
        Shape: (n_nodes, n_attributes)
    """
    # Validate attributes
    for attr in attributes:
        _ensure_nonnegative_vertex_attribute(graph, attr)

    # Parse mask input
    mask_specs = _parse_mask_input(mask, attributes)
    masks = _get_attribute_masks(graph, attributes, mask_specs)

    # Create null graph with uniform attributes
    null_graph = graph.copy()

    for j, attr in enumerate(attributes):
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
        null_graph, attributes, method, additional_propagation_args
    )


def gaussian_null(
    graph: ig.Graph,
    attributes: List[str],
    method: str = "personalized_pagerank",
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = "attr",
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Generate Gaussian null distribution based on observed values and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to generate nulls for.
    method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
    n_samples : int
        Number of null samples to generate.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with Gaussian distribution over masked nodes.
        Shape: (n_samples * n_nodes, n_attributes)
    """
    # Validate attributes
    for attr in attributes:
        _ensure_nonnegative_vertex_attribute(graph, attr)

    # Parse mask input
    mask_specs = _parse_mask_input(mask, attributes)
    masks = _get_attribute_masks(graph, attributes, mask_specs)

    # Calculate mean and std from observed values for each attribute
    params = {}
    for attr in attributes:
        attr_mask = masks[attr]
        attr_values = np.array(graph.vs[attr])
        masked_values = attr_values[attr_mask]
        masked_nonzero = masked_values[masked_values > 0]

        if len(masked_nonzero) == 0:
            raise ValueError(f"No nonzero values in mask for attribute '{attr}'")

        # Check if data seems non-Gaussian
        if (
            np.all(masked_nonzero == masked_nonzero.astype(int))
            and len(np.unique(masked_nonzero)) < 5
        ):
            logger.warning(
                f"Attribute '{attr}' appears to be integer-valued with <5 distinct values, may not be suitable for Gaussian null."
            )

        params[attr] = {
            "mean": np.mean(masked_nonzero),
            "std": np.std(masked_nonzero),
            "mask": attr_mask,
        }

    # Get node names
    node_names = (
        graph.vs["name"]
        if "name" in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Pre-allocate for results
    all_results = []

    # Generate samples
    for i in range(n_samples):
        # Create null graph with Gaussian attributes
        null_graph = graph.copy()

        for j, attr in enumerate(attributes):
            attr_mask = params[attr]["mask"]
            mean = params[attr]["mean"]
            std = params[attr]["std"]

            # Generate Gaussian values for masked nodes
            null_attr_values = np.zeros(graph.vcount())
            n_masked = attr_mask.sum()
            gaussian_values = np.random.normal(mean, std, n_masked)
            # Ensure non-negative values
            gaussian_values = np.maximum(gaussian_values, 0)
            null_attr_values[attr_mask] = gaussian_values
            null_graph.vs[attr] = null_attr_values.tolist()

        # Apply propagation method to null graph
        result = net_propagate_attributes(
            null_graph, attributes, method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def node_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    method: str = "personalized_pagerank",
    additional_propagation_args: Optional[dict] = None,
    mask: Optional[Union[str, np.ndarray, List, Dict]] = "attr",
    replace: bool = False,
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Generate null distribution by permuting node attribute values and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to permute.
    method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    mask : str, np.ndarray, List, Dict, or None
        Mask specification. Default is "attr" (use each attribute as its own mask).
    replace : bool
        Whether to sample with replacement.
    n_samples : int
        Number of null samples to generate.

    Returns
    -------
    pd.DataFrame
        Propagated null samples with permuted attribute values.
        Shape: (n_samples * n_nodes, n_attributes)
    """
    # Validate attributes
    for attr in attributes:
        _ensure_nonnegative_vertex_attribute(graph, attr)

    # Parse mask input
    mask_specs = _parse_mask_input(mask, attributes)
    masks = _get_attribute_masks(graph, attributes, mask_specs)

    # Get original attribute values
    original_values = {}
    for attr in attributes:
        original_values[attr] = np.array(graph.vs[attr])

    # Get node names
    node_names = (
        graph.vs["name"]
        if "name" in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Pre-allocate for results
    all_results = []

    # Generate samples
    for i in range(n_samples):
        # Create null graph with permuted attributes
        null_graph = graph.copy()

        # Permute values among masked nodes for each attribute
        for j, attr in enumerate(attributes):
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
            null_graph, attributes, method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


def edge_permutation_null(
    graph: ig.Graph,
    attributes: List[str],
    method: str = "personalized_pagerank",
    additional_propagation_args: Optional[dict] = None,
    burn_in_ratio: int = 10,
    sampling_ratio: float = 0.1,
    n_samples: int = 100,
) -> pd.DataFrame:
    """
    Generate null distribution by edge rewiring and apply propagation method.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    attributes : List[str]
        Attribute names to use (values unchanged by rewiring).
    method : str
        Network propagation method to apply.
    additional_propagation_args : dict, optional
        Additional arguments to pass to the network propagation method.
    burn_in_ratio : int
        Multiplier for initial rewiring.
    sampling_ratio : float
        Proportion of edges to rewire between samples.
    n_samples : int
        Number of null samples to generate.

    Returns
    -------
    pd.DataFrame
        Propagated null samples from rewired network.
        Shape: (n_samples * n_nodes, n_attributes)
    """
    # Validate attributes
    for attr in attributes:
        _ensure_nonnegative_vertex_attribute(graph, attr)

    # Setup rewired graph
    null_graph = graph.copy()
    n_edges = len(null_graph.es)

    # Initial burn-in
    null_graph.rewire(n=burn_in_ratio * n_edges)

    # Get node names
    node_names = (
        graph.vs["name"]
        if "name" in graph.vs.attributes()
        else list(range(graph.vcount()))
    )

    # Pre-allocate for results
    all_results = []

    # Generate samples
    for i in range(n_samples):
        # Incremental rewiring
        null_graph.rewire(n=int(sampling_ratio * n_edges))

        # Apply propagation method to rewired graph (attributes unchanged)
        result = net_propagate_attributes(
            null_graph, attributes, method, additional_propagation_args
        )
        all_results.append(result)

    # Combine all results
    full_index = node_names * n_samples
    all_data = np.vstack([result.values for result in all_results])

    return pd.DataFrame(all_data, index=full_index, columns=attributes)


# Null generator registry
NULL_GENERATORS = {
    "uniform": uniform_null,
    "gaussian": gaussian_null,
    "node_permutation": node_permutation_null,
    "edge_permutation": edge_permutation_null,
}


def get_null_generator(strategy: str):
    """Get null generator function by name."""
    if strategy not in NULL_GENERATORS:
        raise ValueError(
            f"Unknown null strategy: {strategy}. Available: {list(NULL_GENERATORS.keys())}"
        )
    return NULL_GENERATORS[strategy]
