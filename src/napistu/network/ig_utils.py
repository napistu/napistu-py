"""
General utilities for working with igraph.Graph objects.

This module contains utilities that can be broadly applied to any igraph.Graph
object, not specific to NapistuGraph subclasses.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Optional, Sequence, List, Dict, Union

import igraph as ig
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_graph_summary(graph: ig.Graph) -> dict[str, Any]:
    """
    Calculate common summary statistics for an igraph network.

    Parameters
    ----------
    graph : ig.Graph
        The input network.

    Returns
    -------
    dict
        A dictionary of summary statistics with the following keys:
        - n_edges (int): number of edges
        - n_vertices (int): number of vertices
        - n_components (int): number of weakly connected components
        - stats_component_sizes (dict): summary statistics for the component sizes
        - top10_large_components (list[dict]): the top 10 largest components with 10 example vertices
        - top10_smallest_components (list[dict]): the top 10 smallest components with 10 example vertices
        - average_path_length (float): the average shortest path length between all vertices
        - top10_betweenness (list[dict]): the top 10 vertices by betweenness centrality
        - top10_harmonic_centrality (list[dict]): the top 10 vertices by harmonic centrality
    """
    stats = {}
    stats["n_edges"] = graph.ecount()
    stats["n_vertices"] = graph.vcount()
    components = graph.components(mode="weak")
    stats["n_components"] = len(components)
    component_sizes = [len(c) for c in components]
    stats["stats_component_sizes"] = pd.Series(component_sizes).describe().to_dict()

    # get the top 10 largest components and 10 example nodes
    stats["top10_large_components"] = _get_top_n_component_stats(
        graph, components, component_sizes, n=10, ascending=False
    )
    stats["top10_smallest_components"] = _get_top_n_component_stats(
        graph, components, component_sizes, n=10, ascending=True
    )

    stats["average_path_length"] = graph.average_path_length()

    # Top 10 by betweenness and harmonic centrality
    betweenness = graph.betweenness()
    stats["top10_betweenness"] = _get_top_n_nodes(
        graph, betweenness, "betweenness", n=10
    )
    harmonic = graph.harmonic_centrality()
    stats["top10_harmonic_centrality"] = _get_top_n_nodes(
        graph, harmonic, "harmonic_centrality", n=10
    )

    return stats


def filter_to_largest_subgraph(graph: ig.Graph) -> ig.Graph:
    """
    Filter an igraph to its largest weakly connected component.

    Parameters
    ----------
    graph : ig.Graph
        The input network.

    Returns
    -------
    ig.Graph
        The largest weakly connected component.
    """
    component_members = graph.components(mode="weak")
    component_sizes = [len(x) for x in component_members]

    top_component_members = [
        m
        for s, m in zip(component_sizes, component_members)
        if s == max(component_sizes)
    ][0]

    largest_subgraph = graph.induced_subgraph(top_component_members)
    return largest_subgraph


def filter_to_largest_subgraphs(graph: ig.Graph, top_k: int) -> list[ig.Graph]:
    """
    Filter an igraph to its largest weakly connected components.

    Parameters
    ----------
    graph : ig.Graph
        The input network.
    top_k : int
        The number of largest components to return.

    Returns
    -------
    list[ig.Graph]
        A list of the top K largest components as graphs.
    """
    if top_k < 1:
        raise ValueError("top_k must be 1 or greater.")

    component_members = graph.components(mode="weak")
    if not component_members:
        return []

    component_sizes = [len(x) for x in component_members]

    # Sort components by size in descending order
    sorted_components = sorted(
        zip(component_sizes, component_members), key=lambda x: x[0], reverse=True
    )

    # Return a list of the top K subgraphs
    top_k_components = sorted_components[:top_k]
    return [graph.induced_subgraph(members) for _, members in top_k_components]


def graph_to_pandas_dfs(graph: ig.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert an igraph to Pandas DataFrames for vertices and edges.

    Parameters
    ----------
    graph : ig.Graph
        An igraph network.

    Returns
    -------
    vertices : pandas.DataFrame
        A table with one row per vertex.
    edges : pandas.DataFrame
        A table with one row per edge.
    """
    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in graph.vs]
    )
    edges = pd.DataFrame(
        [
            {**{"source": e.source, "target": e.target}, **e.attributes()}
            for e in graph.es
        ]
    )
    return vertices, edges


def create_induced_subgraph(
    graph: ig.Graph,
    vertices: Optional[list[str]] = None,
    n_vertices: int = 5000,
) -> ig.Graph:
    """
    Create a subgraph from an igraph including a set of vertices and their connections.

    Parameters
    ----------
    graph : ig.Graph
        The input network.
    vertices : list, optional
        List of vertex names to include. If None, a random sample is selected.
    n_vertices : int, optional
        Number of vertices to sample if `vertices` is None. Default is 5000.

    Returns
    -------
    ig.Graph
        The induced subgraph.
    """
    if vertices is not None:
        selected_vertices = vertices
    else:
        # Assume vertices have a 'name' attribute, fallback to indices
        if "name" in graph.vs.attributes():
            vertex_names = graph.vs["name"]
        else:
            vertex_names = list(range(graph.vcount()))
        selected_vertices = random.sample(
            vertex_names, min(n_vertices, len(vertex_names))
        )

    subgraph = graph.induced_subgraph(selected_vertices)
    return subgraph


def validate_edge_attributes(graph: ig.Graph, edge_attributes: list[str]) -> None:
    """
    Validate that all required edge attributes exist in an igraph.

    Parameters
    ----------
    graph : ig.Graph
        The network.
    edge_attributes : list of str
        List of edge attribute names to check.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If "edge_attributes" is not a list or str.
    ValueError
        If any required edge attribute is missing from the graph.
    """
    if isinstance(edge_attributes, list):
        attrs = edge_attributes
    elif isinstance(edge_attributes, str):
        attrs = [edge_attributes]
    else:
        raise TypeError('"edge_attributes" must be a list or str')

    available_attributes = graph.es[0].attributes().keys() if graph.ecount() > 0 else []
    missing_attributes = set(attrs).difference(available_attributes)
    n_missing_attrs = len(missing_attributes)

    if n_missing_attrs > 0:
        raise ValueError(
            f"{n_missing_attrs} edge attributes were missing ({', '.join(missing_attributes)}). "
            f"The available edge attributes are {', '.join(available_attributes)}"
        )

    return None


def validate_vertex_attributes(graph: ig.Graph, vertex_attributes: list[str]) -> None:
    """
    Validate that all required vertex attributes exist in an igraph.

    Parameters
    ----------
    graph : ig.Graph
        The network.
    vertex_attributes : list of str
        List of vertex attribute names to check.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If "vertex_attributes" is not a list or str.
    ValueError
        If any required vertex attribute is missing from the graph.
    """
    if isinstance(vertex_attributes, list):
        attrs = vertex_attributes
    elif isinstance(vertex_attributes, str):
        attrs = [vertex_attributes]
    else:
        raise TypeError('"vertex_attributes" must be a list or str')

    available_attributes = graph.vs[0].attributes().keys() if graph.vcount() > 0 else []
    missing_attributes = set(attrs).difference(available_attributes)
    n_missing_attrs = len(missing_attributes)

    if n_missing_attrs > 0:
        raise ValueError(
            f"{n_missing_attrs} vertex attributes were missing ({', '.join(missing_attributes)}). "
            f"The available vertex attributes are {', '.join(available_attributes)}"
        )

    return None


# Internal utility functions


def _get_top_n_idx(arr: Sequence, n: int, ascending: bool = False) -> Sequence[int]:
    """Returns the indices of the top n values in an array

    Args:
        arr (Sequence): An array of values
        n (int): The number of top values to return
        ascending (bool, optional): Whether to return the top or bottom n values. Defaults to False.

    Returns:
        Sequence[int]: The indices of the top n values
    """
    order = np.argsort(arr)
    if ascending:
        return order[:n]  # type: ignore
    else:
        return order[-n:][::-1]  # type: ignore


def _get_top_n_objects(
    object_vals: Sequence, objects: Sequence, n: int = 10, ascending: bool = False
) -> list:
    """Get the top N objects based on a ranking measure."""
    idxs = _get_top_n_idx(object_vals, n, ascending=ascending)
    top_objects = [objects[idx] for idx in idxs]
    return top_objects


def _get_top_n_component_stats(
    graph: ig.Graph,
    components,
    component_sizes: Sequence[int],
    n: int = 10,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """
    Summarize the top N components' network properties.

    Parameters
    ----------
    graph : ig.Graph
        The network.
    components : list
        List of components (as lists of vertex indices).
    component_sizes : Sequence[int]
        Sizes of each component.
    n : int, optional
        Number of top components to return. Default is 10.
    ascending : bool, optional
        If True, return smallest components; otherwise, largest. Default is False.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'n': size of the component
        - 'examples': up to 10 example vertex attribute dicts from the component
    """
    top_components = _get_top_n_objects(component_sizes, components, n, ascending)
    top_component_stats = [
        {"n": len(c), "examples": [graph.vs[n].attributes() for n in c[:10]]}
        for c in top_components
    ]
    return top_component_stats


def _get_top_n_nodes(
    graph: ig.Graph,
    vals: Sequence,
    val_name: str,
    n: int = 10,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """
    Get the top N nodes by a node attribute.

    Parameters
    ----------
    graph : ig.Graph
        The network.
    vals : Sequence
        Sequence of node attribute values.
    val_name : str
        Name of the attribute.
    n : int, optional
        Number of top nodes to return. Default is 10.
    ascending : bool, optional
        If True, return nodes with smallest values; otherwise, largest. Default is False.

    Returns
    -------
    list of dict
        Each dict contains the value and the node's attributes.
    """
    top_idxs = _get_top_n_idx(vals, n, ascending=ascending)
    top_node_attrs = [graph.vs[idx].attributes() for idx in top_idxs]
    top_vals = [vals[idx] for idx in top_idxs]
    return [{val_name: val, **node} for val, node in zip(top_vals, top_node_attrs)]


def _parse_mask_input(
    mask_input: Optional[Union[str, np.ndarray, List, Dict]],
    attributes: List[str],
    verbose: bool = False,
) -> Dict[str, Union[str, np.ndarray, List, None]]:
    """
    Parse mask input and convert to attribute-specific mask specifications.

    Parameters
    ----------
    mask_input : str, np.ndarray, List, Dict, or None
        Mask specification that can be:
        - None: use all nodes for all attributes
        - "attr": use each attribute as its own mask
        - np.ndarray/List: use same mask for all attributes
        - Dict: attribute-specific mask specifications
    attributes : List[str]
        List of attribute names.
    verbose : bool, optional
        Whether to print the mask input parsing result. Default is False.

    Returns
    -------
    Dict[str, Union[str, np.ndarray, List, None]]
        Dictionary mapping each attribute to its mask specification.
    """
    if mask_input is None:
        masks = {attr: None for attr in attributes}
    elif isinstance(mask_input, str):
        if mask_input == "attr":
            masks = {attr: attr for attr in attributes}
        else:
            # Single attribute name used for all
            masks = {attr: mask_input for attr in attributes}
    elif isinstance(mask_input, (np.ndarray, list)):
        # Same mask for all attributes
        masks = {attr: mask_input for attr in attributes}
    elif isinstance(mask_input, dict):
        # Validate all attributes are present
        for attr in attributes:
            if attr not in mask_input:
                raise ValueError(f"Attribute '{attr}' not found in mask dictionary")
        masks = mask_input
    else:
        raise ValueError(f"Invalid mask input type: {type(mask_input)}")

    if verbose:
        _print_mask_input_result(masks)

    return masks


def _get_attribute_masks(
    graph: ig.Graph,
    mask_specs: Dict[str, Union[str, np.ndarray, List, None]],
) -> Dict[str, np.ndarray]:
    """
    Generate boolean masks for each attribute based on specifications.

    Parameters
    ----------
    graph : ig.Graph
        Input graph.
    mask_specs : Dict[str, Union[str, np.ndarray, List, None]]
        Dictionary mapping each attribute to its mask specification.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each attribute to its boolean mask array.
    """
    n_nodes = graph.vcount()
    masks = {}

    invalid_attrs = set(mask_specs.keys()).difference(graph.vs.attributes())
    if invalid_attrs:
        raise ValueError(f"Attributes {invalid_attrs} not found in graph")

    for attr in mask_specs.keys():

        mask_spec = mask_specs[attr]

        if mask_spec is None:
            masks[attr] = np.ones(n_nodes, dtype=bool)
        elif isinstance(mask_spec, str):
            attr_values = np.array(graph.vs[mask_spec])
            masks[attr] = attr_values > 0
        elif isinstance(mask_spec, np.ndarray):
            masks[attr] = mask_spec.astype(bool)
        elif isinstance(mask_spec, list):
            mask_array = np.zeros(n_nodes, dtype=bool)
            if isinstance(mask_spec[0], str):
                # Node names
                node_names = (
                    graph.vs["name"] if "name" in graph.vs.attributes() else None
                )
                if node_names is None:
                    raise ValueError("Graph has no 'name' attribute for string mask")
                for name in mask_spec:
                    idx = node_names.index(name)
                    mask_array[idx] = True
            else:
                # Node indices
                mask_array[mask_spec] = True
            masks[attr] = mask_array
        else:
            raise ValueError(
                f"Invalid mask specification for attribute '{attr}': {type(mask_spec)}"
            )

    return masks


def _print_mask_input_result(masks):
    """
    Print a readable summary of the result of _parse_mask_input(mask_input, attributes).
    Shows each attribute and its corresponding mask specification.
    """

    logger.info("Mask input parsing result:")
    for attr, spec in masks.items():
        logger.info(f"  Attribute: {attr!r} -> Mask spec: {repr(spec)}")


def _ensure_valid_attribute(graph: ig.Graph, attribute: str, non_negative: bool = True):
    """
    Ensure a vertex attribute is present, numeric, finite, and optionally non-negative for all vertices.

    This utility checks that the specified vertex attribute exists, is numeric, and (optionally) non-negative
    for all vertices in the graph. Missing or None values are treated as 0. Raises ValueError
    if the attribute is missing for all vertices, if all values are zero, or if any value is negative (if non_negative=True).

    Parameters
    ----------
    graph : NapistuGraph or ig.Graph
        The input graph (NapistuGraph or igraph.Graph).
    attribute : str
        The name of the vertex attribute to check.
    non_negative : bool, default True
        Whether to require all values to be non-negative.

    Returns
    -------
    np.ndarray
        Array of attribute values (with missing/None replaced by 0).

    Raises
    ------
    ValueError
        If the attribute is missing for all vertices, all values are zero, or any value is negative (if non_negative=True).
    """
    all_missing = all(
        (attribute not in v.attributes() or v[attribute] is None) for v in graph.vs
    )
    if all_missing:
        raise ValueError(f"Vertex attribute '{attribute}' is missing for all vertices.")

    values = [
        (
            v[attribute]
            if (attribute in v.attributes() and v[attribute] is not None)
            else 0.0
        )
        for v in graph.vs
    ]

    arr = np.array(values, dtype=float)

    if np.all(arr == 0):
        raise ValueError(
            f"Vertex attribute '{attribute}' is zero for all vertices; cannot use as reset vector."
        )
    if non_negative and np.any(arr < 0):
        raise ValueError(f"Attribute '{attribute}' contains negative values.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Attribute '{attribute}' contains non-finite values (NaN or inf)."
        )

    return arr
