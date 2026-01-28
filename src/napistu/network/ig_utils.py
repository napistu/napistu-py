"""
General utilities for working with igraph.Graph objects.

This module contains utilities that can be broadly applied to any igraph.Graph
object, not specific to NapistuGraph subclasses.

Public Functions
----------------
create_induced_subgraph(graph: Graph, vertices: Optional[list[str]] = None, n_vertices: int = 5000) -> Graph:
    Create a subgraph from an igraph including a set of vertices and their connections.
define_graph_universe(graph: Graph, vertex_names: Optional[Union[List[str], pd.Series]] = None, edgelist: Optional[pd.DataFrame] = None, observed_only: bool = False, edge_filter_logic: str = 'and', include_self_edges: bool = False) -> Graph:
    Define a graph universe for enrichment-style analyses.
filter_to_largest_subgraph(graph: Graph) -> Graph:
    Filter an igraph to its largest weakly connected component.
filter_to_largest_subgraphs(graph: Graph, top_k: int) -> list[Graph]:
    Filter an igraph to its largest weakly connected components.
get_graph_summary(graph: Graph) -> dict[str, Any]:
    Calculate common summary statistics for an igraph network.
get_merge_keys(graph: Graph, merge_by: str = IGRAPH_DEFS.NAME) -> tuple[str, str, str]:
    Get the merge keys for a graph based on the merge_by parameter.
graph_to_pandas_dfs(graph: Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    Convert an igraph to Pandas DataFrames for vertices and edges.
validate_edge_attributes(graph: Graph, edge_attributes: list[str]) -> None:
    Validate that all required edge attributes exist in an igraph.
validate_vertex_attributes(graph: Graph, vertex_attributes: list[str]) -> None:
    Validate that all required vertex attributes exist in an igraph.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from igraph import Graph

from napistu.network.constants import (
    IGRAPH_DEFS,
    NAPISTU_GRAPH_EDGES,
    UNIVERSE_GATES,
    VALID_UNIVERSE_GATES,
)

logger = logging.getLogger(__name__)


def create_induced_subgraph(
    graph: Graph,
    vertices: Optional[list[str]] = None,
    n_vertices: int = 5000,
) -> Graph:
    """
    Create a subgraph from an igraph including a set of vertices and their connections.

    Parameters
    ----------
    graph : igraph.Graph
        The input network.
    vertices : list, optional
        List of vertex names to include. If None, a random sample is selected.
    n_vertices : int, optional
        Number of vertices to sample if `vertices` is None. Default is 5000.

    Returns
    -------
    Graph
        The induced subgraph.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    if vertices is not None:
        selected_vertices = vertices
    else:
        # Assume vertices have a 'name' attribute, fallback to indices
        if IGRAPH_DEFS.NAME in graph.vs.attributes():
            vertex_names = graph.vs[IGRAPH_DEFS.NAME]
        else:
            vertex_names = list(range(graph.vcount()))
        selected_vertices = random.sample(
            vertex_names, min(n_vertices, len(vertex_names))
        )

    subgraph = graph.induced_subgraph(selected_vertices)
    return subgraph


def define_graph_universe(
    graph: Graph,
    vertex_names: Optional[Union[List[str], pd.Series]] = None,
    edgelist: Optional[pd.DataFrame] = None,
    observed_only: bool = False,
    edge_filter_logic: str = UNIVERSE_GATES.AND,
    include_self_edges: bool = False,
) -> Graph:
    """
    Create a graph defining the search space for enrichment-style analyses.

    The graph represents all possible edges for the null model.
    By default (no filters), creates a COMPLETE graph on all vertices.

    Parameters
    ----------
    graph : Graph
        Source graph (used for vertex names and directionality)
    vertex_names : list of str or pd.Series, optional
        Vertex names to include in universe (matching graph vertex 'name' attribute).
        If None, includes all vertices.
    edgelist : pd.DataFrame, optional
        Two-column DataFrame with columns 'source' and 'target' containing vertex names.
        Specifies edges to include in universe.
        If None and observed_only=False, creates complete graph.
    observed_only : bool
        If True, extract edgelist from original graph where 'observed' attribute is True.
    edge_filter_logic : str
        How to combine edgelist and observed_only filters:
        - 'and': Keep edges in BOTH edgelists (intersection)
        - 'or': Keep edges in EITHER edgelist (union)
    include_self_edges : bool
        If True, include self-edges (i -> i) in universe.
        Default is False.

    Returns
    -------
    Graph
        Universe graph with same directionality as source.
        Vertex indices match the filtered vertex set.
    """
    # Validate edge filter logic
    if edge_filter_logic not in VALID_UNIVERSE_GATES:
        raise ValueError(
            f"edge_filter_logic must be one of {VALID_UNIVERSE_GATES}, got {edge_filter_logic}"
        )

    # Get directionality from source graph
    is_directed = graph.is_directed()

    # Step 1: Filter vertices
    selected_names = _get_universe_vertex_names(graph, vertex_names)

    # Step 2: Build edge filters
    edge_filters = _get_universe_edge_filters(graph, edgelist, observed_only)

    # Step 3: Create final edgelist
    final_edgelist = _create_universe_edgelist(
        edge_filters, edge_filter_logic, selected_names, is_directed
    )

    # Filter edgelist to only vertices in selected_names
    selected_names_set = set(selected_names)
    final_edgelist = final_edgelist[
        final_edgelist[IGRAPH_DEFS.SOURCE].isin(selected_names_set)
        & final_edgelist[IGRAPH_DEFS.TARGET].isin(selected_names_set)
    ].reset_index(drop=True)

    # Step 4: Create universe graph
    universe = Graph(directed=is_directed)

    # Add vertices with names
    n_vertices = len(selected_names)
    universe.add_vertices(n_vertices)
    universe.vs[IGRAPH_DEFS.NAME] = selected_names

    # Build name to new index mapping
    name_to_new_idx = {name: i for i, name in enumerate(selected_names)}

    # Add edges
    if len(final_edgelist) > 0:
        edge_tuples = [
            (
                name_to_new_idx[row[IGRAPH_DEFS.SOURCE]],
                name_to_new_idx[row[IGRAPH_DEFS.TARGET]],
            )
            for _, row in final_edgelist.iterrows()
        ]
        universe.add_edges(edge_tuples)

    # Simplify: remove duplicate edges and optionally self-loops
    universe.simplify(multiple=True, loops=not include_self_edges)

    return universe


def filter_to_largest_subgraph(graph: Graph) -> Graph:
    """
    Filter an igraph to its largest weakly connected component.

    Parameters
    ----------
    graph : Graph
        The input network.

    Returns
    -------
    Graph
        The largest weakly connected component.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    component_members = graph.components(mode="weak")
    component_sizes = [len(x) for x in component_members]

    top_component_members = [
        m
        for s, m in zip(component_sizes, component_members)
        if s == max(component_sizes)
    ][0]

    largest_subgraph = graph.induced_subgraph(top_component_members)
    return largest_subgraph


def filter_to_largest_subgraphs(graph: Graph, top_k: int) -> list[Graph]:
    """
    Filter an igraph to its largest weakly connected components.

    Parameters
    ----------
    graph : Graph
        The input network.
    top_k : int
        The number of largest components to return.

    Returns
    -------
    list[Graph]
        A list of the top K largest components as graphs.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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


def get_graph_summary(graph: Graph) -> dict[str, Any]:
    """
    Calculate common summary statistics for an igraph network.

    Parameters
    ----------
    graph : Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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


def get_merge_keys(
    graph: Graph, merge_by: str = IGRAPH_DEFS.NAME
) -> tuple[str, str, str]:
    """
    Get the merge keys for a graph based on the merge_by parameter.

    Parameters
    ----------
    graph : Graph
        The graph to get the merge keys for.
    merge_by : str
        The attribute to merge by. Must be one of IGRAPH_DEFS.NAME or IGRAPH_DEFS.INDEX.

    Returns
    -------
    tuple[str, str, str]
        The merge keys.

    Raises
    ------
    ValueError
        If merge_by is not one of IGRAPH_DEFS.NAME or IGRAPH_DEFS.INDEX.
        If the vertex attribute is not unique.
        If the expected attributes are not present in the graph.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    VALID_MERGE_BY = [IGRAPH_DEFS.NAME, IGRAPH_DEFS.INDEX]
    if merge_by not in VALID_MERGE_BY:
        raise ValueError(f"merge_by must be one of {VALID_MERGE_BY}, got {merge_by}")

    vertex_id_attr = (
        IGRAPH_DEFS.NAME if merge_by == IGRAPH_DEFS.NAME else IGRAPH_DEFS.INDEX
    )
    source_id_attr = (
        NAPISTU_GRAPH_EDGES.FROM if merge_by == IGRAPH_DEFS.NAME else IGRAPH_DEFS.SOURCE
    )
    target_id_attr = (
        NAPISTU_GRAPH_EDGES.TO if merge_by == IGRAPH_DEFS.NAME else IGRAPH_DEFS.TARGET
    )

    # validate that these attributes are present and vertices are unique
    if vertex_id_attr not in graph.vs.attributes():
        raise ValueError(f"Vertex attribute '{vertex_id_attr}' not found in graph")
    if source_id_attr not in graph.es.attributes():
        raise ValueError(f"Edge attribute '{source_id_attr}' not found in graph")
    if target_id_attr not in graph.es.attributes():
        raise ValueError(f"Edge attribute '{target_id_attr}' not found in graph")
    if graph.vs[vertex_id_attr].nunique() != graph.vcount():
        raise ValueError(f"Vertex attribute '{vertex_id_attr}' is not unique")

    return (vertex_id_attr, source_id_attr, target_id_attr)


def graph_to_pandas_dfs(graph: Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert an igraph to Pandas DataFrames for vertices and edges.

    Parameters
    ----------
    graph : Graph
        An igraph network.

    Returns
    -------
    vertices : pandas.DataFrame
        A table with one row per vertex.
    edges : pandas.DataFrame
        A table with one row per edge.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    vertices = pd.DataFrame(
        [{**{IGRAPH_DEFS.INDEX: v.index}, **v.attributes()} for v in graph.vs]
    )
    edges = pd.DataFrame(
        [
            {
                **{IGRAPH_DEFS.SOURCE: e.source, IGRAPH_DEFS.TARGET: e.target},
                **e.attributes(),
            }
            for e in graph.es
        ]
    )
    return vertices, edges


def validate_edge_attributes(graph: Graph, edge_attributes: list[str]) -> None:
    """
    Validate that all required edge attributes exist in an igraph.

    Parameters
    ----------
    graph : Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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


def validate_vertex_attributes(graph: Graph, vertex_attributes: list[str]) -> None:
    """
    Validate that all required vertex attributes exist in an igraph.

    Parameters
    ----------
    graph : Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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


def _create_universe_edgelist(
    edge_filters: List[pd.DataFrame],
    edge_filter_logic: str,
    selected_names: List[str],
    is_directed: bool,
) -> pd.DataFrame:
    """
    Create edgelist for universe from filters or complete graph.

    Parameters
    ----------
    edge_filters : List[pd.DataFrame]
        List of edgelist DataFrames to combine
    edge_filter_logic : str
        'and' for intersection, 'or' for union
    selected_names : List[str]
        Vertex names in the universe
    is_directed : bool
        Whether graph is directed

    Returns
    -------
    pd.DataFrame
        Final edgelist with 'source' and 'target' columns
    """
    if len(edge_filters) == 0:
        # Create complete graph
        n_vertices = len(selected_names)
        if is_directed:
            complete_edges = [
                {IGRAPH_DEFS.SOURCE: src, IGRAPH_DEFS.TARGET: tgt}
                for src in selected_names
                for tgt in selected_names
            ]
        else:
            complete_edges = [
                {
                    IGRAPH_DEFS.SOURCE: selected_names[i],
                    IGRAPH_DEFS.TARGET: selected_names[j],
                }
                for i in range(n_vertices)
                for j in range(i, n_vertices)
            ]
        return pd.DataFrame(complete_edges)

    if len(edge_filters) == 1:
        return edge_filters[0].copy()

    # Combine multiple filters
    if edge_filter_logic == UNIVERSE_GATES.AND:
        # Intersection: edges in both
        result = edge_filters[0]
        for ef in edge_filters[1:]:
            result = pd.merge(
                result, ef, on=[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET], how="inner"
            )
        return result
    elif edge_filter_logic == UNIVERSE_GATES.OR:
        # Union: edges in either
        return pd.concat(edge_filters, ignore_index=True).drop_duplicates(
            subset=[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET],
        )
    else:
        raise ValueError(f"Invalid edge_filter_logic: {edge_filter_logic}")


def _ensure_valid_attribute(graph: Graph, attribute: str, non_negative: bool = True):
    """
    Ensure a vertex attribute is present, numeric, finite, and optionally non-negative for all vertices.

    This utility checks that the specified vertex attribute exists, is numeric, and (optionally) non-negative
    for all vertices in the graph. Missing or None values are treated as 0. Raises ValueError
    if the attribute is missing for all vertices, if all values are zero, or if any value is negative (if non_negative=True).

    Parameters
    ----------
    graph : NapistuGraph or Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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


def _get_attribute_masks(
    graph: Graph,
    mask_specs: Dict[str, Union[str, np.ndarray, List, None]],
) -> Dict[str, np.ndarray]:
    """
    Generate boolean masks for each attribute based on specifications.

    Parameters
    ----------
    graph : Graph
        Input graph.
    mask_specs : Dict[str, Union[str, np.ndarray, List, None]]
        Dictionary mapping each attribute to its mask specification.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each attribute to its boolean mask array.
    """

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

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
                    graph.vs[IGRAPH_DEFS.NAME]
                    if IGRAPH_DEFS.NAME in graph.vs.attributes()
                    else None
                )
                if node_names is None:
                    raise ValueError(
                        f"Graph has no '{IGRAPH_DEFS.NAME}' attribute for string mask"
                    )
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
    graph: Graph,
    components,
    component_sizes: Sequence[int],
    n: int = 10,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """
    Summarize the top N components' network properties.

    Parameters
    ----------
    graph : Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    top_components = _get_top_n_objects(component_sizes, components, n, ascending)
    top_component_stats = [
        {"n": len(c), "examples": [graph.vs[n].attributes() for n in c[:10]]}
        for c in top_components
    ]
    return top_component_stats


def _get_top_n_nodes(
    graph: Graph,
    vals: Sequence,
    val_name: str,
    n: int = 10,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """
    Get the top N nodes by a node attribute.

    Parameters
    ----------
    graph : Graph
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

    if not isinstance(graph, Graph):
        raise ValueError(f"graph must be an igraph.Graph object but was {type(graph)}")

    top_idxs = _get_top_n_idx(vals, n, ascending=ascending)
    top_node_attrs = [graph.vs[idx].attributes() for idx in top_idxs]
    top_vals = [vals[idx] for idx in top_idxs]
    return [{val_name: val, **node} for val, node in zip(top_vals, top_node_attrs)]


def _get_universe_edge_filters(
    graph: Graph,
    edgelist: Optional[pd.DataFrame] = None,
    observed_only: bool = False,
) -> List[pd.DataFrame]:
    """
    Build list of edge filters from user inputs.

    Parameters
    ----------
    graph : Graph
        Source graph
    edgelist : pd.DataFrame, optional
        User-provided edgelist with 'source' and 'target' columns
    observed_only : bool
        If True, extract observed edges from graph

    Returns
    -------
    List[pd.DataFrame]
        List of edgelist DataFrames to filter by
    """
    edge_filters = []

    # Extract observed edges if requested (all edges in the original graph)
    if observed_only:
        observed_edges = pd.DataFrame(
            [
                {
                    IGRAPH_DEFS.SOURCE: graph.vs[e.source][IGRAPH_DEFS.NAME],
                    IGRAPH_DEFS.TARGET: graph.vs[e.target][IGRAPH_DEFS.NAME],
                }
                for e in graph.es
            ]
        )
        edge_filters.append(observed_edges)

    # Add user-provided edgelist
    if edgelist is not None:
        if (
            IGRAPH_DEFS.SOURCE not in edgelist.columns
            or IGRAPH_DEFS.TARGET not in edgelist.columns
        ):
            raise ValueError(
                f"edgelist must have columns '{IGRAPH_DEFS.SOURCE}' and '{IGRAPH_DEFS.TARGET}'"
            )

        edge_endpoints = set(
            edgelist[IGRAPH_DEFS.SOURCE].tolist()
            + edgelist[IGRAPH_DEFS.TARGET].tolist()
        )
        valid_vertex_names = set(graph.vs[IGRAPH_DEFS.NAME])
        invalid_endpoints = edge_endpoints - valid_vertex_names
        if invalid_endpoints:
            example_invalid_endpoints = list(invalid_endpoints)[
                : min(5, len(invalid_endpoints))
            ]
            raise ValueError(
                f"{len(invalid_endpoints)} edge endpoint(s) not found in graph: {example_invalid_endpoints}"
            )
        edge_filters.append(edgelist[[IGRAPH_DEFS.SOURCE, IGRAPH_DEFS.TARGET]])

    return edge_filters


def _get_universe_degrees(
    universe: Graph,
    directed: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract degree information from universe graph.

    Always returns (out_degrees, in_degrees).
    For undirected graphs, out_degrees == in_degrees == total_degree.

    Parameters
    ----------
    universe : igraph.Graph
        Universe graph
    directed : bool
        Whether to compute directed degrees. If True, computes separate out and in degrees.
        If False, returns total degree for both out and in.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (out_degrees, in_degrees) where each is shape (n_vertices,)
        For undirected graphs, both arrays are identical.

    Notes
    -----
    For undirected graphs, out_degree == in_degree == total_degree.
    This allows calling code to avoid branching logic based on directedness.

    Examples
    --------
    >>> # Works the same for both directed and undirected
    >>> out_deg, in_deg = _get_universe_degrees(universe, directed=False)
    >>> out_deg, in_deg = _get_universe_degrees(universe, directed=True)
    """
    if directed:
        out_degrees = np.array(universe.degree(mode="out"))
        in_degrees = np.array(universe.degree(mode="in"))
    else:
        degrees = np.array(universe.degree())
        out_degrees = degrees
        in_degrees = degrees

    return out_degrees, in_degrees


def _get_universe_vertex_names(
    graph: Graph,
    vertex_names: Optional[Union[List[str], pd.Series]] = None,
) -> List[str]:
    """
    Get and validate vertex names for the universe.

    Parameters
    ----------
    graph : Graph
        Source graph
    vertex_names : list of str or pd.Series, optional
        Vertex names to include. If None, includes all vertices.

    Returns
    -------
    List[str]
        List of vertex names to include in universe

    Raises
    ------
    ValueError
        If any vertex names are not found in the graph
    """
    if vertex_names is not None:
        if isinstance(vertex_names, pd.Series):
            vertex_names = vertex_names.tolist()

        # Require at least one vertex if provided
        if len(vertex_names) == 0:
            raise ValueError("vertex_names must contain at least one vertex name")

        # Build name to index mapping for original graph
        name_to_idx = {v[IGRAPH_DEFS.NAME]: v.index for v in graph.vs}

        # Validate all names exist
        missing = set(vertex_names) - set(name_to_idx.keys())
        if missing:
            example_missing = list(missing)[: min(5, len(missing))]
            raise ValueError(
                f"{len(missing)} vertex name(s) not found in graph: {example_missing}"
            )

        return vertex_names
    else:
        return [v[IGRAPH_DEFS.NAME] for v in graph.vs]


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


def _print_mask_input_result(masks):
    """
    Print a readable summary of the result of _parse_mask_input(mask_input, attributes).
    Shows each attribute and its corresponding mask specification.
    """

    logger.info("Mask input parsing result:")
    for attr, spec in masks.items():
        logger.info(f"  Attribute: {attr!r} -> Mask spec: {repr(spec)}")
