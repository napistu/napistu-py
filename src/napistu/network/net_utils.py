from __future__ import annotations

import logging
import os
import random
import textwrap
import yaml
from typing import Any
from typing import Sequence

import igraph as ig
import numpy as np
import pandas as pd
from napistu import sbml_dfs_core
from napistu import source
from napistu.network import net_create

from napistu.constants import SBML_DFS
from napistu.constants import SOURCE_SPEC

from napistu.identifiers import _validate_assets_sbml_ids
from napistu.network.constants import CPR_GRAPH_NODES
from napistu.network.constants import CPR_GRAPH_TYPES

logger = logging.getLogger(__name__)


def compartmentalize_species(
    sbml_dfs: sbml_dfs_core.SBML_dfs, species: str | list[str]
) -> pd.DataFrame:
    """
    Compartmentalize Species

    Returns the compartmentalized species IDs (sc_ids) corresponding to a list of species (s_ids)

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    species : list
        Species IDs

    Returns
    -------
    pd.DataFrame containings the s_id and sc_id pairs
    """

    if isinstance(species, str):
        species = [species]
    if not isinstance(species, list):
        raise TypeError("species is not a str or list")

    return sbml_dfs.compartmentalized_species[
        sbml_dfs.compartmentalized_species[SBML_DFS.S_ID].isin(species)
    ].reset_index()[[SBML_DFS.S_ID, SBML_DFS.SC_ID]]


def compartmentalize_species_pairs(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    origin_species: str | list[str],
    dest_species: str | list[str],
) -> pd.DataFrame:
    """
    Compartmentalize Shortest Paths

    For a set of origin and destination species pairs, consider each species in every
      compartment it operates in, seperately.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    origin_species : list
        Species IDs as starting points
    dest_species : list
        Species IDs as ending points

    Returns
    -------
    pd.DataFrame containing pairs of origin and destination compartmentalized species
    """

    compartmentalized_origins = compartmentalize_species(
        sbml_dfs, origin_species
    ).rename(columns={SBML_DFS.SC_ID: "sc_id_origin", SBML_DFS.S_ID: "s_id_origin"})
    if isinstance(origin_species, str):
        origin_species = [origin_species]

    compartmentalized_dests = compartmentalize_species(sbml_dfs, dest_species).rename(
        columns={SBML_DFS.SC_ID: "sc_id_dest", SBML_DFS.S_ID: "s_id_dest"}
    )
    if isinstance(dest_species, str):
        dest_species = [dest_species]

    # create an all x all of origins and destinations
    target_species_paths = pd.DataFrame(
        [(x, y) for x in origin_species for y in dest_species]
    )
    target_species_paths.columns = ["s_id_origin", "s_id_dest"]

    target_species_paths = target_species_paths.merge(compartmentalized_origins).merge(
        compartmentalized_dests
    )

    if target_species_paths.shape[0] == 0:
        raise ValueError(
            "No compartmentalized paths exist, this is unexpected behavior"
        )

    return target_species_paths


def get_minimal_sources_edges(
    vertices: pd.DataFrame, sbml_dfs: sbml_dfs_core.SBML_dfs
) -> pd.DataFrame | None:
    """Assign edges to a set of sources."""

    nodes = vertices["node"].tolist()
    present_reactions = sbml_dfs.reactions[sbml_dfs.reactions.index.isin(nodes)]

    if len(present_reactions) == 0:
        return None

    table_schema = sbml_dfs.schema[SBML_DFS.REACTIONS]
    source_df = source.unnest_sources(present_reactions, table_schema["source"])

    if source_df is None:
        return None
    else:
        edge_sources = source.greedy_set_coverge_of_sources(source_df, table_schema)
        return edge_sources.reset_index()[
            [SBML_DFS.R_ID, SOURCE_SPEC.PATHWAY_ID, SOURCE_SPEC.NAME]
        ]


def get_graph_summary(graph: ig.Graph) -> dict[str, Any]:
    """Calculates common summary statistics for a network

    Args:
        graph (ig.Graph): An igraph

    returns:
        dict: A dictionary of summary statistics with values
            n_edges [int]: number of edges
            n_vertices [int]: number of vertices
            n_components [int]: number of weakly connected components
                (i.e. without considering edge directionality)
            stats_component_sizes [dict[str, float]]: summary statistics for the component sizes
            top10_large_components [list[dict[str, Any]]]: the top 10 largest components with 10 example vertices
            top10_smallest_components [list[dict[str, Any]]]: the top 10 smallest components with 10 example vertices
            average_path_length [float]: the average shortest path length between all vertices
            top10_betweenness [list[dict[str, Any]]]: the top 10 vertices by betweenness centrality.
                Roughly: measures how many shortest paths go through a vertices
            top10_harmonic_centrality [list[dict[str, Any]]]: the top 10 vertices by harmonic centrality:
                Roughly: mean inverse distance to all other vertices
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

    between = list(graph.betweenness(directed=False))
    stats["top10_betweenness"] = _get_top_n_nodes(
        graph, between, "betweenness", n=10, ascending=False
    )

    harmonic_centrality = list(graph.harmonic_centrality())
    stats["top10_harmonic_centrality"] = _get_top_n_nodes(
        graph, harmonic_centrality, "harmonic_centrality", n=10, ascending=False
    )

    return stats


def export_networks(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    model_prefix: str,
    outdir: str,
    directeds: list[bool] = [True, False],
    graph_types: list[str] = [CPR_GRAPH_TYPES.BIPARTITE, CPR_GRAPH_TYPES.REGULATORY],
) -> None:
    """
    Exports Networks

    Create one or more network from a pathway model and pickle the results

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A pathway model
    model_prefix: str
        Label to prepend to all exported files
    outdir: str
        Path to an existing directory where results should be saved
    directeds : [bool]
        List of directed types to export: a directed (True) or undirected graph be made (False)
    graph_types : [str]
        Types of graphs to construct, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - regulatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products

    Returns:
    ----------
    None
    """

    if not isinstance(sbml_dfs, sbml_dfs_core.SBML_dfs):
        raise TypeError(
            f"sbml_dfs must be a sbml_dfs_core.SBML_dfs, but was {type(sbml_dfs)}"
        )
    if not isinstance(model_prefix, str):
        raise TypeError(f"model_prefix was a {type(model_prefix)} and must be a str")
    if not os.path.isdir(outdir):
        raise FileNotFoundError(f"{outdir} does not exist")
    if not isinstance(directeds, list):
        raise TypeError(f"directeds must be a list, but was {type(directeds)}")
    if not isinstance(graph_types, list):
        raise TypeError(f"graph_types must be a list but was a {type(graph_types)}")

    # iterate through provided graph_types and export each type
    for graph_type in graph_types:
        for directed in directeds:
            export_pkl_path = _create_network_save_string(
                model_prefix=model_prefix,
                outdir=outdir,
                directed=directed,
                graph_type=graph_type,
            )
            print(f"Exporting {graph_type} network to {export_pkl_path}")

            network_graph = net_create.process_cpr_graph(
                sbml_dfs=sbml_dfs,
                directed=directed,
                graph_type=graph_type,
                verbose=True,
            )

            network_graph.write_pickle(export_pkl_path)

    return None


def read_network_pkl(
    model_prefix: str,
    network_dir: str,
    graph_type: str,
    directed: bool = True,
) -> ig.Graph:
    """
    Read Network Pickle

    Read a saved network representation.

    Params
    ------
    model_prefix: str
        Type of model to import
    network_dir: str
        Path to a directory containing all saved networks.
    directed : bool
        Should a directed (True) or undirected graph be loaded (False)
    graph_type : [str]
        Type of graphs to read, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - reguatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products

    Returns
    -------
    network_graph: igraph.Graph
        An igraph network of the pathway

    """

    if not isinstance(model_prefix, str):
        raise TypeError(f"model_prefix was a {type(model_prefix)} and must be a str")
    if not os.path.isdir(network_dir):
        raise FileNotFoundError(f"{network_dir} does not exist")
    if not isinstance(directed, bool):
        raise TypeError(f"directed must be a bool, but was {type(directed)}")
    if not isinstance(graph_type, str):
        raise TypeError(f"graph_type must be a str but was a {type(graph_type)}")

    import_pkl_path = _create_network_save_string(
        model_prefix, network_dir, directed, graph_type
    )
    if not os.path.isfile(import_pkl_path):
        raise FileNotFoundError(f"{import_pkl_path} does not exist")
    print(f"Importing {graph_type} network from {import_pkl_path}")

    network_graph = ig.Graph.Read_Pickle(fname=import_pkl_path)

    return network_graph


def filter_to_largest_subgraph(cpr_graph: ig.Graph) -> ig.Graph:
    """Filter a graph to its largest weakly connected component."""

    component_members = cpr_graph.components(mode="weak")
    component_sizes = [len(x) for x in component_members]

    top_component_members = [
        m
        for s, m in zip(component_sizes, component_members)
        if s == max(component_sizes)
    ][0]

    largest_subgraph = cpr_graph.induced_subgraph(top_component_members)

    return largest_subgraph


def validate_assets(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
    precomputed_distances: pd.DataFrame,
    identifiers_df: pd.DataFrame,
) -> None:
    """
    Validate Assets

    Perform a few quick checks of inputs to catch inconsistencies.

    Args:
        sbml_dfs (sbml_dfs_core.SBML_dfs):
            A pathway representation.
        cpr_graph (igraph.Graph):
            A network-based representation of "sbml_dfs".
        precomputed_distances (pd.DataFrame):
            Precomputed distances between vertices in "cpr_graph".
        identifiers_df (pd.DataFrame):
            A table of systematic identifiers for compartmentalized species in "sbml_dfs".

    Returns:
        None


    """

    # compare cpr_graph to sbml_dfs
    # test for consistent sc_id to sc_name mappings
    _validate_assets_sbml_graph(sbml_dfs, cpr_graph)

    # compare precomputed_distances to cpr_graph
    # test whether dircetly connected sc_ids are in the same reaction
    _validate_assets_graph_dist(cpr_graph, precomputed_distances)

    # compare identifiers_df to sbml_dfs
    # do the (sc_id, s_name) tuples in in identifiers match (sc_id, s_name) tuples in sbml_dfs
    _validate_assets_sbml_ids(sbml_dfs, identifiers_df)

    return None


def cpr_graph_to_pandas_dfs(cpr_graph: ig.Graph):
    """
    CPR Graph to Pandas DataFrames

    Take an igraph representation of a network and turn it into vertices and edges tables.

    Args:
        cpr_graph(ig.Graph): an igraph network

    Returns:
        vertices (pd.DataFrame):
            A table with one row per vertex.
        edges (pd.DataFrame):
            A table with one row per edge.
    """

    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in cpr_graph.vs]
    )
    edges = pd.DataFrame(
        [
            {**{"source": e.source, "target": e.target}, **e.attributes()}
            for e in cpr_graph.es
        ]
    )

    return vertices, edges


def safe_fill(x, fill_width=15):
    if x == "":
        return ""
    else:
        return textwrap.fill(x, 15)


def read_graph_attrs_spec(graph_attrs_spec_uri: str) -> dict:
    """Read a YAML file containing the specification for adding reaction- and/or species-attributes to a cpr_graph."""

    with open(graph_attrs_spec_uri) as f:
        graph_attrs_spec = yaml.safe_load(f)

    VALID_SPEC_SECTIONS = ["species", "reactions"]
    defined_spec_sections = set(graph_attrs_spec.keys()).intersection(
        VALID_SPEC_SECTIONS
    )

    if len(defined_spec_sections) == 0:
        raise ValueError(
            f"The provided graph attributes spec did not contain either of the expected sections: {', '.join(VALID_SPEC_SECTIONS)}"
        )

    if "reactions" in defined_spec_sections:
        net_create._validate_entity_attrs(graph_attrs_spec["reactions"])

    if "species" in defined_spec_sections:
        net_create._validate_entity_attrs(graph_attrs_spec["reactions"])

    return graph_attrs_spec


def _create_network_save_string(
    model_prefix: str, outdir: str, directed: bool, graph_type: str
) -> str:
    if directed:
        directed_str = "directed"
    else:
        directed_str = "undirected"

    export_pkl_path = os.path.join(
        outdir, model_prefix + "_network_" + graph_type + "_" + directed_str + ".pkl"
    )

    return export_pkl_path


def _create_induced_subgraph(
    cpr_graph: ig.Graph, vertices=None, n_vertices: int = 5000
) -> ig.Graph:
    """
    Utility function for creating subgraphs including a set of vertices and their connections

    """

    if vertices is not None:
        selected_vertices = vertices
    else:
        vertex_names = cpr_graph.vs[CPR_GRAPH_NODES.NAME]
        selected_vertices = random.sample(vertex_names, n_vertices)

    subgraph = cpr_graph.induced_subgraph(selected_vertices)

    return subgraph


def _validate_assets_sbml_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs, cpr_graph: ig.Graph
) -> None:
    """ "Check an sbml_dfs model and cpr_graph for inconsistencies."""

    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in cpr_graph.vs]
    )

    matched_cspecies = sbml_dfs.compartmentalized_species.reset_index()[
        ["sc_id", "sc_name"]
    ].merge(
        vertices.query("node_type == 'species'"),
        left_on=["sc_id"],
        right_on=["name"],
    )

    mismatched_names = [
        f"{x} != {y}"
        for x, y in zip(matched_cspecies["sc_name"], matched_cspecies["node_name"])
        if x != y
    ]

    if len(mismatched_names) > 0:
        example_names = mismatched_names[: min(10, len(mismatched_names))]

        raise ValueError(
            f"{len(mismatched_names)} species names do not match between sbml_dfs and cpr_graph: {example_names}"
        )

    return None


def _validate_assets_graph_dist(
    cpr_graph: ig.Graph, precomputed_distances: pd.DataFrame
) -> None:
    """ "Check an cpr_graph and precomputed distances table for inconsistencies."""

    edges = pd.DataFrame(
        [{**{"index": e.index}, **e.attributes()} for e in cpr_graph.es]
    )

    direct_interactions = precomputed_distances.query("path_length == 1")

    edges_with_distances = direct_interactions.merge(
        edges[["from", "to", "weights", "upstream_weights"]],
        left_on=["sc_id_origin", "sc_id_dest"],
        right_on=["from", "to"],
    )

    inconsistent_weights = edges_with_distances.query("path_weights != weights")
    if inconsistent_weights.shape[0] > 0:
        logger.warning(
            f"{inconsistent_weights.shape[0]} edges' weights are inconsistent between",
            "edges in the cpr_graph and length 1 paths in precomputed_distances."
            f"This is {inconsistent_weights.shape[0] / edges_with_distances.shape[0]:.2%} of all edges.",
        )

    return None


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
    """Summarize the top N components' network properties."""

    top_components = _get_top_n_objects(component_sizes, components, n, ascending)
    top_component_stats = [
        {"n": len(c), "examples": [graph.vs[n].attributes() for n in c[:10]]}
        for c in top_components
    ]
    return top_component_stats


def _get_top_n_nodes(
    graph: ig.Graph, vals: Sequence, val_name: str, n: int = 10, ascending: bool = False
) -> list[dict[str, Any]]:
    """Get the top N nodes by a node attribute."""

    top_idxs = _get_top_n_idx(vals, n, ascending=ascending)
    top_node_attrs = [graph.vs[idx].attributes() for idx in top_idxs]
    top_vals = [vals[idx] for idx in top_idxs]
    return [{val_name: val, **node} for val, node in zip(top_vals, top_node_attrs)]


def _validate_edge_attributes(graph: ig.Graph, edge_attributes: list[str]) -> None:
    """Check for the existence of one or more edge attributes."""

    if isinstance(edge_attributes, list):
        attrs = edge_attributes
    elif isinstance(edge_attributes, str):
        attrs = [edge_attributes]
    else:
        raise TypeError('"edge_attributes" must be a list or str')

    available_attributes = graph.es[0].attributes().keys()
    missing_attributes = set(attrs).difference(available_attributes)
    n_missing_attrs = len(missing_attributes)

    if n_missing_attrs > 0:
        raise ValueError(
            f"{n_missing_attrs} edge attributes were missing ({', '.join(missing_attributes)}). The available edge attributes are {', '.join(available_attributes)}"
        )

    return None


def _validate_vertex_attributes(graph: ig.Graph, vertex_attributes: list[str]) -> None:
    """Check for the existence of one or more vertex attributes."""

    if isinstance(vertex_attributes, list):
        attrs = vertex_attributes
    elif isinstance(vertex_attributes, str):
        attrs = [vertex_attributes]
    else:
        raise TypeError('"vertex_attributes" must be a list or str')

    available_attributes = graph.vs[0].attributes().keys()
    missing_attributes = set(attrs).difference(available_attributes)
    n_missing_attrs = len(missing_attributes)

    if n_missing_attrs > 0:
        raise ValueError(
            f"{n_missing_attrs} vertex attributes were missing ({', '.join(missing_attributes)}). The available vertex attributes are {', '.join(available_attributes)}"
        )

    return None
