"""
Utilities specific to NapistuGraph objects and the wider Napistu ecosystem.

This module contains utilities that are specific to NapistuGraph subclasses
and require knowledge of the Napistu data model (SBML_dfs objects, etc.).
"""

from __future__ import annotations

import logging
import os
import yaml
from typing import Optional, Union

import igraph as ig
import pandas as pd
from napistu import sbml_dfs_core
from napistu import source
from napistu.network import net_create
from napistu.network.napistu_graph_core import NapistuGraph

from napistu.constants import SBML_DFS
from napistu.constants import SOURCE_SPEC
from napistu.identifiers import _validate_assets_sbml_ids
from napistu.network.constants import GRAPH_WIRING_APPROACHES
from napistu.network.constants import NAPISTU_GRAPH_DIRECTEDNESS

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


def export_networks(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    model_prefix: str,
    outdir: str,
    directeds: list[bool] = [True, False],
    graph_types: list[str] = [
        GRAPH_WIRING_APPROACHES.BIPARTITE,
        GRAPH_WIRING_APPROACHES.REGULATORY,
    ],
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

            network_graph = net_create.process_napistu_graph(
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
) -> NapistuGraph:
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
    network_graph: NapistuGraph
        A NapistuGraph network of the pathway

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


def validate_assets(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: Optional[Union[NapistuGraph, ig.Graph]] = None,
    precomputed_distances: Optional[pd.DataFrame] = None,
    identifiers_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    Validate Assets

    Perform a few quick checks of inputs to catch inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A pathway representation. (Required)
    napistu_graph : NapistuGraph, optional
        A network-based representation of `sbml_dfs`. NapistuGraph is a subclass of igraph.Graph.
    precomputed_distances : pandas.DataFrame, optional
        Precomputed distances between vertices in `napistu_graph`.
    identifiers_df : pandas.DataFrame, optional
        A table of systematic identifiers for compartmentalized species in `sbml_dfs`.

    Returns
    -------
    None

    Warns
    -----
    If only sbml_dfs is provided and no other assets are given, a warning is logged.

    Raises
    ------
    ValueError
        If precomputed_distances is provided but napistu_graph is not.
    """
    if (
        napistu_graph is None
        and precomputed_distances is None
        and identifiers_df is None
    ):
        logger.warning(
            "validate_assets: Only sbml_dfs was provided; nothing to validate."
        )
        return None

    # Validate napistu_graph if provided
    if napistu_graph is not None:
        _validate_assets_sbml_graph(sbml_dfs, napistu_graph)

    # Validate precomputed_distances if provided (requires napistu_graph)
    if precomputed_distances is not None:
        if napistu_graph is None:
            raise ValueError(
                "napistu_graph must be provided if precomputed_distances is provided."
            )
        _validate_assets_graph_dist(napistu_graph, precomputed_distances)

    # Validate identifiers_df if provided
    if identifiers_df is not None:
        _validate_assets_sbml_ids(sbml_dfs, identifiers_df)

    return None


def napistu_graph_to_pandas_dfs(
    napistu_graph: Union[NapistuGraph, ig.Graph],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert a NapistuGraph to Pandas DataFrames for vertices and edges.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A NapistuGraph network (subclass of igraph.Graph).

    Returns
    -------
    vertices : pandas.DataFrame
        A table with one row per vertex.
    edges : pandas.DataFrame
        A table with one row per edge.
    """
    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in napistu_graph.vs]
    )
    edges = pd.DataFrame(
        [
            {**{"source": e.source, "target": e.target}, **e.attributes()}
            for e in napistu_graph.es
        ]
    )
    return vertices, edges


def read_graph_attrs_spec(graph_attrs_spec_uri: str) -> dict:
    """Read a YAML file containing the specification for adding reaction- and/or species-attributes to a napistu_graph."""
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


# Internal utility functions
def _create_network_save_string(
    model_prefix: str, outdir: str, directed: bool, graph_type: str
) -> str:
    if directed:
        directed_str = NAPISTU_GRAPH_DIRECTEDNESS.DIRECTED
    else:
        directed_str = NAPISTU_GRAPH_DIRECTEDNESS.UNDIRECTED

    export_pkl_path = os.path.join(
        outdir, model_prefix + "_network_" + graph_type + "_" + directed_str + ".pkl"
    )

    return export_pkl_path


def _validate_assets_sbml_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs, napistu_graph: Union[NapistuGraph, ig.Graph]
) -> None:
    """
    Check an sbml_dfs model and NapistuGraph for inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The pathway representation.
    napistu_graph : NapistuGraph
        The network representation (subclass of igraph.Graph).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If species names do not match between sbml_dfs and napistu_graph.
    """
    vertices = pd.DataFrame(
        [{**{"index": v.index}, **v.attributes()} for v in napistu_graph.vs]
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
            f"{len(mismatched_names)} species names do not match between sbml_dfs and napistu_graph: {example_names}"
        )
    return None


def _validate_assets_graph_dist(
    napistu_graph: NapistuGraph, precomputed_distances: pd.DataFrame
) -> None:
    """
    Check a NapistuGraph and precomputed distances table for inconsistencies.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The network representation (subclass of igraph.Graph).
    precomputed_distances : pandas.DataFrame
        Precomputed distances between vertices in the network.

    Returns
    -------
    None

    Warns
    -----
    If edge weights are inconsistent between the graph and precomputed distances.
    """
    edges = pd.DataFrame(
        [{**{"index": e.index}, **e.attributes()} for e in napistu_graph.es]
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
            "edges in the napistu_graph and length 1 paths in precomputed_distances."
            f"This is {inconsistent_weights.shape[0] / edges_with_distances.shape[0]:.2%} of all edges.",
        )
    return None
