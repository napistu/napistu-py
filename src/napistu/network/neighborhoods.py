from __future__ import annotations

import logging
import math
import os
import pickle
import shutil
import textwrap
import warnings
from collections import ChainMap
from typing import Any

import igraph as ig
import numpy as np
import pandas as pd
from napistu import sbml_dfs_core
from napistu import utils
from napistu.network import ng_utils
from napistu.network import paths

from napistu.constants import (
    MINI_SBO_NAME_TO_POLARITY,
    MINI_SBO_TO_NAME,
    NAPISTU_EDGELIST,
    ONTOLOGIES,
    SBML_DFS,
)

from napistu.network.constants import (
    DISTANCES,
    GRAPH_RELATIONSHIPS,
    GRAPH_WIRING_APPROACHES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NEIGHBORHOOD_DICT_KEYS,
    NEIGHBORHOOD_NETWORK_TYPES,
    NET_POLARITY,
    VALID_NEIGHBORHOOD_NETWORK_TYPES,
)

logger = logging.getLogger(__name__)


def find_and_prune_neighborhoods(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    compartmentalized_species: str | list[str],
    precomputed_distances: pd.DataFrame | None = None,
    min_pw_size: int = 3,
    source_total_counts: pd.Series | pd.DataFrame | None = None,
    network_type: str = NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    order: int = 3,
    verbose: bool = True,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Find and Prune Neighborhoods

    Wrapper which combines find_neighborhoods() and prune_neighborhoods()

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic molecular model
    napistu_graph : igraph.Graph
        A bipartite network connecting molecular species and reactions
    compartmentalized_species : [str] or str
        Compartmentalized species IDs for neighborhood centers
    precomputed_distances : pd.DataFrame or None
        If provided, an edgelist of origin->destination path weights and lengths
    min_pw_size: int
        the minimum size of a pathway to be considered
    source_total_counts: pd.Series | pd.DataFrame | None
        Optional, A series of the total counts of each source or a pd.DataFrame with two columns:
        pathway_id and total_counts. As produced by sbml_dfs.get_source_total_counts().
        If None, pathways will be selected by size rather than statistical enrichment.
    network_type: str
        If the network is directed should neighbors be located "downstream",
        or "upstream" of each compartmentalized species. The "hourglass" option
        locates both upstream and downstream species.
    order: int
        Max steps away from center node
    verbose: bool
        Extra reporting
    top_n: int
        How many neighboring molecular species should be retained?
        If the neighborhood includes both upstream and downstream connections
        (i.e., hourglass), this filter will be applied to both sets separately.

    Returns:
    ----------
    A dict containing the neighborhood of each compartmentalized species.
    Each entry in the dict is a dict of the subgraph, vertices, and edges.
    """

    if not isinstance(network_type, str):
        raise TypeError(f"network_type was a {type(network_type)} and must be an str")

    if not isinstance(order, int):
        raise TypeError(f"order was a {type(order)} and must be an int")

    if not isinstance(top_n, int):
        raise TypeError(f"top_n was a {type(top_n)} and must be an int")

    if isinstance(compartmentalized_species, str):
        compartmentalized_species = [compartmentalized_species]
    if not isinstance(compartmentalized_species, list):
        raise TypeError("compartmentalized_species must be a list")

    invalid_cspecies = [
        x
        for x in compartmentalized_species
        if x not in sbml_dfs.compartmentalized_species.index
    ]
    if len(invalid_cspecies) > 0:
        raise ValueError(
            f"compartmentalized_species contains invalid species: {invalid_cspecies}"
        )

    if isinstance(precomputed_distances, pd.DataFrame):
        logger.info("Pre-computed neighbors based on precomputed_distances")

        precomputed_neighbors = _precompute_neighbors(
            compartmentalized_species,
            precomputed_distances=precomputed_distances,
            sbml_dfs=sbml_dfs,
            network_type=network_type,
            order=order,
            top_n=math.ceil(top_n * 1.1),  # ties when using head()?
        )
    else:
        precomputed_neighbors = None

    neighborhood_dicts = find_neighborhoods(
        sbml_dfs=sbml_dfs,
        napistu_graph=napistu_graph,
        compartmentalized_species=compartmentalized_species,
        network_type=network_type,
        order=order,
        precomputed_neighbors=precomputed_neighbors,
        min_pw_size=min_pw_size,
        source_total_counts=source_total_counts,
        verbose=verbose,
    )

    pruned_neighborhoods = prune_neighborhoods(neighborhood_dicts, top_n=top_n)

    return pruned_neighborhoods


def load_neighborhoods(
    s_ids: list[str],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    output_dir: str,
    network_type: str,
    order: int,
    top_n: int,
    overwrite: bool = False,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load Neighborhoods

    Load existing neighborhoods if they exist
    (and overwrite = False) and otherwise construct
    neighborhoods using the provided settings

    Parameters
    ----------
    s_ids: list(str)
        create a neighborhood around each species
    sbml_dfs: sbml_dfs_core.SBML_dfs
        network model
    napistu_graph: igraph.Graph
        network associated with sbml_dfs
    output_dir: str
        path to existing output directory
    network_type: str
        downstream, upstream or hourglass (i.e., downstream and upstream)
    order: 10
        maximum number of steps from the focal node
    top_n: 30
        target number of upstream and downstream species to retain
    overwrite: bool
        ignore cached files and regenerate neighborhoods
    verbose: bool
        extra reporting

    Returns
    -------
    all_neighborhoods_df: pd.DataFrame
        A table containing all species in each query s_ids neighborhood
    neighborhood_dicts: dict
        Outputs from find_and_prune_neighborhoods for each s_id

    """

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"{output_dir} does not exist")

    neighborhood_prefix = create_neighborhood_prefix(network_type, order, top_n)
    vertices_path = os.path.join(output_dir, f"{neighborhood_prefix}_vertices.tsv")
    networks_path = os.path.join(output_dir, f"{neighborhood_prefix}_networks.pkl")
    neighborhood_paths = [vertices_path, networks_path]

    if all([os.path.isfile(x) for x in neighborhood_paths]) and overwrite is False:
        logger.info(f"loading existing neighborhoods for {neighborhood_prefix}")

        all_neighborhoods_df = pd.read_csv(vertices_path, sep="\t")
        with open(networks_path, "rb") as in_file:
            neighborhood_dicts = pickle.load(in_file)

    else:
        logger.info(f"creating neighborhoods based on {neighborhood_prefix}")

        all_neighborhoods_df, neighborhood_dicts = create_neighborhoods(
            s_ids=s_ids,
            sbml_dfs=sbml_dfs,
            napistu_graph=napistu_graph,
            network_type=network_type,
            order=order,
            top_n=top_n,
            verbose=verbose,
        )

        # save df
        all_neighborhoods_df.to_csv(vertices_path, sep="\t", index=False)

        # pickle neighborhoods
        with open(networks_path, "wb") as fh:
            pickle.dump(neighborhood_dicts, fh)

    return all_neighborhoods_df, neighborhood_dicts


def create_neighborhoods(
    s_ids: list[str],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    network_type: str,
    order: int,
    top_n: int,
    verbose: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """
    Create Neighborhoods

    Create neighborhoods for a set of species and return

    Parameters
    ----------
    s_ids: list(str)
        create a neighborhood around each species
    sbml_dfs: sbml_dfs_core.SBML_dfs
        network model
    napistu_graph: igraph.Graph
        network associated with sbml_dfs
    network_type: str
        downstream, upstream or hourglass (i.e., downstream and upstream)
    order: 10
        maximum number of steps from the focal node
    top_n: 30
        target number of upstream and downstream species to retain
    verbose: bool
        extra reporting

    Returns
    -------
    all_neighborhoods_df: pd.DataFrame
        A table containing all species in each query s_ids neighborhood
    neighborhood_dicts: dict
        Outputs from find_and_prune_neighborhoods for each s_id
    """

    if not isinstance(s_ids, list):
        raise TypeError(f"s_ids was a {type(s_ids)} and must be an list")

    for s_id in s_ids:
        if not isinstance(s_id, str):
            raise TypeError(f"s_id was a {type(s_id)} and must be an str")

    if not isinstance(network_type, str):
        raise TypeError(f"network_type was a {type(network_type)} and must be an str")

    if not isinstance(order, int):
        raise TypeError(f"order was a {type(order)} and must be an int")

    if not isinstance(top_n, int):
        raise TypeError(f"top_n was a {type(top_n)} and must be an int")

    neighborhoods_list = list()
    neighborhood_dicts = dict()
    for s_id in s_ids:
        query_sc_species = ng_utils.compartmentalize_species(sbml_dfs, s_id)

        compartmentalized_species = query_sc_species[SBML_DFS.SC_ID].tolist()

        neighborhood_dicts = find_and_prune_neighborhoods(
            sbml_dfs,
            napistu_graph,
            compartmentalized_species=compartmentalized_species,
            network_type=network_type,
            order=order,
            top_n=top_n,
            verbose=verbose,
        )

        # combine multiple neighborhoods

        neighborhood_entities = pd.concat(
            [
                neighborhood_dicts[sc_id][NEIGHBORHOOD_DICT_KEYS.VERTICES].assign(
                    focal_sc_id=sc_id
                )
                for sc_id in neighborhood_dicts.keys()
            ]
        ).assign(focal_s_id=s_id)

        neighborhood_species = neighborhood_entities.merge(
            sbml_dfs.compartmentalized_species[SBML_DFS.S_ID],
            left_on=NAPISTU_GRAPH_VERTICES.NAME,
            right_index=True,
        )

        neighborhoods_list.append(neighborhood_species)
        neighborhood_dicts[s_id] = neighborhood_dicts

    all_neighborhoods_df = pd.concat(neighborhoods_list).reset_index(drop=True)

    return all_neighborhoods_df, neighborhood_dicts


def create_neighborhood_prefix(network_type: str, order: int, top_n: int) -> str:
    if not isinstance(network_type, str):
        raise TypeError(f"network_type was a {type(network_type)} and must be a str")

    if network_type not in VALID_NEIGHBORHOOD_NETWORK_TYPES:
        raise ValueError(
            f"network_type was {network_type} and must be one of {', '.join(VALID_NEIGHBORHOOD_NETWORK_TYPES)}"
        )
    if not isinstance(order, int):
        raise ValueError("order must be an int")
    if not isinstance(top_n, int):
        raise ValueError("top_n must be an int")

    return f"{network_type[0]}{order}s{top_n}n"


def load_neighborhoods_by_partition(
    selected_partition: int,
    neighborhood_outdir: str,
    cache_dir: str,
    wiring_approach: str = GRAPH_WIRING_APPROACHES.REGULATORY,
) -> None:
    """
    Load Neighborhoods By Partition

    Call load_neighborhoods for a subset of species ids defined by a partition.
    This function is setup to be called in a slurm job.

    Params
    ------
    selected_partition: int
        A partition of sids to search
    neighborhood_outdir: str
        Output directory


    Returns
    -------
    None, used for side-effects

    """

    if not os.path.isdir(neighborhood_outdir):
        raise FileNotFoundError(f"{neighborhood_outdir} does not exist")

    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"{cache_dir} does not exist")

    partition_output = os.path.join(
        neighborhood_outdir, f"partition_{selected_partition}"
    )
    # initialize an empty output
    if os.path.isdir(partition_output):
        logger.warning(f"removing existing directory: {partition_output}")
        shutil.rmtree(partition_output)
    os.makedirs(partition_output)

    # format partition s_ids

    sids_to_partition = pd.read_csv(os.path.join(neighborhood_outdir, "partitions.csv"))
    parition_sids_df = sids_to_partition[
        sids_to_partition["partition"] == selected_partition
    ]

    if parition_sids_df.shape[0] == 0:
        raise ValueError(f"No s_ids associated with partition {selected_partition}")

    parition_sids = parition_sids_df[SBML_DFS.S_ID].tolist()

    # read pathway and network data

    # read model containing Calico curations. this is primarily to support search programs
    # to not use these switch to refined.pkl
    refined_model_pkl_path = os.path.join(cache_dir, "curated.pkl")
    with open(refined_model_pkl_path, "rb") as in_file:
        refined_model = pickle.load(in_file)
    refined_model.validate()

    # load the graph
    napistu_graph = ng_utils.read_network_pkl(
        model_prefix="curated",
        network_dir=cache_dir,
        directed=True,
        wiring_approach=wiring_approach,
    )

    _, _ = load_neighborhoods(
        s_ids=parition_sids,
        sbml_dfs=refined_model,
        napistu_graph=napistu_graph,
        output_dir=partition_output,
        network_type="hourglass",
        order=12,
        top_n=100,
        overwrite=True,
        verbose=True,
    )

    return None


def read_paritioned_neighborhoods(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    partitions_path: str,
    n_partitions: int = 200,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Read Partitioned Neighborhoods

    Import a set of neighborhoods produced by the find_neighborhoods_batch.sh slurm job

    Params
    ------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        network model
    napistu_graph: igraph.Graph
        network associated with sbml_dfs
    partitions_path: str
        Path to a directory containing folders for each partition's results
    n_partitions: int
        Number of partitions that exist

    Returns
    -------
    all_neighborhoods_df: pd.DataFrame
        A table containing all species in each query s_ids neighborhood
    neighborhood_dicts: dict
        Outputs from find_and_prune_neighborhoods for each s_id

    """

    # check for partition directories
    expected_partition_dirs = ["partition_" + str(p) for p in range(0, n_partitions)]
    missing_partition_dirs = set(expected_partition_dirs).difference(
        set(os.listdir(partitions_path))
    )
    if len(missing_partition_dirs) != 0:
        raise FileNotFoundError(
            f"{len(missing_partition_dirs)} neighborhood partition directories were not found:"
            f" {', '.join(missing_partition_dirs)}"
        )

    # check for required files
    expected_files = ["h12s100n_vertices.tsv", "h12s100n_networks.pkl"]
    expected_paths_df = pd.DataFrame(
        [
            {"partition": p, "file": f}
            for p in expected_partition_dirs
            for f in expected_files
        ]
    )
    expected_paths_df["path"] = [
        os.path.join(partitions_path, p, f)
        for p, f in zip(expected_paths_df["partition"], expected_paths_df["file"])
    ]
    expected_paths_df["exists"] = [os.path.isfile(p) for p in expected_paths_df["path"]]
    missing_expected_paths_df = expected_paths_df[~expected_paths_df["exists"]]

    if missing_expected_paths_df.shape[0] > 0:
        styled_df = utils.style_df(
            missing_expected_paths_df.drop(["exists"], axis=1), headers="keys"
        )
        logger.warning(styled_df)

        raise FileNotFoundError(
            f"missing {missing_expected_paths_df.shape[0]} required files"
        )

    neighborhood_paths_list = list()
    path_dict_list = list()

    for p in expected_partition_dirs:
        partition_paths, partition_dict = load_neighborhoods(
            s_ids=["stub"],
            sbml_dfs=sbml_dfs,
            napistu_graph=napistu_graph,
            output_dir=os.path.join(partitions_path, p),
            # these settings define the neighborhood string so they must
            # match the settings at the time of network generation
            network_type="hourglass",
            order=12,
            top_n=100,
            overwrite=False,
            verbose=False,
        )

        neighborhood_paths_list.append(partition_paths)
        path_dict_list.append(partition_dict)

    # combine all partitions' dfs and dicts
    all_neighborhoods_df = pd.concat(neighborhood_paths_list).reset_index(drop=True)
    neighborhood_dicts = dict(ChainMap(*path_dict_list))

    # TO DO - remove s_id duplication (these are present in the vertices table in the partition outputs)
    if not all(all_neighborhoods_df["s_id_x"] == all_neighborhoods_df["s_id_y"]):
        raise ValueError("The patch won't hold")
    all_neighborhoods_df = all_neighborhoods_df.drop(["s_id_y"], axis=1).rename(
        {"s_id_x": "s_id"}, axis=1
    )

    return all_neighborhoods_df, neighborhood_dicts


def find_neighborhoods(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    compartmentalized_species: list[str],
    network_type: str = NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    order: int = 3,
    min_pw_size: int = 3,
    precomputed_neighbors: pd.DataFrame | None = None,
    source_total_counts: pd.Series | pd.DataFrame | None = None,
    verbose: bool = True,
) -> dict:
    """
    Find Neighborhood

    Create a network composed of all species and reactions within N steps of
    each of a set of compartmentalized species.

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic molecular model
    napistu_graph : igraph.Graph
        A network connecting molecular species and reactions
    compartmentalized_species : [str]
        Compartmentalized species IDs for neighborhood centers
    network_type: str
        If the network is directed should neighbors be located "downstream",
        or "upstream" of each compartmentalized species. The "hourglass" option
        locates both upstream and downstream species.
    order: int
        Max steps away from center node
    precomputed_neighbors: pd.DataFrame or None
        If provided, a pre-filtered table of nodes nearby the compartmentalized species
        which will be used to skip on-the-fly neighborhood generation.
    min_pw_size: int
        the minimum size of a pathway to be considered
    source_total_counts: pd.Series | pd.DataFrame | None
        Optional, A series of the total counts of each source or a pd.DataFrame with two columns:
        pathway_id and total_counts. As produced by sbml_dfs.get_source_total_counts().
        If None, pathways will be selected by size rather than statistical enrichment.
    verbose: bool
        Extra reporting

    Returns:
    ----------
    A dict containing the neighborhood of each compartmentalized species.
    Each entry in the dict is a dict of the subgraph, vertices, and edges.
    """

    if not isinstance(network_type, str):
        raise TypeError(f"network_type was a {type(network_type)} and must be a str")

    if network_type not in VALID_NEIGHBORHOOD_NETWORK_TYPES:
        raise ValueError(
            f"network_type must be one of {', '.join(VALID_NEIGHBORHOOD_NETWORK_TYPES)}"
        )

    if not isinstance(order, int):
        raise TypeError(f"order was a {type(order)} and must be an int")

    invalid_cspecies = [
        x
        for x in compartmentalized_species
        if x not in sbml_dfs.compartmentalized_species.index
    ]
    if len(invalid_cspecies) > 0:
        raise ValueError(
            f"compartmentalized_species contains invalid species: {invalid_cspecies}"
        )

    # cspecies missing from napistu_graph
    missing_cspecies = [
        x
        for x in compartmentalized_species
        if x not in napistu_graph.vs[NAPISTU_GRAPH_VERTICES.NAME]
    ]
    if len(missing_cspecies) > 0:
        logger.warning(
            f"{len(missing_cspecies)} compartmentalized_species are present in the `sbml_dfs` but missing from the `napistu_graph`: {missing_cspecies}. This can occur if either these are species which are involved in no reactions or it could point to an incompatibility between the `sbml_dfs` and `napistu_graph` which could be further explored with `ng_utils.validate_assets()`."
        )
        compartmentalized_species = [
            x for x in compartmentalized_species if x not in missing_cspecies
        ]

        if len(compartmentalized_species) == 0:
            raise ValueError(
                "No compartmentalized species remain after removing those missing from the `napistu_graph`."
            )

    # create a table which includes cspecies and reaction nearby each of the
    # focal compartmentalized_speecies
    neighborhood_df = _build_raw_neighborhood_df(
        napistu_graph=napistu_graph,
        compartmentalized_species=compartmentalized_species,
        network_type=network_type,
        order=order,
        precomputed_neighbors=precomputed_neighbors,
    )

    # format the vertices and edges in each compartmentalized species' network
    neighborhood_dict = {
        sc_id: create_neighborhood_dict_entry(
            sc_id,
            neighborhood_df=neighborhood_df,
            sbml_dfs=sbml_dfs,
            napistu_graph=napistu_graph,
            min_pw_size=min_pw_size,
            source_total_counts=source_total_counts,
            verbose=verbose,
        )
        for sc_id in compartmentalized_species
    }

    return neighborhood_dict


def create_neighborhood_dict_entry(
    sc_id: str,
    neighborhood_df: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: ig.Graph,
    min_pw_size: int = 3,
    source_total_counts: pd.Series | pd.DataFrame | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Create Neighborhood Dict Entry

    Generate a summary of a compartmentalized species' neighborhood

    Parameters
    ----------
    sc_id: str
        A compartmentalized species id
    neighborhood_df: pd.DataFrame
        A table of upstream and/or downstream neighbors of all compartmentalized species
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic molecular model
    napistu_graph: igraph.Graph
        A network connecting molecular species and reactions
    min_pw_size: int
        the minimum size of a pathway to be considered
    source_total_counts: pd.Series | pd.DataFrame
        Optional, A series of the total counts of each source or a pd.DataFrame with two columns:
        pathway_id and total_counts. As produced by sbml_dfs.get_source_total_counts()
    verbose: bool
        Extra reporting?

    Returns
    -------
    dict containing:
        graph: igraph.Graph
            subgraph of sc_id's neighborhood,
        vertices: pd.DataFrame
            nodes in the neighborhood
        edges: pd.DataFrame
            edges in the neighborhood
        reaction_sources: pd.DataFrame
            models that reactions were derived from
        neighborhood_path_entities: dict
            upstream and downstream dicts representing entities in paths.
            If the keys are to be included in a neighborhood, the
            values should be as well in order to maintain connection to the
            focal node.
    """

    one_neighborhood_df = neighborhood_df[neighborhood_df[SBML_DFS.SC_ID] == sc_id]

    if verbose:
        _create_neighborhood_dict_entry_logging(sc_id, one_neighborhood_df, sbml_dfs)

    if not one_neighborhood_df[NAPISTU_GRAPH_VERTICES.NAME].eq(sc_id).any():
        raise ValueError(
            f"The focal node sc_id = {sc_id} was not in 'one_neighborhood_df'.\
            By convention it should be part of its neighborhood"
        )

    # create the subgraph formed by filtering to neighborhoods
    neighborhood_graph = napistu_graph.subgraph(
        napistu_graph.vs[one_neighborhood_df["neighbor"]], implementation="auto"
    )

    vertices = pd.DataFrame([v.attributes() for v in neighborhood_graph.vs])
    edges = pd.DataFrame([e.attributes() for e in neighborhood_graph.es])

    # add edge polarity: whether edges are activating, inhibiting or unknown
    if edges.shape[0] > 0:
        edges[NET_POLARITY.LINK_POLARITY] = (
            edges[SBML_DFS.SBO_TERM]
            .map(MINI_SBO_TO_NAME)
            .map(MINI_SBO_NAME_TO_POLARITY)
        )

    try:
        reaction_sources = ng_utils.get_minimal_sources_edges(
            vertices.rename(columns={NAPISTU_GRAPH_VERTICES.NAME: "node"}),
            sbml_dfs,
            min_pw_size=min_pw_size,
            # optional, counts of sources across the whole model
            source_total_counts=source_total_counts,
            verbose=verbose,
        )
    except Exception as e:
        logger.warning(
            f"Could not get reaction sources for {sc_id}; returning None. Error: {e}"
        )
        reaction_sources = None

    # to add weights to the network solve the shortest path problem
    # from the focal node to each neighbor
    # solve this problem separately whether a given neighbor is an
    # ancestor or descendant

    (
        downstream_path_attrs,
        downstream_entity_dict,
        upstream_path_attrs,
        upstream_entity_dict,
    ) = _find_neighbors_paths(
        neighborhood_graph,
        one_neighborhood_df,
        sc_id,
        edges,
    )

    # combine upstream and downstream shortest paths
    # in cases a node is upstream and downstream of the focal node
    # by taking the lowest path weight
    vertex_neighborhood_attrs = (
        pd.concat([downstream_path_attrs, upstream_path_attrs])
        .sort_values(DISTANCES.PATH_WEIGHT)
        .groupby("neighbor")
        .first()
    )
    # label the focal node
    vertex_neighborhood_attrs.loc[sc_id, "node_orientation"] = GRAPH_RELATIONSHIPS.FOCAL

    # if the precomputed distances, graph and/or sbml_dfs are inconsistent
    # then the shortest paths search may just return empty lists
    # throw a clearer error message in this case.
    EXPECTED_VERTEX_ATTRS = {
        DISTANCES.FINAL_FROM,
        DISTANCES.FINAL_TO,
        NET_POLARITY.NET_POLARITY,
    }
    missing_vertex_attrs = EXPECTED_VERTEX_ATTRS.difference(
        set(vertex_neighborhood_attrs.columns.tolist())
    )

    if len(missing_vertex_attrs) > 0:
        raise ValueError(
            f"vertex_neighborhood_attrs did not contain the expected columns: {EXPECTED_VERTEX_ATTRS}."
            "This is likely because of inconsistencies between the precomputed distances, graph and/or sbml_dfs."
            "Please try ng_utils.validate_assets() to check for consistency."
        )

    # add net_polarity to edges in addition to nodes
    edges = edges.merge(
        vertex_neighborhood_attrs.reset_index()[
            [DISTANCES.FINAL_FROM, DISTANCES.FINAL_TO, NET_POLARITY.NET_POLARITY]
        ].dropna(),
        left_on=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO],
        right_on=[DISTANCES.FINAL_FROM, DISTANCES.FINAL_TO],
        how="left",
    )

    vertices = vertices.merge(
        vertex_neighborhood_attrs, left_on=NAPISTU_GRAPH_VERTICES.NAME, right_index=True
    )

    # drop nodes with a path length / weight of zero
    # which are NOT the focal node
    # these were cases where no path to/from the focal node to the query node was found
    disconnected_neighbors = vertices.query(
        f"(not node_orientation == '{GRAPH_RELATIONSHIPS.FOCAL}') and {DISTANCES.PATH_WEIGHT} == 0"
    )
    vertices = vertices[~vertices.index.isin(disconnected_neighbors.index.tolist())]

    # add reference urls
    vertices = add_vertices_uri_urls(vertices, sbml_dfs)

    neighborhood_path_entities = {
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM: downstream_entity_dict,
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM: upstream_entity_dict,
    }

    # update graph with additional vertex and edge attributes
    updated_napistu_graph = ig.Graph.DictList(
        vertices=vertices.to_dict("records"),
        edges=edges.to_dict("records"),
        directed=napistu_graph.is_directed(),
        vertex_name_attr=NAPISTU_GRAPH_VERTICES.NAME,
        edge_foreign_keys=(NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO),
    )

    outdict = {
        NEIGHBORHOOD_DICT_KEYS.GRAPH: updated_napistu_graph,
        NEIGHBORHOOD_DICT_KEYS.VERTICES: vertices,
        NEIGHBORHOOD_DICT_KEYS.EDGES: edges,
        NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES: reaction_sources,
        NEIGHBORHOOD_DICT_KEYS.NEIGHBORHOOD_PATH_ENTITIES: neighborhood_path_entities,
    }

    return outdict


def _create_neighborhood_dict_entry_logging(
    sc_id: str, one_neighborhood_df: pd.DataFrame, sbml_dfs: sbml_dfs_core.SBML_dfs
):
    df_summary = one_neighborhood_df.copy()
    df_summary[NAPISTU_GRAPH_VERTICES.NODE_TYPE] = [
        NAPISTU_GRAPH_NODE_TYPES.SPECIES if x else NAPISTU_GRAPH_NODE_TYPES.REACTION
        for x in df_summary[NAPISTU_GRAPH_VERTICES.NAME].isin(
            sbml_dfs.compartmentalized_species.index
        )
    ]
    relationship_counts = df_summary.value_counts(
        ["relationship", "node_type"]
    ).sort_index()

    relation_strings = list()
    for relation in relationship_counts.index.get_level_values(0).unique():
        relation_str = " and ".join(
            [
                f"{relationship_counts[relation][i]} {i}"
                for i in relationship_counts[relation].index
            ]
        )
        relation_strings.append(f"{relation}: {relation_str}")

    msg = f"{sc_id} neighborhood: {'; '.join(relation_strings)}"
    logger.info(msg)


def add_vertices_uri_urls(
    vertices: pd.DataFrame, sbml_dfs: sbml_dfs_core.SBML_dfs
) -> pd.DataFrame:
    """
    Add URI URLs to neighborhood vertices DataFrame.

    This function enriches a vertices DataFrame with URI URLs for both species and
    reactions. For species, it adds standard reference identifiers and Pharos IDs
    where available. For reactions, it adds reaction-specific URI URLs.

    Parameters
    ----------
    vertices: pd.DataFrame
        DataFrame containing neighborhood vertices with the following required columns:
        - NAPISTU_GRAPH_VERTICES.NAME: The name/identifier of each vertex
        - NAPISTU_GRAPH_VERTICES.NODE_TYPE: The type of node, either
        NAPISTU_GRAPH_NODE_TYPES.SPECIES or NAPISTU_GRAPH_NODE_TYPES.REACTION
    sbml_dfs: sbml_dfs_core.SBML_dfs
        Pathway model including species, compartmentalized species, reactions and ontologies

    Returns
    -------
    pd.DataFrame
        Input vertices DataFrame enriched with URI URL columns:
        - For species: standard reference identifier URLs and Pharos IDs
        - For reactions: reaction-specific URI URLs
        - Empty strings for missing URLs

    Raises
    ------
    ValueError
        If vertices DataFrame is empty (no rows)
    TypeError
        If the output is not a pandas DataFrame
    ValueError
        If the output row count doesn't match the input row count

    Notes
    -----
    - Species vertices are merged with compartmentalized_species to get s_id mappings
    - Reaction vertices are processed directly using their names
    - Missing URLs are filled with empty strings
    - The function preserves the original row order and count
    """

    if vertices.shape[0] <= 0:
        raise ValueError("vertices must have at least one row")

    # add uri urls for each node

    # add s_ids
    neighborhood_species = vertices[
        vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE] == NAPISTU_GRAPH_NODE_TYPES.SPECIES
    ].merge(
        sbml_dfs.compartmentalized_species[SBML_DFS.S_ID],
        left_on=NAPISTU_GRAPH_VERTICES.NAME,
        right_index=True,
        how="left",
    )

    # add a standard reference identifier
    neighborhood_species_aug = neighborhood_species.merge(
        sbml_dfs.get_uri_urls(
            NAPISTU_GRAPH_NODE_TYPES.SPECIES, neighborhood_species[SBML_DFS.S_ID]
        ),
        left_on=SBML_DFS.S_ID,
        right_index=True,
        how="left",
        # add pharos ids where available
    ).merge(
        sbml_dfs.get_uri_urls(
            NAPISTU_GRAPH_NODE_TYPES.SPECIES,
            neighborhood_species[SBML_DFS.S_ID],
            required_ontology=ONTOLOGIES.PHAROS,
        ).rename(ONTOLOGIES.PHAROS),
        left_on=SBML_DFS.S_ID,
        right_index=True,
        how="left",
    )

    if (
        sum(
            vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
            == NAPISTU_GRAPH_NODE_TYPES.REACTION
        )
        > 0
    ):
        neighborhood_reactions = vertices[
            vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
            == NAPISTU_GRAPH_NODE_TYPES.REACTION
        ].merge(
            sbml_dfs.get_uri_urls(
                SBML_DFS.REACTIONS,
                vertices[
                    vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
                    == NAPISTU_GRAPH_NODE_TYPES.REACTION
                ][NAPISTU_GRAPH_VERTICES.NAME],
            ),
            left_on=NAPISTU_GRAPH_VERTICES.NAME,
            right_index=True,
            how="left",
        )
    else:
        neighborhood_reactions = None

    if neighborhood_reactions is None:
        updated_vertices = neighborhood_species_aug.fillna("")
    else:
        updated_vertices = pd.concat(
            [neighborhood_species_aug, neighborhood_reactions]
        ).fillna("")

    if not isinstance(updated_vertices, pd.DataFrame):
        raise TypeError("updated_vertices must be a pandas DataFrame")
    if vertices.shape[0] != updated_vertices.shape[0]:
        raise ValueError("output vertices rows did not match input")

    return updated_vertices


def prune_neighborhoods(neighborhoods: dict, top_n: int = 100) -> dict:
    """
    Prune Neighborhoods

    Take a possibly very large neighborhood around a set of focal nodes
    and prune to the most highly weighted nodes. Nodes weights are
    constructed as the sum of path weights from the focal node to each
    neighbor so each pruned neighborhood will still be a single subnetwork.

    Parameters
    ----------
    neighborhoods: dict
        A dictionary of sc_id neighborhoods as produced by find_neighborhoods()
    top_n: int
        How many neighbors should be retained? If the neighborhood includes
        both upstream and downstream connections (i.e., hourglass), this filter
        will be applied to both sets separately

    Returns
    -------
    neighborhoods: dict
        Same structure as neighborhoods input
    """

    if not isinstance(top_n, int):
        raise TypeError(f"top_n was a {type(top_n)} and must be an int")

    pruned_neighborhood_dicts = dict()

    for an_sc_id in neighborhoods.keys():
        one_neighborhood = neighborhoods[an_sc_id]

        # filter to the desired number of vertices w/ lowest path_weight (from focal node)
        # filter neighborhood to high-weight vertices
        pruned_vertices = _prune_vertex_set(one_neighborhood, top_n=top_n)

        # reduce neighborhood to this set of high-weight vertices
        all_neighbors = pd.DataFrame(
            {
                NAPISTU_GRAPH_VERTICES.NAME: one_neighborhood[
                    NEIGHBORHOOD_DICT_KEYS.GRAPH
                ].vs[NAPISTU_GRAPH_VERTICES.NAME]
            }
        )
        pruned_vertices_indices = all_neighbors[
            all_neighbors[NAPISTU_GRAPH_VERTICES.NAME].isin(
                pruned_vertices[NAPISTU_GRAPH_VERTICES.NAME]
            )
        ].index.tolist()

        pruned_neighborhood = one_neighborhood[NEIGHBORHOOD_DICT_KEYS.GRAPH].subgraph(
            one_neighborhood[NEIGHBORHOOD_DICT_KEYS.GRAPH].vs[pruned_vertices_indices],
            implementation="auto",
        )

        pruned_edges = pd.DataFrame([e.attributes() for e in pruned_neighborhood.es])

        pruned_reactions = pruned_vertices[
            pruned_vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
            == NAPISTU_GRAPH_NODE_TYPES.REACTION
        ][NAPISTU_GRAPH_VERTICES.NAME]

        if pruned_reactions.shape[0] != 0:
            if one_neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES] is None:
                # allow for missing source information since this is currently optional
                pruned_reaction_sources = one_neighborhood[
                    NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES
                ]
            else:
                pruned_reaction_sources = one_neighborhood[
                    NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES
                ][
                    one_neighborhood[NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES][
                        SBML_DFS.R_ID
                    ].isin(pruned_reactions)
                ]
        else:
            pruned_reaction_sources = one_neighborhood[
                NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES
            ]

        pruned_neighborhood_dicts[an_sc_id] = {
            NEIGHBORHOOD_DICT_KEYS.GRAPH: pruned_neighborhood,
            NEIGHBORHOOD_DICT_KEYS.VERTICES: pruned_vertices,
            NEIGHBORHOOD_DICT_KEYS.EDGES: pruned_edges,
            NEIGHBORHOOD_DICT_KEYS.REACTION_SOURCES: pruned_reaction_sources,
        }

    return pruned_neighborhood_dicts


def plot_neighborhood(
    neighborhood_graph: ig.Graph,
    name_nodes: bool = False,
    plot_size: int = 1000,
    network_layout: str = "drl",
) -> ig.plot:
    """
    Plot Neighborhood

    Parameters:
    ----------
    neighborhood_graph: igraph.Graph
        An igraph network
    name_nodes: bool
        Should nodes be named
    plot_size: int
        Plot width/height in pixels
    network_layout: str
        Igraph network layout method

    Returns:
    ----------
    An igraph plot
    """

    neighborhood_graph_layout = neighborhood_graph.layout(network_layout)

    if "net_polarity" not in neighborhood_graph.es.attributes():
        logger.warning(
            "net_polarity was not defined as an edge attribute so edges will not be colored"
        )
        neighborhood_graph.es.set_attribute_values("net_polarity", np.nan)

    color_dict = {
        "focal disease": "lime",
        "disease": "aquamarine",
        "focal": "lightcoral",
        NAPISTU_GRAPH_NODE_TYPES.SPECIES: "firebrick",
        NAPISTU_GRAPH_NODE_TYPES.REACTION: "dodgerblue",
    }

    edge_polarity_colors = {
        NET_POLARITY.AMBIGUOUS: "dimgray",
        NET_POLARITY.ACTIVATION: "gold",
        NET_POLARITY.INHIBITION: "royalblue",
        NET_POLARITY.AMBIGUOUS_ACTIVATION: "palegoldenrod",
        NET_POLARITY.AMBIGUOUS_INHIBITION: "powerblue",
        np.nan: "dimgray",
    }

    visual_style = {}  # type: dict[str,Any]
    visual_style["background"] = "black"
    visual_style["vertex_size"] = 10
    if name_nodes:
        visual_style["vertex_label"] = [
            textwrap.fill(x, 15)
            for x in neighborhood_graph.vs[NAPISTU_GRAPH_VERTICES.NODE_NAME]
        ]
    visual_style["vertex_label_color"] = "white"
    visual_style["vertex_label_size"] = 8
    visual_style["vertex_label_angle"] = 90
    visual_style["vertex_label_dist"] = 3
    visual_style["vertex_color"] = [
        color_dict[x] for x in neighborhood_graph.vs[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
    ]
    visual_style["edge_color"] = [
        edge_polarity_colors[x]
        for x in neighborhood_graph.es[NET_POLARITY.NET_POLARITY]
    ]
    visual_style["layout"] = neighborhood_graph_layout
    visual_style["bbox"] = (plot_size, plot_size)
    visual_style["margin"] = 50
    visual_style["title"] = "foobar"

    return ig.plot(neighborhood_graph, **visual_style)


def _precompute_neighbors(
    compartmentalized_species: list[str],
    precomputed_distances: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    network_type: str = NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
    order: int = 3,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Precompute Neighbors

    Identify compartmentalized_species' most tightly connected neighbors using parameters
    shared by the on-the-fly methods (order for identifying neighbors within N steps;
    top_n for identifying the most the lowest weight network paths between the focal node
    and each possible neighbors). This precomputation will greatly speed up the neighborhood
    generation for highly connected species or densely connected networks. In those situations
    naively creating a neighborhood in N steps could contain thousands of neighbors.

    """

    # check that compartmentalized_species are included in precomputed_distances
    all_cspecies = {
        *precomputed_distances[NAPISTU_EDGELIST.SC_ID_ORIGIN].tolist(),
        *precomputed_distances[NAPISTU_EDGELIST.SC_ID_DEST].tolist(),
    }
    missing_cspecies = set(compartmentalized_species).difference(all_cspecies)
    if len(missing_cspecies) > 0:
        logged_specs = ", ".join(list(missing_cspecies)[0:10])
        logger.warning(
            f"{len(missing_cspecies)} cspecies were missing from precomputed_distances including {logged_specs}"
        )

    # filter precomputed_distances to those which originate or end with one of the compartmentalized_species
    # if we are looking for downstream species then we want relationships where a cspecies is the origin
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        valid_origin = precomputed_distances[NAPISTU_EDGELIST.SC_ID_ORIGIN].isin(
            compartmentalized_species
        )
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        valid_dest = precomputed_distances[NAPISTU_EDGELIST.SC_ID_DEST].isin(
            compartmentalized_species
        )

    if network_type == NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS:
        cspecies_subset_precomputed_distances = precomputed_distances[
            [True if (x or y) else False for (x, y) in zip(valid_origin, valid_dest)]
        ]
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM:
        cspecies_subset_precomputed_distances = precomputed_distances.loc[valid_origin]
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM:
        cspecies_subset_precomputed_distances = precomputed_distances.loc[valid_dest]
    else:
        raise ValueError(
            f"network_type was {network_type} and must by one of 'hourglass', 'downstream', 'upstream'"
        )

    logger.debug(
        f"Pre-filtered neighbors {cspecies_subset_precomputed_distances.shape[0]}"
    )

    # filter by distance
    close_cspecies_subset_precomputed_distances = cspecies_subset_precomputed_distances[
        cspecies_subset_precomputed_distances[DISTANCES.PATH_LENGTH] <= order
    ]

    # filter to retain top_n
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        top_descendants = (
            close_cspecies_subset_precomputed_distances[
                close_cspecies_subset_precomputed_distances[
                    DISTANCES.SC_ID_ORIGIN
                ].isin(compartmentalized_species)
            ]
            # sort by path_weight so we can retain the lowest weight neighbors
            .sort_values(DISTANCES.PATH_WEIGHT)
            .groupby(NAPISTU_EDGELIST.SC_ID_ORIGIN)
            .head(top_n)
        )

        logger.debug(f"N top_descendants {top_descendants.shape[0]}")

    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        top_ancestors = (
            close_cspecies_subset_precomputed_distances[
                close_cspecies_subset_precomputed_distances[
                    NAPISTU_EDGELIST.SC_ID_DEST
                ].isin(compartmentalized_species)
            ]
            # sort by path_upstream_weight so we can retain the lowest weight neighbors
            # we allow for upstream weights to differ from downstream weights
            # when creating a network in process_napistu_graph.
            #
            # the default network weighting penalizing an edge from a node
            # based on the number of children it has. this captures the idea
            # that if there are many children we might expect that each
            # of them is less likely to transduct an effect.
            # the logic is flipped if we are looking for ancestors where
            # we penalize based on the number of parents of a node when
            # we use it (i.e., the default upstream_weight).
            .sort_values(DISTANCES.PATH_UPSTREAM_WEIGHT)
            .groupby(NAPISTU_EDGELIST.SC_ID_DEST)
            .head(top_n)
        )

        logger.debug(f"N top_ancestors {top_ancestors.shape[0]}")

    # add reactions

    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        downstream_reactions = _find_reactions_by_relationship(
            precomputed_neighbors=top_descendants,
            compartmentalized_species=compartmentalized_species,
            sbml_dfs=sbml_dfs,
            relationship=GRAPH_RELATIONSHIPS.DESCENDANTS,
        )

        if downstream_reactions is not None:
            logger.debug(f"N downstream reactions {downstream_reactions.shape[0]}")

    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        upstream_reactions = _find_reactions_by_relationship(
            precomputed_neighbors=top_ancestors,
            compartmentalized_species=compartmentalized_species,
            sbml_dfs=sbml_dfs,
            relationship=GRAPH_RELATIONSHIPS.ANCESTORS,
        )

        if upstream_reactions is not None:
            logger.debug(f"N upstream reactions {upstream_reactions.shape[0]}")

    # add the self links since sc_id_dest will be used to define
    # an sc_id_origin-specific subgraph
    identity_df = pd.DataFrame(
        {
            NAPISTU_EDGELIST.SC_ID_ORIGIN: compartmentalized_species,
            NAPISTU_EDGELIST.SC_ID_DEST: compartmentalized_species,
        }
    )

    # combine all ancestor-descendent edges into the precomputed_neighbors edgelist
    if network_type == NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS:
        precomputed_neighbors = pd.concat(
            [
                top_ancestors,
                top_descendants,
                upstream_reactions,  # type: ignore
                downstream_reactions,  # type: ignore
                identity_df,
            ]
        )[
            [NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST]
        ].drop_duplicates()
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM:
        precomputed_neighbors = pd.concat([top_descendants, downstream_reactions, identity_df])[  # type: ignore
            [NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST]
        ].drop_duplicates()
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM:
        precomputed_neighbors = pd.concat([top_ancestors, upstream_reactions, identity_df])[  # type: ignore
            [NAPISTU_EDGELIST.SC_ID_ORIGIN, NAPISTU_EDGELIST.SC_ID_DEST]
        ].drop_duplicates()
    else:
        raise ValueError("This error shouldn't happen")

    return precomputed_neighbors


def _build_raw_neighborhood_df(
    napistu_graph: ig.Graph,
    compartmentalized_species: list[str],
    network_type: str,
    order: int,
    precomputed_neighbors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # report if network_type is not the default and will be ignored due to the network
    #   being undirected
    is_directed = napistu_graph.is_directed()
    if not is_directed and network_type != NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM:
        logger.warning(
            "Network is undirected; network_type will be treated as 'downstream'"
        )
        network_type = NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM

    # create the "out-network" of descendant nodes
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        descendants_df = _find_neighbors(
            napistu_graph=napistu_graph,
            compartmentalized_species=compartmentalized_species,
            relationship=GRAPH_RELATIONSHIPS.DESCENDANTS,
            order=order,
            precomputed_neighbors=precomputed_neighbors,
        )

    # create the "in-network" of ancestor nodes
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        ancestors_df = _find_neighbors(
            napistu_graph=napistu_graph,
            compartmentalized_species=compartmentalized_species,
            relationship=GRAPH_RELATIONSHIPS.ANCESTORS,
            order=order,
            precomputed_neighbors=precomputed_neighbors,
        )

    if network_type == NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS:
        # merge descendants and ancestors
        neighborhood_df = pd.concat([ancestors_df, descendants_df])
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM:
        neighborhood_df = descendants_df
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM:
        neighborhood_df = ancestors_df
    else:
        raise NotImplementedError("invalid network_type")

    # add name since this is an easy way to lookup igraph vertices
    neighborhood_df[NAPISTU_GRAPH_VERTICES.NAME] = [
        x[NAPISTU_GRAPH_VERTICES.NAME]
        for x in napistu_graph.vs[neighborhood_df["neighbor"]]
    ]

    return neighborhood_df


def _find_neighbors(
    napistu_graph: ig.Graph,
    compartmentalized_species: list[str],
    relationship: str,
    order: int = 3,
    precomputed_neighbors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Find Neighbors

    Identify the neighbors nearby each of the requested compartmentalized_species

    If 'precomputed_neighbors' are provided, neighbors will be summarized by reformatting
    this table. Otherwise, neighbors will be found on-the-fly using the igraph.neighborhood() method.

    """

    if isinstance(precomputed_neighbors, pd.DataFrame):
        # add graph indices to neighbors
        nodes_to_names = (
            pd.DataFrame(
                {
                    NAPISTU_GRAPH_VERTICES.NAME: napistu_graph.vs[
                        NAPISTU_GRAPH_VERTICES.NAME
                    ]
                }
            )
            .reset_index()
            .rename({"index": "neighbor"}, axis=1)
        )

        if relationship == GRAPH_RELATIONSHIPS.DESCENDANTS:
            bait_id = NAPISTU_EDGELIST.SC_ID_ORIGIN
            target_id = NAPISTU_EDGELIST.SC_ID_DEST
        elif relationship == GRAPH_RELATIONSHIPS.ANCESTORS:
            bait_id = NAPISTU_EDGELIST.SC_ID_DEST
            target_id = NAPISTU_EDGELIST.SC_ID_ORIGIN
        else:
            raise ValueError(
                f"relationship must be 'descendants' or 'ancestors' but was {relationship}"
            )

        neighbors_df = (
            precomputed_neighbors[
                precomputed_neighbors[bait_id].isin(compartmentalized_species)
            ]
            .merge(
                nodes_to_names.rename({NAPISTU_GRAPH_VERTICES.NAME: target_id}, axis=1)
            )
            .rename({bait_id: SBML_DFS.SC_ID}, axis=1)
            .drop([target_id], axis=1)
            .assign(relationship=relationship)
        )
    else:
        if relationship == GRAPH_RELATIONSHIPS.DESCENDANTS:
            mode_type = "out"
        elif relationship == GRAPH_RELATIONSHIPS.ANCESTORS:
            mode_type = "in"
        else:
            raise ValueError(
                f"relationship must be 'descendants' or 'ancestors' but was {relationship}"
            )

        neighbors = napistu_graph.neighborhood(
            # mode = out queries outgoing edges and is ignored if the network is undirected
            vertices=compartmentalized_species,
            order=order,
            mode=mode_type,
        )

        neighbors_df = pd.concat(
            [
                pd.DataFrame({SBML_DFS.SC_ID: c, "neighbor": x}, index=range(0, len(x)))
                for c, x in zip(compartmentalized_species, neighbors)
            ]
        ).assign(relationship=relationship)

    return neighbors_df


def _find_reactions_by_relationship(
    precomputed_neighbors,
    compartmentalized_species: list,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    relationship: str,
) -> pd.DataFrame | None:
    """
    Find Reactions by Relationship

    Based on an ancestor-descendant edgelist of compartmentalized species find all reactions which involve 2+ members

    Since we primarily care about paths between species and reactions are more of a means-to-an-end of
    connecting pairs of species precomputed_distances are generated between just pairs of species
    this also makes the problem feasible since the number of species is upper bounded at <100K but
    the number of reactions is unbounded. Having a bound ensures that we can calculate
    the precomputed_distances efficiently using matrix operations whose memory footprint scales with O(N^2).
    """

    # if there are no neighboring cspecies then there will be no reactions
    if precomputed_neighbors.shape[0] == 0:
        return None

    if relationship == GRAPH_RELATIONSHIPS.DESCENDANTS:
        bait_id = NAPISTU_EDGELIST.SC_ID_ORIGIN
        target_id = NAPISTU_EDGELIST.SC_ID_DEST
    elif relationship == GRAPH_RELATIONSHIPS.ANCESTORS:
        bait_id = NAPISTU_EDGELIST.SC_ID_DEST
        target_id = NAPISTU_EDGELIST.SC_ID_ORIGIN
    else:
        raise ValueError(
            f"relationship must be 'descendants' or 'ancestors' but was {relationship}"
        )

    # index by the bait id to create a series with all relatives of the specified relationship
    indexed_relatives = (
        precomputed_neighbors[
            precomputed_neighbors[bait_id].isin(compartmentalized_species)
        ]
        .set_index(bait_id)
        .sort_index()
    )

    reaction_relatives = list()

    # loop through compartmentalized species in precomputed_neighbors
    for uq in indexed_relatives.index.unique():
        relatives = indexed_relatives.loc[uq, target_id]
        if isinstance(relatives, str):
            relatives = [relatives]
        elif isinstance(relatives, pd.Series):
            relatives = relatives.tolist()
        else:
            raise ValueError("relatives is an unexpected type")

        # add the focal node to the set of relatives
        relatives_cspecies = {*relatives, *[uq]}
        # count the number of relative cspecies including each reaction
        rxn_species_counts = sbml_dfs.reaction_species[
            sbml_dfs.reaction_species[SBML_DFS.SC_ID].isin(relatives_cspecies)
        ].value_counts(SBML_DFS.R_ID)

        # retain reactions involving 2+ cspecies.
        # some of these reactions will be irrelevant and will be excluded when
        # calculating the shortest paths from/to the focal node from each neighbor
        # in prune_neighborhoods()
        neighboring_reactions = rxn_species_counts[
            rxn_species_counts >= 2
        ].index.tolist()

        # create new entries for reaction relatives
        kws = {bait_id: uq}
        new_entries = pd.DataFrame({target_id: neighboring_reactions}).assign(**kws)

        reaction_relatives.append(new_entries)

    reactions_df = pd.concat(reaction_relatives)

    return reactions_df


def _prune_vertex_set(one_neighborhood: dict, top_n: int) -> pd.DataFrame:
    """
    Prune Vertex Set

    Filter a neighborhood to the lowest weight neighbors connected to the focal node.
    During this process upstream and downstream nodes are treated separately.

    Parameters
    ----------
    one_neighborhood: dict
        The neighborhood around a single compartmentalized species - one of the values
        in dict created by find_neighborhoods().
    top_n: int
        How many neighboring molecular species should be retained?
        If the neighborhood includes both upstream and downstream connections
        (i.e., hourglass), this filter will be applied to both sets separately.

    Returns
    -------
    vertices: pd.DataFrame
        the vertices in one_neighborhood with high weight neighbors removed.

    """

    neighborhood_vertices = one_neighborhood[NEIGHBORHOOD_DICT_KEYS.VERTICES]

    indexed_neighborhood_species = neighborhood_vertices[
        neighborhood_vertices[NAPISTU_GRAPH_VERTICES.NODE_TYPE]
        == NAPISTU_GRAPH_NODE_TYPES.SPECIES
    ].set_index("node_orientation")

    pruned_oriented_neighbors = list()
    for a_node_orientation in indexed_neighborhood_species.index.unique().tolist():
        vertex_subset = indexed_neighborhood_species.loc[a_node_orientation]
        if type(vertex_subset) is pd.Series:
            # handle cases where only one entry exists to DF->series coercion occurs
            vertex_subset = vertex_subset.to_frame().T

        sorted_vertex_set = vertex_subset.sort_values(DISTANCES.PATH_WEIGHT)
        weight_cutoff = sorted_vertex_set[DISTANCES.PATH_WEIGHT].iloc[
            min(top_n - 1, sorted_vertex_set.shape[0] - 1)
        ]

        top_neighbors = sorted_vertex_set[
            sorted_vertex_set[DISTANCES.PATH_WEIGHT] <= weight_cutoff
        ][NAPISTU_GRAPH_VERTICES.NAME].tolist()

        # include reactions and other species necessary to reach the top neighbors
        # by pulling in the past solutions to weighted shortest paths problems
        if a_node_orientation in one_neighborhood["neighborhood_path_entities"].keys():
            # path to/from focal node to each species
            neighborhood_path_entities = one_neighborhood["neighborhood_path_entities"][
                a_node_orientation
            ]

            top_neighbors = set().union(
                *[neighborhood_path_entities[p] for p in top_neighbors]
            )

        pruned_oriented_neighbors.append(top_neighbors)

    # combine all neighbors
    pruned_neighbors = set().union(*pruned_oriented_neighbors)
    pruned_vertices = neighborhood_vertices[
        neighborhood_vertices[NAPISTU_GRAPH_VERTICES.NAME].isin(pruned_neighbors)
    ].reset_index(drop=True)

    return pruned_vertices


def _calculate_path_attrs(
    neighborhood_paths: list[list],
    edges: pd.DataFrame,
    vertices: list,
    weight_var: str = NAPISTU_GRAPH_EDGES.WEIGHT,
) -> tuple[pd.DataFrame, dict[Any, set]]:
    """
    Calculate Path Attributes

    Return the vertices and path weights (sum of edge weights) for a list of paths.

    Parameters
    ----------
    neighborhood_paths: list
        List of lists of edge indices
    edges: pd.DataFrame
        Edges with rows correponding to entries in neighborhood_paths inner lists
    vertices: list
        List of vertices correponding to the ordering of neighborhood_paths
    weights_var: str
        variable in edges to use for scoring path weights

    Returns
    -------
    path_attributes_df: pd.DataFrame
        A table containing attributes summarizing the path to each neighbor
    neighborhood_path_entities: dict
        Dict mapping from each neighbor to the entities connecting it to the focal node

    """

    if not isinstance(neighborhood_paths, list):
        raise TypeError("neighborhood_paths should be a list of lists of edge indices")
    if not isinstance(vertices, list):
        raise TypeError("vertices should be a list of list of vertices")
    if len(vertices) <= 0:
        raise ValueError("vertices must have length greater than zero")
    if len(neighborhood_paths) != len(vertices):
        raise ValueError("vertices and neighborhood_paths were not the same length")

    if any([len(x) > 0 for x in neighborhood_paths]):
        all_path_edges = (
            # create a table of edges traversed to reach each neighbor
            pd.concat(
                [
                    edges.iloc[neighborhood_paths[i]].assign(neighbor=vertices[i])
                    for i in range(0, len(neighborhood_paths))
                ]
            ).groupby("neighbor")
        )

        # if all_path_edges.ngroups > 0:
        path_attributes_df = pd.concat(
            [
                all_path_edges[weight_var].agg("sum").rename(DISTANCES.PATH_WEIGHT),
                all_path_edges.agg("size").rename(DISTANCES.PATH_LENGTH),
                all_path_edges[NET_POLARITY.LINK_POLARITY]
                .agg(paths._terminal_net_polarity)
                .rename(NET_POLARITY.NET_POLARITY),
                # add the final edge since this can be used to add path attributes to edges
                # i.e., apply net_polarity to an edge
                all_path_edges["from"].agg("last").rename(DISTANCES.FINAL_FROM),
                all_path_edges["to"].agg("last").rename(DISTANCES.FINAL_TO),
            ],
            axis=1,
        ).reset_index()

        # create a dict mapping from a neighbor to all mediating nodes
        neighborhood_path_entities = {
            group_name: set().union(*[dat["from"], dat["to"]])
            for group_name, dat in all_path_edges
        }

    else:
        # catch case where there are no paths
        path_attributes_df = pd.DataFrame()
        neighborhood_path_entities = dict()

    # add entries with no edges
    edgeless_nodes = [
        vertices[i]
        for i in range(0, len(neighborhood_paths))
        if len(neighborhood_paths[i]) == 0
    ]
    edgeles_nodes_df = pd.DataFrame({"neighbor": edgeless_nodes}).assign(
        **{
            DISTANCES.PATH_LENGTH: 0,
            DISTANCES.PATH_WEIGHT: 0,
            NET_POLARITY.NET_POLARITY: None,
        }
    )

    # add edgeless entries as entries in the two outputs
    path_attributes_df = pd.concat([path_attributes_df, edgeles_nodes_df])
    neighborhood_path_entities.update({x: {x} for x in edgeless_nodes})

    if path_attributes_df.shape[0] != len(neighborhood_paths):
        raise ValueError(
            "path_attributes_df row count must match number of neighborhood_paths"
        )
    if len(neighborhood_path_entities) != len(neighborhood_paths):
        raise ValueError(
            "neighborhood_path_entities length must match number of neighborhood_paths"
        )

    return path_attributes_df, neighborhood_path_entities


def _find_neighbors_paths(
    neighborhood_graph: ig.Graph,
    one_neighborhood_df: pd.DataFrame,
    sc_id: str,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[Any, set], pd.DataFrame, dict[Any, set]]:
    """
    Find shortest paths between the focal node and its neighbors in both directions.

    This function calculates shortest paths from the focal node to its descendants
    (downstream) and ancestors (upstream) using igraph's shortest path algorithms.
    It uses _calculate_path_attrs to compute path attributes including path weights,
    lengths, and polarity information.

    Parameters
    ----------
    neighborhood_graph: ig.Graph
        The igraph Graph object representing the neighborhood network
    one_neighborhood_df: pd.DataFrame
        DataFrame containing neighborhood information with 'relationship' column
        indicating 'descendants' or 'ancestors' for each node
    sc_id: str
        The compartmentalized species ID of the focal node
    edges: pd.DataFrame
        DataFrame containing edge information with columns for 'from', 'to',
        weights, and link polarity

    Returns
    -------
    downstream_path_attrs: pd.DataFrame
        DataFrame containing path attributes for downstream paths from focal node
        to descendants. Includes columns: neighbor, path_weight, path_length,
        net_polarity, final_from, final_to, node_orientation
    downstream_entity_dict: dict[Any, set]
        Dictionary mapping each descendant neighbor to the set of entities
        (nodes) connecting it to the focal node
    upstream_path_attrs: pd.DataFrame
        DataFrame containing path attributes for upstream paths from focal node
        to ancestors. Includes columns: neighbor, path_weight, path_length,
        net_polarity, final_from, final_to, node_orientation
    upstream_entity_dict: dict[Any, set]
        Dictionary mapping each ancestor neighbor to the set of entities
        (nodes) connecting it to the focal node
    """

    one_descendants_df = one_neighborhood_df[
        one_neighborhood_df["relationship"] == GRAPH_RELATIONSHIPS.DESCENDANTS
    ]
    descendants_list = list(
        set(one_descendants_df[NAPISTU_GRAPH_VERTICES.NAME].tolist()).union({sc_id})
    )

    # hide warnings which are mostly just Dijkstra complaining about not finding neighbors
    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        neighborhood_paths = neighborhood_graph.get_shortest_paths(
            # focal node
            v=sc_id,
            to=descendants_list,
            weights=NAPISTU_GRAPH_EDGES.WEIGHT,
            mode="out",
            output="epath",
        )

    downstream_path_attrs, downstream_entity_dict = _calculate_path_attrs(
        neighborhood_paths,
        edges,
        vertices=descendants_list,
        weight_var=NAPISTU_GRAPH_EDGES.WEIGHT,
    )
    downstream_path_attrs = downstream_path_attrs.assign(
        node_orientation=NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM
    )

    # ancestors -> focal_node

    one_ancestors_df = one_neighborhood_df[
        one_neighborhood_df["relationship"] == GRAPH_RELATIONSHIPS.ANCESTORS
    ]
    ancestors_list = list(
        set(one_ancestors_df[NAPISTU_GRAPH_VERTICES.NAME].tolist()).union({sc_id})
    )

    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        neighborhood_paths = neighborhood_graph.get_shortest_paths(
            v=sc_id,
            to=ancestors_list,
            weights=NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
            mode="in",
            output="epath",
        )

    upstream_path_attrs, upstream_entity_dict = _calculate_path_attrs(
        neighborhood_paths,
        edges,
        vertices=ancestors_list,
        weight_var=NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
    )
    upstream_path_attrs = upstream_path_attrs.assign(
        node_orientation=NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM
    )

    return (
        downstream_path_attrs,
        downstream_entity_dict,
        upstream_path_attrs,
        upstream_entity_dict,
    )
