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
from napistu.network import net_utils
from napistu.network import paths

from napistu.constants import SBML_DFS
from napistu.constants import MINI_SBO_NAME_TO_POLARITY
from napistu.constants import MINI_SBO_TO_NAME

from napistu.network.constants import CPR_GRAPH_TYPES
from napistu.network.constants import NEIGHBORHOOD_NETWORK_TYPES
from napistu.network.constants import VALID_NEIGHBORHOOD_NETWORK_TYPES

logger = logging.getLogger(__name__)


def find_and_prune_neighborhoods(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
    compartmentalized_species: str | list[str],
    precomputed_distances: pd.DataFrame | None = None,
    network_type: str = NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
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
    cpr_graph : igraph.Graph
        A bipartite network connecting molecular species and reactions
    compartmentalized_species : [str] or str
        Compartmentalized species IDs for neighborhood centers
    precomputed_distances : pd.DataFrame or None
        If provided, an edgelist of origin->destination path weights and lengths
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

    neighborhoods = find_neighborhoods(
        sbml_dfs=sbml_dfs,
        cpr_graph=cpr_graph,
        compartmentalized_species=compartmentalized_species,
        network_type=network_type,
        order=order,
        verbose=verbose,
        precomputed_neighbors=precomputed_neighbors,
    )

    pruned_neighborhoods = prune_neighborhoods(neighborhoods, top_n=top_n)

    return pruned_neighborhoods


def load_neighborhoods(
    s_ids: list[str],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
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
    cpr_graph: igraph.Graph
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
    neighborhoods_dict: dict
        Outputs from find_and_prune_neighborhoods for each s_id

    """

    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"{output_dir} does not exist")

    neighborhood_prefix = create_neighborhood_prefix(network_type, order, top_n)
    vertices_path = os.path.join(output_dir, f"{neighborhood_prefix}_vertices.tsv")
    networks_path = os.path.join(output_dir, f"{neighborhood_prefix}_networks.pkl")
    neighborhood_paths = [vertices_path, networks_path]

    if all([os.path.isfile(x) for x in neighborhood_paths]) and overwrite is False:
        print(f"loading existing neighborhoods for {neighborhood_prefix}")

        all_neighborhoods_df = pd.read_csv(vertices_path, sep="\t")
        with open(networks_path, "rb") as in_file:
            neighborhoods_dict = pickle.load(in_file)

    else:
        print(f"creating neighborhoods based on {neighborhood_prefix}")

        all_neighborhoods_df, neighborhoods_dict = create_neighborhoods(
            s_ids=s_ids,
            sbml_dfs=sbml_dfs,
            cpr_graph=cpr_graph,
            network_type=network_type,
            order=order,
            top_n=top_n,
            verbose=verbose,
        )

        # save df
        all_neighborhoods_df.to_csv(vertices_path, sep="\t", index=False)

        # pickle neighborhoods
        with open(networks_path, "wb") as fh:
            pickle.dump(neighborhoods_dict, fh)

    return all_neighborhoods_df, neighborhoods_dict


def create_neighborhoods(
    s_ids: list[str],
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
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
    cpr_graph: igraph.Graph
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
    neighborhoods_dict: dict
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
    neighborhoods_dict = dict()
    for s_id in s_ids:
        query_sc_species = net_utils.compartmentalize_species(sbml_dfs, s_id)

        compartmentalized_species = query_sc_species[SBML_DFS.SC_ID].tolist()

        neighborhoods = find_and_prune_neighborhoods(
            sbml_dfs,
            cpr_graph,
            compartmentalized_species=compartmentalized_species,
            network_type=network_type,
            order=order,
            top_n=top_n,
            verbose=verbose,
        )

        # combine multiple neighborhoods

        neighborhood_entities = pd.concat(
            [
                neighborhoods[sc_id]["vertices"].assign(focal_sc_id=sc_id)
                for sc_id in neighborhoods.keys()
            ]
        ).assign(focal_s_id=s_id)

        neighborhood_species = neighborhood_entities.merge(
            sbml_dfs.compartmentalized_species[SBML_DFS.S_ID],
            left_on="name",
            right_index=True,
        )

        neighborhoods_list.append(neighborhood_species)
        neighborhoods_dict[s_id] = neighborhoods

    all_neighborhoods_df = pd.concat(neighborhoods_list).reset_index(drop=True)

    return all_neighborhoods_df, neighborhoods_dict


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
    graph_type: str = CPR_GRAPH_TYPES.REGULATORY,
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

    consensus_root = "/group/cpr/consensus"
    consensus_name = "reactome"
    consensus_outdir = os.path.join(consensus_root, consensus_name)

    if not os.path.isdir(neighborhood_outdir):
        raise FileNotFoundError(f"{neighborhood_outdir} does not exist")

    partition_output = os.path.join(
        neighborhood_outdir, f"partition_{selected_partition}"
    )
    # initialize an empty output
    if os.path.isdir(partition_output):
        print(f"removing existing directory: {partition_output}")
        shutil.rmtree(partition_output)
    os.makedirs(partition_output)

    # format partition s_ids

    sids_to_partition = pd.read_csv(os.path.join(neighborhood_outdir, "partitions.csv"))
    parition_sids_df = sids_to_partition[
        sids_to_partition["partition"] == selected_partition
    ]

    if parition_sids_df.shape[0] == 0:
        raise ValueError(f"No s_ids associated with partition {selected_partition}")

    parition_sids = parition_sids_df["s_id"].tolist()

    # read pathway and network data

    # read model containing Calico curations. this is primarily to support search programs
    # to not use these switch to refined.pkl
    refined_model_pkl_path = os.path.join(consensus_outdir, "curated.pkl")
    with open(refined_model_pkl_path, "rb") as in_file:
        refined_model = pickle.load(in_file)
    refined_model.validate()

    # load the graph
    cpr_graph = net_utils.read_network_pkl(
        model_prefix="curated",
        network_dir=consensus_outdir,
        directed=True,
        graph_type=graph_type,
    )

    all_neighborhoods_df, neighborhoods_dict = load_neighborhoods(
        s_ids=parition_sids,
        sbml_dfs=refined_model,
        cpr_graph=cpr_graph,
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
    cpr_graph: ig.Graph,
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
    cpr_graph: igraph.Graph
        network associated with sbml_dfs
    partitions_path: str
        Path to a directory containing folders for each partition's results
    n_partitions: int
        Number of partitions that exist

    Returns
    -------
    all_neighborhoods_df: pd.DataFrame
        A table containing all species in each query s_ids neighborhood
    neighborhoods_dict: dict
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
            cpr_graph=cpr_graph,
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
    neighborhoods_dict = dict(ChainMap(*path_dict_list))

    # TO DO - remove s_id duplication (these are present in the vertices table in the partition outputs)
    if not all(all_neighborhoods_df["s_id_x"] == all_neighborhoods_df["s_id_y"]):
        raise ValueError("The patch won't hold")
    all_neighborhoods_df = all_neighborhoods_df.drop(["s_id_y"], axis=1).rename(
        {"s_id_x": "s_id"}, axis=1
    )

    return all_neighborhoods_df, neighborhoods_dict


def find_neighborhoods(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
    compartmentalized_species: list[str],
    network_type: str = "downstream",
    order: int = 3,
    verbose: bool = True,
    precomputed_neighbors: pd.DataFrame | None = None,
) -> dict:
    """
    Find Neighborhood

    Create a network composed of all species and reactions within N steps of
      each of a set of compartmentalized species.

    Parameters
    ----------
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic molecular model
    cpr_graph : igraph.Graph
        A network connecting molecular species and reactions
    compartmentalized_species : [str]
        Compartmentalized species IDs for neighborhood centers
    network_type: str
        If the network is directed should neighbors be located "downstream",
        or "upstream" of each compartmentalized species. The "hourglass" option
        locates both upstream and downstream species.
    order: int
        Max steps away from center node
    verbose: bool
        Extra reporting
    precomputed_neighbors: pd.DataFrame or None
        If provided, a pre-filtered table of nodes nearby the compartmentalized species
        which will be used to skip on-the-fly neighborhood generation.

    Returns:
    ----------
    A dict containing the neighborhood of each compartmentalized species.
      Each entry in the dict is a dict of the subgraph, vertices, and edges.
    """

    if not isinstance(network_type, str):
        raise TypeError(f"network_type was a {type(network_type)} and must be a str")

    valid_network_types = ["downstream", "upstream", "hourglass"]
    if network_type not in valid_network_types:
        raise ValueError(
            f"network_type must be one of {', '.join(valid_network_types)}"
        )

    if not isinstance(order, int):
        raise TypeError(f"order was a {type(order)} and must be an int")

    # create a table which includes cspecies and reaction nearby each of the
    # focal compartmentalized_speecies
    neighborhood_df = _build_raw_neighborhood_df(
        cpr_graph=cpr_graph,
        compartmentalized_species=compartmentalized_species,
        network_type=network_type,
        order=order,
        precomputed_neighbors=precomputed_neighbors,
    )

    # format the vertices and edges in each compartmentalized species' network
    neighborhood_dict = {
        sc_id: create_neighborhood_dict_entry(
            sc_id, neighborhood_df, sbml_dfs, cpr_graph, verbose=verbose
        )
        for sc_id in compartmentalized_species
    }

    return neighborhood_dict


def create_neighborhood_dict_entry(
    sc_id: str,
    neighborhood_df: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cpr_graph: ig.Graph,
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
    cpr_graph: igraph.Graph
        A network connecting molecular species and reactions
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
        edge_sources: pd.DataFrame
            models that edges were derived from
        neighborhood_path_entities: dict
            upstream and downstream dicts representing entities in paths.
            If the keys are to be included in a neighborhood, the
            values should be as well in order to maintain connection to the
            focal node.
    """

    one_neighborhood_df = neighborhood_df[neighborhood_df["sc_id"] == sc_id]

    if verbose:
        _create_neighborhood_dict_entry_logging(sc_id, one_neighborhood_df, sbml_dfs)

    if not one_neighborhood_df["name"].eq(sc_id).any():
        raise ValueError(
            f"The focal node sc_id = {sc_id} was not in 'one_neighborhood_df'.\
            By convention it should be part of its neighborhood"
        )

    # create the subgraph formed by filtering to neighborhoods
    neighborhood_graph = cpr_graph.subgraph(
        cpr_graph.vs[one_neighborhood_df["neighbor"]], implementation="auto"
    )

    vertices = pd.DataFrame([v.attributes() for v in neighborhood_graph.vs])
    edges = pd.DataFrame([e.attributes() for e in neighborhood_graph.es])

    # add edge polarity: whether edges are activating, inhibiting or unknown
    if edges.shape[0] > 0:
        edges["link_polarity"] = (
            edges["sbo_term"].map(MINI_SBO_TO_NAME).map(MINI_SBO_NAME_TO_POLARITY)
        )

    try:
        edge_sources = net_utils.get_minimal_sources_edges(
            vertices.rename(columns={"name": "node"}), sbml_dfs
        )
    except Exception:
        edge_sources = None

    # to add weights to the network solve the shortest path problem
    # from the focal node to each neighbor
    # solve this problem separately whether a given neighbor is an
    # ancestor or descendant

    # focal node -> descendants

    one_descendants_df = one_neighborhood_df[
        one_neighborhood_df["relationship"] == "descendants"
    ]
    descendants_list = list(set(one_descendants_df["name"].tolist()).union({sc_id}))

    # hide warnings which are mostly just Dijkstra complaining about not finding neighbors
    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        neighborhood_paths = neighborhood_graph.get_shortest_paths(
            # focal node
            v=sc_id,
            to=descendants_list,
            weights="weights",
            mode="out",
            output="epath",
        )

    downstream_path_attrs, downstream_entity_dict = _calculate_path_attrs(
        neighborhood_paths, edges, vertices=descendants_list, weight_var="weights"
    )
    downstream_path_attrs = downstream_path_attrs.assign(node_orientation="downstream")

    # ancestors -> focal_node

    one_ancestors_df = one_neighborhood_df[
        one_neighborhood_df["relationship"] == "ancestors"
    ]
    ancestors_list = list(set(one_ancestors_df["name"].tolist()).union({sc_id}))

    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        neighborhood_paths = neighborhood_graph.get_shortest_paths(
            v=sc_id,
            to=ancestors_list,
            weights="upstream_weights",
            mode="in",
            output="epath",
        )

    upstream_path_attrs, upstream_entity_dict = _calculate_path_attrs(
        neighborhood_paths,
        edges,
        vertices=ancestors_list,
        weight_var="upstream_weights",
    )
    upstream_path_attrs = upstream_path_attrs.assign(node_orientation="upstream")

    # combine upstream and downstream shortest paths
    # in cases a node is upstream and downstream of the focal node
    # by taking the lowest path weight
    vertex_neighborhood_attrs = (
        pd.concat([downstream_path_attrs, upstream_path_attrs])
        .sort_values("path_weight")
        .groupby("neighbor")
        .first()
    )
    # label the focal node
    vertex_neighborhood_attrs.loc[sc_id, "node_orientation"] = "focal"

    # if the precomputed distances, graph and/or sbml_dfs are inconsistent
    # then the shortest paths search may just return empty lists
    # throw a clearer error message in this case.
    EXPECTED_VERTEX_ATTRS = {"final_from", "final_to", "net_polarity"}
    missing_vertex_attrs = EXPECTED_VERTEX_ATTRS.difference(
        set(vertex_neighborhood_attrs.columns.tolist())
    )

    if len(missing_vertex_attrs) > 0:
        raise ValueError(
            f"vertex_neighborhood_attrs did not contain the expected columns: {EXPECTED_VERTEX_ATTRS}."
            "This is likely because of inconsistencies between the precomputed distances, graph and/or sbml_dfs."
            "Please try net_utils.validate_assets() to check for consistency."
        )

    # add net_polarity to edges in addition to nodes
    edges = edges.merge(
        vertex_neighborhood_attrs.reset_index()[
            ["final_from", "final_to", "net_polarity"]
        ].dropna(),
        left_on=["from", "to"],
        right_on=["final_from", "final_to"],
        how="left",
    )

    vertices = vertices.merge(
        vertex_neighborhood_attrs, left_on="name", right_index=True
    )

    # drop nodes with a path length / weight of zero
    # which are NOT the focal node
    # these were cases where no path to/from the focal node to the query node was found
    disconnected_neighbors = vertices.query(
        "(not node_orientation == 'focal') and path_weight == 0"
    )
    vertices = vertices[~vertices.index.isin(disconnected_neighbors.index.tolist())]

    # add reference urls
    vertices = add_vertices_uri_urls(vertices, sbml_dfs)

    neighborhood_path_entities = {
        "downstream": downstream_entity_dict,
        "upstream": upstream_entity_dict,
    }

    # update graph with additional vertex and edge attributes
    updated_cpr_graph = ig.Graph.DictList(
        vertices=vertices.to_dict("records"),
        edges=edges.to_dict("records"),
        directed=cpr_graph.is_directed(),
        vertex_name_attr="name",
        edge_foreign_keys=("from", "to"),
    )

    outdict = {
        "graph": updated_cpr_graph,
        "vertices": vertices,
        "edges": edges,
        "edge_sources": edge_sources,
        "neighborhood_path_entities": neighborhood_path_entities,
    }

    return outdict


def _create_neighborhood_dict_entry_logging(
    sc_id: str, one_neighborhood_df: pd.DataFrame, sbml_dfs: sbml_dfs_core.SBML_dfs
):
    df_summary = one_neighborhood_df.copy()
    df_summary["node_type"] = [
        "species" if x else "reactions"
        for x in df_summary["name"].isin(sbml_dfs.compartmentalized_species.index)
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
    Add Vertices URI URLs

    Add a url variable to the neighborhood vertices pd.DataFrame

    Parameters
    ----------
    vertices: pd.DataFrame
        table of neighborhood vertices
    sbml_dfs: sbml_dfs_core.SBML_dfs
        consensus network model

    Returns
    -------
    vertices: pd.DataFrame
        input table with a url field

    """

    if vertices.shape[0] <= 0:
        raise ValueError("vertices must have at least one row")

    # add uri urls for each node

    # add s_ids
    neighborhood_species = vertices[vertices["node_type"] == "species"].merge(
        sbml_dfs.compartmentalized_species["s_id"],
        left_on="name",
        right_index=True,
        how="left",
    )

    # add a standard reference identifier
    neighborhood_species_aug = neighborhood_species.merge(
        sbml_dfs.get_uri_urls("species", neighborhood_species["s_id"]),
        left_on="s_id",
        right_index=True,
        how="left",
        # add pharos ids where available
    ).merge(
        sbml_dfs.get_uri_urls(
            "species", neighborhood_species["s_id"], required_ontology="pharos"
        ).rename("pharos"),
        left_on="s_id",
        right_index=True,
        how="left",
    )

    if sum(vertices["node_type"] == "reaction") > 0:
        neighborhood_reactions = vertices[vertices["node_type"] == "reaction"].merge(
            sbml_dfs.get_uri_urls(
                "reactions", vertices[vertices["node_type"] == "reaction"]["name"]
            ),
            left_on="name",
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

    pruned_neighborhoods_dict = dict()

    for an_sc_id in neighborhoods.keys():
        one_neighborhood = neighborhoods[an_sc_id]

        # filter to the desired number of vertices w/ lowest path_weight (from focal node)
        # filter neighborhood to high-weight vertices
        pruned_vertices = _prune_vertex_set(one_neighborhood, top_n=top_n)

        # reduce neighborhood to this set of high-weight vertices
        all_neighbors = pd.DataFrame({"name": one_neighborhood["graph"].vs["name"]})
        pruned_vertices_indices = all_neighbors[
            all_neighbors["name"].isin(pruned_vertices["name"])
        ].index.tolist()

        pruned_neighborhood = one_neighborhood["graph"].subgraph(
            one_neighborhood["graph"].vs[pruned_vertices_indices],
            implementation="auto",
        )

        pruned_edges = pd.DataFrame([e.attributes() for e in pruned_neighborhood.es])

        pruned_reactions = pruned_vertices[pruned_vertices["node_type"] == "reaction"][
            "name"
        ]

        if pruned_reactions.shape[0] != 0:
            if one_neighborhood["edge_sources"] is None:
                # allow for missing source information since this is currently optional
                pruned_edge_sources = one_neighborhood["edge_sources"]
            else:
                pruned_edge_sources = one_neighborhood["edge_sources"][
                    one_neighborhood["edge_sources"]["r_id"].isin(pruned_reactions)
                ]
        else:
            pruned_edge_sources = one_neighborhood["edge_sources"]

        pruned_neighborhoods_dict[an_sc_id] = {
            "graph": pruned_neighborhood,
            "vertices": pruned_vertices,
            "edges": pruned_edges,
            "edge_sources": pruned_edge_sources,
        }

    return pruned_neighborhoods_dict


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
        "species": "firebrick",
        "reaction": "dodgerblue",
    }

    edge_polarity_colors = {
        "ambiguous": "dimgray",
        "activation": "gold",
        "inhibition": "royalblue",
        "ambiguous activation": "palegoldenrod",
        "ambiguous inhibition": "powerblue",
        np.nan: "dimgray",
    }

    visual_style = {}  # type: dict[str,Any]
    visual_style["background"] = "black"
    visual_style["vertex_size"] = 10
    if name_nodes:
        visual_style["vertex_label"] = [
            textwrap.fill(x, 15) for x in neighborhood_graph.vs["node_name"]
        ]
    visual_style["vertex_label_color"] = "white"
    visual_style["vertex_label_size"] = 8
    visual_style["vertex_label_angle"] = 90
    visual_style["vertex_label_dist"] = 3
    visual_style["vertex_color"] = [
        color_dict[x] for x in neighborhood_graph.vs["node_type"]
    ]
    visual_style["edge_color"] = [
        edge_polarity_colors[x] for x in neighborhood_graph.es["net_polarity"]
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
        *precomputed_distances["sc_id_origin"].tolist(),
        *precomputed_distances["sc_id_dest"].tolist(),
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
        valid_origin = precomputed_distances["sc_id_origin"].isin(
            compartmentalized_species
        )
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        valid_dest = precomputed_distances["sc_id_dest"].isin(compartmentalized_species)

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
        cspecies_subset_precomputed_distances["path_length"] <= order
    ]

    # filter to retain top_n
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        top_descendants = (
            close_cspecies_subset_precomputed_distances[
                close_cspecies_subset_precomputed_distances["sc_id_origin"].isin(
                    compartmentalized_species
                )
            ]
            # sort by path_weight so we can retain the lowest weight neighbors
            .sort_values("path_weights")
            .groupby("sc_id_origin")
            .head(top_n)
        )

        logger.debug(f"N top_descendants {top_descendants.shape[0]}")

    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        top_ancestors = (
            close_cspecies_subset_precomputed_distances[
                close_cspecies_subset_precomputed_distances["sc_id_dest"].isin(
                    compartmentalized_species
                )
            ]
            # sort by path_upstream_weights so we can retain the lowest weight neighbors
            # we allow for upstream weights to differ from downstream weights
            # when creating a network in process_cpr_graph.
            #
            # the default network weighting penalizing an edge from a node
            # based on the number of children it has. this captures the idea
            # that if there are many children we might expect that each
            # of them is less likely to transduct an effect.
            # the logic is flipped if we are looking for ancestors where
            # we penalize based on the number of parents of a node when
            # we use it (i.e., the default upstream_weights).
            .sort_values("path_upstream_weights")
            .groupby("sc_id_dest")
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
            relationship="descendants",
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
            relationship="ancestors",
        )

        if upstream_reactions is not None:
            logger.debug(f"N upstream reactions {upstream_reactions.shape[0]}")

    # add the self links since sc_id_dest will be used to define
    # an sc_id_origin-specific subgraph
    identity_df = pd.DataFrame(
        {
            "sc_id_origin": compartmentalized_species,
            "sc_id_dest": compartmentalized_species,
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
        )[["sc_id_origin", "sc_id_dest"]].drop_duplicates()
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.DOWNSTREAM:
        precomputed_neighbors = pd.concat([top_descendants, downstream_reactions, identity_df])[  # type: ignore
            ["sc_id_origin", "sc_id_dest"]
        ].drop_duplicates()
    elif network_type == NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM:
        precomputed_neighbors = pd.concat([top_ancestors, upstream_reactions, identity_df])[  # type: ignore
            ["sc_id_origin", "sc_id_dest"]
        ].drop_duplicates()
    else:
        raise ValueError("This error shouldn't happen")

    return precomputed_neighbors


def _build_raw_neighborhood_df(
    cpr_graph: ig.Graph,
    compartmentalized_species: list[str],
    network_type: str,
    order: int,
    precomputed_neighbors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # report if network_type is not the default and will be ignored due to the network
    #   being undirected
    is_directed = cpr_graph.is_directed()
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
            cpr_graph=cpr_graph,
            compartmentalized_species=compartmentalized_species,
            relationship="descendants",
            order=order,
            precomputed_neighbors=precomputed_neighbors,
        )

    # create the "in-network" of ancestor nodes
    if network_type in [
        NEIGHBORHOOD_NETWORK_TYPES.UPSTREAM,
        NEIGHBORHOOD_NETWORK_TYPES.HOURGLASS,
    ]:
        ancestors_df = _find_neighbors(
            cpr_graph=cpr_graph,
            compartmentalized_species=compartmentalized_species,
            relationship="ancestors",
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
    neighborhood_df["name"] = [
        x["name"] for x in cpr_graph.vs[neighborhood_df["neighbor"]]
    ]

    return neighborhood_df


def _find_neighbors(
    cpr_graph: ig.Graph,
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
            pd.DataFrame({"name": cpr_graph.vs["name"]})
            .reset_index()
            .rename({"index": "neighbor"}, axis=1)
        )

        if relationship == "descendants":
            bait_id = "sc_id_origin"
            target_id = "sc_id_dest"
        elif relationship == "ancestors":
            bait_id = "sc_id_dest"
            target_id = "sc_id_origin"
        else:
            raise ValueError(
                f"relationship must be 'descendants' or 'ancestors' but was {relationship}"
            )

        neighbors_df = (
            precomputed_neighbors[
                precomputed_neighbors[bait_id].isin(compartmentalized_species)
            ]
            .merge(nodes_to_names.rename({"name": target_id}, axis=1))
            .rename({bait_id: "sc_id"}, axis=1)
            .drop([target_id], axis=1)
            .assign(relationship=relationship)
        )
    else:
        if relationship == "descendants":
            mode_type = "out"
        elif relationship == "ancestors":
            mode_type = "in"
        else:
            raise ValueError(
                f"relationship must be 'descendants' or 'ancestors' but was {relationship}"
            )

        neighbors = cpr_graph.neighborhood(
            # mode = out queries outgoing edges and is ignored if the network is undirected
            vertices=compartmentalized_species,
            order=order,
            mode=mode_type,
        )

        neighbors_df = pd.concat(
            [
                pd.DataFrame({"sc_id": c, "neighbor": x}, index=range(0, len(x)))
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

    if relationship == "descendants":
        bait_id = "sc_id_origin"
        target_id = "sc_id_dest"
    elif relationship == "ancestors":
        bait_id = "sc_id_dest"
        target_id = "sc_id_origin"
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
            sbml_dfs.reaction_species["sc_id"].isin(relatives_cspecies)
        ].value_counts("r_id")

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

    neighborhood_vertices = one_neighborhood["vertices"]

    indexed_neighborhood_species = neighborhood_vertices[
        neighborhood_vertices["node_type"] == "species"
    ].set_index("node_orientation")

    pruned_oriented_neighbors = list()
    for a_node_orientation in indexed_neighborhood_species.index.unique().tolist():
        vertex_subset = indexed_neighborhood_species.loc[a_node_orientation]
        if type(vertex_subset) is pd.Series:
            # handle cases where only one entry exists to DF->series coercion occurs
            vertex_subset = vertex_subset.to_frame().T

        sorted_vertex_set = vertex_subset.sort_values("path_weight")
        weight_cutoff = sorted_vertex_set["path_weight"].iloc[
            min(top_n - 1, sorted_vertex_set.shape[0] - 1)
        ]

        top_neighbors = sorted_vertex_set[
            sorted_vertex_set["path_weight"] <= weight_cutoff
        ]["name"].tolist()

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
        neighborhood_vertices["name"].isin(pruned_neighbors)
    ].reset_index(drop=True)

    return pruned_vertices


def _calculate_path_attrs(
    neighborhood_paths: list[list],
    edges: pd.DataFrame,
    vertices: list,
    weight_var: str = "weights",
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
                all_path_edges[weight_var].agg("sum").rename("path_weight"),
                all_path_edges.agg("size").rename("path_length"),
                all_path_edges["link_polarity"]
                .agg(paths._terminal_net_polarity)
                .rename("net_polarity"),
                # add the final edge since this can be used to add path attributes to edges
                # i.e., apply net_polarity to an edge
                all_path_edges["from"].agg("last").rename("final_from"),
                all_path_edges["to"].agg("last").rename("final_to"),
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
        path_length=0, path_weight=0, net_polarity=None
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
