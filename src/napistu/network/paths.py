from __future__ import annotations

import logging
import math
import warnings
from typing import Any

import igraph as ig
import pandas as pd
from napistu import sbml_dfs_core
from napistu import utils
from napistu.constants import CPR_PATH_REQ_VARS
from napistu.constants import MINI_SBO_NAME_TO_POLARITY
from napistu.constants import MINI_SBO_TO_NAME
from napistu.network import net_utils

logger = logging.getLogger(__name__)


def find_shortest_reaction_paths(
    cpr_graph: ig.Graph,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    origin: str,
    dest: str | list,
    weight_var: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Shortest Reaction Paths

    Find all shortest paths between an origin and destination entity

    Parameters
    ----------
    cpr_graph : igraph.Graph
        A bipartite network connecting molecular species and reactions
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A model formed by aggregating pathways
    origin : str
        A node to start at
    dest : str | list
        Node(s) to reach
    weight_var : str
        An edge attribute to use when forming a weighted shortest path

    Returns:
    ----------
    Node paths and edges pd.DataFrames
    """

    # update destination list to exclude the origin vertex
    if isinstance(dest, str):
        if origin == dest:
            logger.info("origin = dest; returning None")
            return None
    elif isinstance(dest, list):
        # drop list entries where
        dest = [d for d in dest if d != origin]
        if len(dest) == 0:
            logger.info("origin = dest; returning None")
            return None

    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        shortest_paths = cpr_graph.get_all_shortest_paths(
            origin, to=dest, weights=weight_var
        )

    if len(shortest_paths) == 0:
        return None

    # summarize the graph which is being evaluated
    with warnings.catch_warnings():
        # igraph throws warnings for each pair of unconnected species
        warnings.simplefilter("ignore")

        shortest_paths = cpr_graph.get_all_shortest_paths(
            origin, to=dest, weights=weight_var
        )

    # summarize the graph which is being evaluated
    cpr_graph_names = [v.attributes()["name"] for v in cpr_graph.vs]

    cpr_graph_edges = pd.DataFrame(
        {
            "from": cpr_graph.es.get_attribute_values("from"),
            "to": cpr_graph.es.get_attribute_values("to"),
            "weights": cpr_graph.es.get_attribute_values(weight_var),
            "sbo_term": cpr_graph.es.get_attribute_values("sbo_term"),
            "direction": cpr_graph.es.get_attribute_values("direction"),
        }
    )

    directed = cpr_graph.is_directed()

    # format shortest paths
    # summaries of nodes
    path_list = list()
    # summaries of edges
    edge_list = list()

    entry = 0
    for path in shortest_paths:
        path_df = (
            pd.DataFrame({"node": [cpr_graph_names[x] for x in path]})
            .reset_index()
            .rename(columns={"index": "step"})
            .assign(path=entry)
        )
        path_df["node_number"] = path

        # reconstruct edges
        path_edges = pd.DataFrame(
            {"from": path_df["node"][:-1].tolist(), "to": path_df["node"][1:].tolist()}
        ).assign(path=entry)

        # add weights to edges

        if directed:
            path_edges = path_edges.merge(
                cpr_graph_edges,
                left_on=["from", "to"],
                right_on=["from", "to"],
            )

            path_edges["link_polarity"] = (
                path_edges["sbo_term"]
                .map(MINI_SBO_TO_NAME)
                .map(MINI_SBO_NAME_TO_POLARITY)
            )
            # is the edge predicted to be activating, inhibiting or ambiguous?
            path_edges["net_polarity"] = _calculate_net_polarity(
                path_edges["link_polarity"]
            )

        else:
            # if undirected then edges utilized may not be defined in the edgelist

            path_edges["step"] = range(0, path_edges.shape[0])

            # allow for matching in either polarity, 1 and only 1 will exist
            path_edges = (
                pd.concat(
                    [
                        path_edges,
                        path_edges.rename(columns={"to": "from", "from": "to"}),
                    ]
                )
                .merge(
                    cpr_graph_edges,
                    left_on=["from", "to"],
                    right_on=["from", "to"],
                    # keep at most 1 entry per step
                )
                .sort_values(["step", "weights"])
                .groupby("step")
                .first()
                .reset_index()
            )

            if path_edges.shape[0] != path_df.shape[0] - 1:
                raise ValueError(
                    "Something has gone wrong when merging attribute onto undirected edges"
                )

            # in an undirected graph it wouldn't make sense to use molecule's
            # SBO terms to determine the likely sign of a regulatory effect
            path_edges = path_edges.assign(link_polarity="ambiguous").assign(
                net_polarity="ambiguous"
            )

            # resort to recover the pre-merge order
            path_edges = path_edges.sort_values(["path", "step"]).drop("step", axis=1)

        # add weights to nodes
        path_df["weights"] = [0] + path_edges["weights"].tolist()

        path_list.append(path_df)
        edge_list.append(path_edges)
        entry += 1

    paths_df_raw = pd.concat(path_list).reset_index(drop=True)
    edges_df = pd.concat(edge_list).reset_index(drop=True)

    # annotate reactions
    labelled_reactions = _label_path_reactions(sbml_dfs, paths_df_raw)

    # annotate species
    labelled_species = (
        # find species among nodes
        paths_df_raw.merge(
            sbml_dfs.compartmentalized_species,
            left_on="node",
            right_index=True,
            how="inner",
        )
        .loc[:, paths_df_raw.columns.tolist()]
        .merge(sbml_dfs.compartmentalized_species, left_on="node", right_index=True)[
            ["step", "node", "path", "sc_name", "node_number", "weights", "s_id"]
        ]
        .rename(columns={"sc_name": "label"})
        .assign(node_type="species")
    )

    # add uri urls
    labelled_species = labelled_species.merge(
        sbml_dfs.get_uri_urls("species", labelled_species["s_id"].tolist()),
        left_on="s_id",
        right_index=True,
        how="left",
    ).drop("s_id", axis=1)

    paths_df = (
        pd.concat([labelled_reactions, labelled_species])
        .sort_values(["path", "step"])
        .fillna("")
    )

    return paths_df, edges_df


def find_all_shortest_reaction_paths(
    cpr_graph: ig.Graph,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    target_species_paths: pd.DataFrame,
    weight_var: str = "weights",
    precomputed_distances: pd.DataFrame | None = None,
):
    """
    Shortest Reaction Paths

    Find all shortest paths between a source and destination entity

    Parameters
    ----------
    cpr_graph : igraph.Graph
        A bipartite network connecting molecular species and reactions
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    target_species_paths : pd.DataFrame
        Pairs of source and destination compartmentalized species; produced by compartmentalize_species_pairs()
    weight_var : str
        An edge attribute to use when forming a weighted shortest path
    precomputed_distances : pd.DataFrame | None
        A table containing precalculated path summaries between pairs of compartmentalized species

    Returns:
    ----------
    all_shortest_reaction_paths_df : pd.DataFrame
        Nodes in all shortest paths
    all_shortest_reaction_path_edges_df : pd.DataFrame
        Edges in all shortest paths
    edge_sources : pd.DataFrame
        Sources of edge identifying the models where they originated
    paths_graph : igraph.Graph
        Network formed by all shortest paths
    """

    # find the shortest path between the origin and dest node for all pairs on target_species_paths

    if not isinstance(weight_var, str):
        raise TypeError(f"weight_var must be a str, but was {type(weight_var)}")

    # filter to valid paths if precomputed distances are provided
    target_species_paths = _filter_paths_by_precomputed_distances(
        target_species_paths, precomputed_distances
    )

    all_shortest_reaction_paths = list()
    all_shortest_reaction_path_edges = list()
    for i in range(target_species_paths.shape[0]):
        one_search = target_species_paths.iloc[i]

        paths = find_shortest_reaction_paths(
            cpr_graph,
            sbml_dfs,
            origin=one_search["sc_id_origin"],
            dest=one_search["sc_id_dest"],
            weight_var=weight_var,
        )

        if paths is None:
            continue

        shortest_paths_df, shortest_path_edges_df = paths

        all_shortest_reaction_paths.append(
            shortest_paths_df.assign(
                origin=one_search["sc_id_origin"], dest=one_search["sc_id_dest"]
            )
        )
        all_shortest_reaction_path_edges.append(
            shortest_path_edges_df.assign(
                origin=one_search["sc_id_origin"], dest=one_search["sc_id_dest"]
            )
        )

    if (
        len(all_shortest_reaction_paths) == 0
        or len(all_shortest_reaction_path_edges) == 0
    ):
        raise ValueError("No paths found")

    all_shortest_reaction_paths_df = pd.concat(
        all_shortest_reaction_paths
    ).reset_index()
    all_shortest_reaction_path_edges_df = pd.concat(
        all_shortest_reaction_path_edges
    ).reset_index()

    # at a minimal set of pathway sources to organize reactions
    edge_sources = net_utils.get_minimal_sources_edges(
        all_shortest_reaction_paths_df, sbml_dfs
    )

    # create a new small network of shortest paths
    unique_path_nodes = (
        all_shortest_reaction_paths_df.groupby(["node"])
        .first()
        .reset_index()
        .drop(columns=["index", "step", "path", "origin", "dest"])
    )

    directed = cpr_graph.is_directed()
    paths_graph = ig.Graph.DictList(
        vertices=unique_path_nodes.to_dict("records"),
        edges=all_shortest_reaction_path_edges_df.to_dict("records"),
        directed=directed,
        vertex_name_attr="node",
        edge_foreign_keys=("from", "to"),
    )

    return (
        all_shortest_reaction_paths_df,
        all_shortest_reaction_path_edges_df,
        edge_sources,
        paths_graph,
    )


def plot_shortest_paths(paths_graph: ig.Graph) -> ig.plot:
    """Plot a shortest paths graph."""

    if "label" not in paths_graph.vs.attributes():
        logger.warning(
            "label was not defined as a vertex attribute so paths will not be colored"
        )
        paths_graph.vs.set_attribute_values("label", "")

    paths_graph_layout = paths_graph.layout("kk")

    color_dict = {"reaction": "dodgerblue", "species": "firebrick"}

    visual_style = {}  # type: dict[str,Any]
    visual_style["background"] = "black"
    visual_style["vertex_size"] = 10
    visual_style["vertex_label"] = [
        net_utils.safe_fill(x) for x in paths_graph.vs["label"]
    ]
    visual_style["vertex_label_color"] = "white"
    visual_style["vertex_label_size"] = 8
    visual_style["vertex_label_angle"] = 90
    visual_style["vertex_color"] = [color_dict[x] for x in paths_graph.vs["node_type"]]
    visual_style["edge_width"] = [math.sqrt(x) for x in paths_graph.es["weights"]]
    visual_style["edge_color"] = "dimgray"
    visual_style["layout"] = paths_graph_layout
    visual_style["bbox"] = (2000, 2000)
    visual_style["margin"] = 50

    return ig.plot(paths_graph, **visual_style)


def _filter_paths_by_precomputed_distances(
    all_species_pairs: pd.DataFrame, precomputed_distances: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Filter source -> destination pairs based on precomputed distances if they were provided."""

    utils.match_pd_vars(all_species_pairs, CPR_PATH_REQ_VARS).assert_present()

    if precomputed_distances is None:
        logger.info(
            "precomputed_distances were not provided; all paths will be calculated on-the-fly"
        )
        return all_species_pairs
    else:
        if not isinstance(precomputed_distances, pd.DataFrame):
            raise TypeError('"precomputed_distances" must be a pd.DataFrame')

    utils.match_pd_vars(precomputed_distances, CPR_PATH_REQ_VARS).assert_present()

    # filter to pairs which are connected in the pre-computed distances table
    valid_all_species_pairs = all_species_pairs.merge(
        precomputed_distances[["sc_id_origin", "sc_id_dest"]],
    )

    return valid_all_species_pairs


def _calculate_net_polarity(link_polarity_series: pd.Series) -> str:
    """Determine whether a path implies activation, inhbition, or an ambiguous regulatory relationship."""

    assert isinstance(link_polarity_series, pd.Series)
    assert link_polarity_series.name == "link_polarity"

    # loop through loop polarity and
    # determine the cumulative polarity account for inhibition steps which flip polarity
    # and ambiguous steps which will add an ambiguous label to the net result

    observed_polarities = set(link_polarity_series.tolist())  # type: set[str]
    valid_polarities = {"activation", "inhibition", "ambiguous"}  # type: set[str]
    invalid_polarities = observed_polarities.difference(
        valid_polarities
    )  # type: set[str]
    if len(invalid_polarities) > 0:
        raise ValueError(
            f"Some edge polarities were invalid: {', '.join(invalid_polarities)}. "
            f"Valid polarities are {', '.join(valid_polarities)}."
        )

    # catch fully ambiguous case
    if link_polarity_series.eq("ambiguous").all():
        running_polarity = [
            "ambiguous" for i in range(link_polarity_series.shape[0])
        ]  # type : list[str]
        return running_polarity

    running_polarity = list()  # type : list[str]
    current_polarity = 1
    ambig_prefix = ""

    for polarity in link_polarity_series:
        if polarity == "ambiguous":
            # once a polarity becomes ambiguous it is stuck
            ambig_prefix = "ambiguous "
        if polarity == "inhibition":
            current_polarity = current_polarity * -1

        if current_polarity == 1:
            running_polarity.append(ambig_prefix + "activation")
        else:
            running_polarity.append(ambig_prefix + "inhibition")

    return running_polarity


def _terminal_net_polarity(link_polarity_series: pd.Series) -> str:
    """Figure out the net polarity for the vertex at the end of a path."""

    # calculate net polarity but only look at the final value
    net_polarity = _calculate_net_polarity(link_polarity_series)
    return net_polarity[-1]


def _patch(x: Any):
    logger.info("silly stub to define Any")


def _label_path_reactions(sbml_dfs: sbml_dfs_core.SBML_dfs, paths_df: pd.DataFrame):
    """Create labels for reactions in a shortest path."""

    # annotate reactions
    # find reactions among nodes
    reaction_paths = paths_df.merge(
        sbml_dfs.reactions, left_on="node", right_index=True, how="inner"
    ).loc[:, paths_df.columns.tolist()]

    if reaction_paths.shape[0] == 0:
        # the path doesn't contain any reactions
        # this can happen with the "regulatory" model
        # network_type specification
        labelled_reactions = None
    else:
        # add reaction label based off stoichiometry + the r_name
        reaction_info = (
            pd.concat(
                [
                    sbml_dfs_core.reaction_summaries(sbml_dfs, r_ids=x)
                    for x in set(reaction_paths["node"])
                ]
            )
            .to_frame()
            .join(sbml_dfs.reactions["r_name"])
        )

        labelled_reactions = (
            reaction_paths.merge(reaction_info, left_on="node", right_index=True)
            .rename(columns={"r_name": "label"})
            .assign(node_type="reaction")
        )

        # add uri urls
        labelled_reactions = labelled_reactions.merge(
            sbml_dfs.get_uri_urls("reactions", labelled_reactions["node"].tolist()),
            left_on="node",
            right_index=True,
            how="left",
        )

    return labelled_reactions
