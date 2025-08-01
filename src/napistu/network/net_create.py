from __future__ import annotations

import copy
import logging
import random
from typing import Optional

import igraph as ig
import pandas as pd

from napistu import sbml_dfs_core
from napistu import utils
from napistu.network import ng_utils
from napistu.network import net_create_utils
from napistu.network.ng_core import NapistuGraph
from napistu.constants import (
    MINI_SBO_FROM_NAME,
    SBO_MODIFIER_NAMES,
    SBOTERM_NAMES,
    SBML_DFS,
    SBML_DFS_METHOD_DEFS,
)
from napistu.network.constants import (
    GRAPH_WIRING_APPROACHES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_EDGE_DIRECTIONS,
    NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NAPISTU_WEIGHTING_STRATEGIES,
    VALID_GRAPH_WIRING_APPROACHES,
    VALID_WEIGHTING_STRATEGIES,
    SOURCE_VARS_DICT,
    DROP_REACTIONS_WHEN,
)

logger = logging.getLogger(__name__)


def create_napistu_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_graph_attrs: Optional[dict] = None,
    directed: bool = True,
    edge_reversed: bool = False,
    wiring_approach: str = GRAPH_WIRING_APPROACHES.REGULATORY,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
    verbose: bool = False,
    custom_transformations: Optional[dict] = None,
) -> NapistuGraph:
    """
    Create a NapistuGraph network from a mechanistic network using one of a set of wiring approaches.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A model formed by aggregating pathways.
    reaction_graph_attrs : dict, optional
        Dictionary containing attributes to pull out of reaction_data and a weighting scheme for the graph.
    directed : bool, optional
        Whether to create a directed (True) or undirected (False) graph. Default is True.
    edge_reversed : bool, optional
        Whether to reverse the directions of edges. Default is False.
    wiring_approach : str, optional
        Type of graph to create. Valid values are:
            - 'bipartite': substrates and modifiers point to the reaction they drive, this reaction points to products
            - 'regulatory': non-enzymatic modifiers point to enzymes, enzymes point to substrates and products
            - 'surrogate': non-enzymatic modifiers -> substrates -> enzymes -> reaction -> products
            - 'bipartite_og': old method for generating a true bipartite graph. Retained primarily for regression testing.
    drop_reactions_when : str, optional
        The condition under which to drop reactions as a network vertex. Valid values are:
            - 'same_tier': drop reactions when all participants are on the same tier of a wiring hierarchy
            - 'edgelist': drop reactions when the reaction species are only 2 (1 reactant + 1 product)
            - 'always': drop reactions regardless of tiers
    verbose : bool, optional
        Extra reporting. Default is False.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.

    Returns
    -------
    NapistuGraph
        A NapistuGraph network (subclass of igraph.Graph).

    Raises
    ------
    ValueError
        If wiring_approach is not valid or if required attributes are missing.
    """

    if reaction_graph_attrs is None:
        reaction_graph_attrs = {}

    if wiring_approach not in VALID_GRAPH_WIRING_APPROACHES + ["bipartite_og"]:
        raise ValueError(
            f"wiring_approach is not a valid value ({wiring_approach}), valid values are {','.join(VALID_GRAPH_WIRING_APPROACHES)}"
        )

    # fail fast if reaction_graph_attrs is not properly formatted
    for k in reaction_graph_attrs.keys():
        ng_utils._validate_entity_attrs(
            reaction_graph_attrs[k], custom_transformations=custom_transformations
        )

    working_sbml_dfs = copy.deepcopy(sbml_dfs)
    reaction_species_counts = working_sbml_dfs.reaction_species.value_counts(
        SBML_DFS.R_ID
    )
    valid_reactions = reaction_species_counts[reaction_species_counts > 1].index
    # due to autoregulation reactions, and removal of cofactors some
    # reactions may have 1 (or even zero) species. drop these.

    n_dropped_reactions = working_sbml_dfs.reactions.shape[0] - len(valid_reactions)
    if n_dropped_reactions != 0:
        logger.info(
            f"Dropping {n_dropped_reactions} reactions with <= 1 reaction species "
            "these underspecified reactions may be due to either unrepresented "
            "autoregulation and/or removal of cofactors."
        )

        working_sbml_dfs.reactions = working_sbml_dfs.reactions[
            working_sbml_dfs.reactions.index.isin(valid_reactions)
        ]
        working_sbml_dfs.reaction_species = working_sbml_dfs.reaction_species[
            working_sbml_dfs.reaction_species[SBML_DFS.R_ID].isin(valid_reactions)
        ]

    logger.info(
        "Organizing all network nodes (compartmentalized species and reactions)"
    )

    network_nodes = list()
    network_nodes.append(
        working_sbml_dfs.compartmentalized_species.reset_index()[
            [SBML_DFS.SC_ID, SBML_DFS.SC_NAME]
        ]
        .rename(columns={SBML_DFS.SC_ID: "node_id", SBML_DFS.SC_NAME: "node_name"})
        .assign(node_type=NAPISTU_GRAPH_NODE_TYPES.SPECIES)
    )
    network_nodes.append(
        working_sbml_dfs.reactions.reset_index()[[SBML_DFS.R_ID, SBML_DFS.R_NAME]]
        .rename(columns={SBML_DFS.R_ID: "node_id", SBML_DFS.R_NAME: "node_name"})
        .assign(node_type=NAPISTU_GRAPH_NODE_TYPES.REACTION)
    )

    # rename nodes to name since it is treated specially
    network_nodes_df = pd.concat(network_nodes).rename(
        columns={"node_id": NAPISTU_GRAPH_VERTICES.NAME}
    )

    logger.info(f"Formatting edges as a {wiring_approach} graph")

    if wiring_approach == "bipartite_og":
        network_edges = _create_napistu_graph_bipartite(working_sbml_dfs)
    elif wiring_approach in VALID_GRAPH_WIRING_APPROACHES:
        # pass wiring_approach so that an appropriate tiered schema can be used.
        network_edges = create_napistu_graph_wiring(
            working_sbml_dfs, wiring_approach, drop_reactions_when
        )
    else:
        raise NotImplementedError("Invalid wiring_approach")

    logger.info("Adding reversibility and other meta-data from reactions_data")
    augmented_network_edges = _augment_network_edges(
        network_edges,
        working_sbml_dfs,
        reaction_graph_attrs,
        custom_transformations=custom_transformations,
    )

    logger.info(
        "Creating reverse reactions for reversible reactions on a directed graph"
    )
    if directed:
        directed_network_edges = pd.concat(
            [
                # assign forward edges
                augmented_network_edges.assign(
                    **{
                        NAPISTU_GRAPH_EDGES.DIRECTION: NAPISTU_GRAPH_EDGE_DIRECTIONS.FORWARD
                    }
                ),
                # create reverse edges for reversible reactions
                _reverse_network_edges(augmented_network_edges),
            ]
        )
    else:
        directed_network_edges = augmented_network_edges.assign(
            **{NAPISTU_GRAPH_EDGES.DIRECTION: NAPISTU_GRAPH_EDGE_DIRECTIONS.UNDIRECTED}
        )

    # de-duplicate edges
    unique_edges = (
        directed_network_edges.groupby(
            [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
        )
        .first()
        .reset_index()
    )

    if unique_edges.shape[0] != directed_network_edges.shape[0]:
        logger.warning(
            f"{directed_network_edges.shape[0] - unique_edges.shape[0]} edges were dropped "
            "due to duplicated origin -> target relationiships, use verbose for "
            "more information"
        )

        if verbose:
            # report duplicated edges
            grouped_edges = directed_network_edges.groupby(
                [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
            )
            duplicated_edges = [
                grouped_edges.get_group(x)
                for x in grouped_edges.groups
                if grouped_edges.get_group(x).shape[0] > 1
            ]
            example_duplicates = pd.concat(
                random.sample(duplicated_edges, min(5, len(duplicated_edges)))
            )

            logger.warning(utils.style_df(example_duplicates, headers="keys"))

    # convert nodes and edgelist into an igraph network
    logger.info("Formatting cpr_graph output")
    napistu_ig_graph = ig.Graph.DictList(
        vertices=network_nodes_df.to_dict("records"),
        edges=unique_edges.to_dict("records"),
        directed=directed,
        vertex_name_attr=NAPISTU_GRAPH_VERTICES.NAME,
        edge_foreign_keys=(NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO),
    )

    # delete singleton nodes (most of these will be reaction nodes associated with pairwise interactions)

    # Always return NapistuGraph
    napistu_graph = NapistuGraph.from_igraph(
        napistu_ig_graph, wiring_approach=wiring_approach, is_reversed=edge_reversed
    )

    # remove singleton nodes (mostly reactions that are not part of any interaction)
    napistu_graph.remove_isolated_vertices()

    if edge_reversed:
        logger.info("Applying edge reversal using reversal utilities")
        napistu_graph.reverse_edges()

    return napistu_graph


def process_napistu_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_graph_attrs: Optional[dict] = None,
    directed: bool = True,
    edge_reversed: bool = False,
    wiring_approach: str = GRAPH_WIRING_APPROACHES.BIPARTITE,
    weighting_strategy: str = NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
    verbose: bool = False,
    custom_transformations: dict = None,
) -> NapistuGraph:
    """
    Process Consensus Graph.

    Sets up a NapistuGraph network and then adds weights and other malleable attributes.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A model formed by aggregating pathways.
    reaction_graph_attrs : dict, optional
        Dictionary containing attributes to pull out of reaction_data and a weighting scheme for the graph.
    directed : bool, optional
        Whether to create a directed (True) or undirected (False) graph. Default is True.
    edge_reversed : bool, optional
        Whether to reverse the directions of edges. Default is False.
    wiring_approach : str, optional
        Type of graph to create. See `create_napistu_graph` for valid values.
    weighting_strategy : str, optional
        A network weighting strategy. Options:
            - 'unweighted': all weights (and upstream_weight for directed graphs) are set to 1.
            - 'topology': weight edges by the degree of the source nodes favoring nodes with few connections.
            - 'mixed': transform edges with a quantitative score based on reaction_attrs; and set edges without quantitative score as a source-specific weight.
    verbose : bool, optional
        Extra reporting. Default is False.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.

    Returns
    -------
    NapistuGraph
        A weighted NapistuGraph network (subclass of igraph.Graph).
    """

    if reaction_graph_attrs is None:
        reaction_graph_attrs = {}

    logging.info("Constructing network")
    napistu_graph = create_napistu_graph(
        sbml_dfs,
        reaction_graph_attrs,
        directed=directed,
        edge_reversed=edge_reversed,
        wiring_approach=wiring_approach,
        verbose=verbose,
        custom_transformations=custom_transformations,
    )

    if "reactions" in reaction_graph_attrs.keys():
        reaction_attrs = reaction_graph_attrs["reactions"]
    else:
        reaction_attrs = dict()

    logging.info(f"Adding edge weights with an {weighting_strategy} strategy")

    weighted_napistu_graph = add_graph_weights(
        napistu_graph=napistu_graph,
        reaction_attrs=reaction_attrs,
        weighting_strategy=weighting_strategy,
    )

    return weighted_napistu_graph


def create_napistu_graph_wiring(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    wiring_approach: str,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
) -> pd.DataFrame:
    """
    Turn an sbml_dfs model into a tiered graph which links upstream entities to downstream ones.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing the model data.
    wiring_approach : str
        The wiring approach to use for the graph.
    drop_reactions_when : str, optional
        The condition under which to drop reactions as a network vertex. Default is 'same_tier'.

    Returns
    -------
    pd.DataFrame
        DataFrame representing the tiered network edges.

    Raises
    ------
    ValueError
        If invalid SBO terms are present or required attributes are missing.
    """

    # organize reaction species for defining connections
    logger.info(
        f"Turning {sbml_dfs.reaction_species.shape[0]} reactions species into edges."
    )

    all_reaction_edges_df = net_create_utils.wire_reaction_species(
        sbml_dfs.reaction_species, wiring_approach, drop_reactions_when
    )

    logger.info("Adding species features to edges.")

    # add compartmentalized species summaries to weight edges
    cspecies_features = sbml_dfs.get_cspecies_features().drop(
        columns=[
            SBML_DFS_METHOD_DEFS.SC_DEGREE,
            SBML_DFS_METHOD_DEFS.SC_CHILDREN,
            SBML_DFS_METHOD_DEFS.SC_PARENTS,
        ]
    )

    # Determine which edges are from reactions vs species
    is_from_reaction = all_reaction_edges_df[NAPISTU_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index
    )

    # Merge species features with edges
    # For edges where FROM is a species (not reaction), use FROM node's features
    # For edges where FROM is a reaction, use TO node's features
    decorated_all_reaction_edges_df = pd.concat(
        [
            all_reaction_edges_df[~is_from_reaction].merge(
                cspecies_features, left_on=NAPISTU_GRAPH_EDGES.FROM, right_index=True
            ),
            all_reaction_edges_df[is_from_reaction].merge(
                cspecies_features, left_on=NAPISTU_GRAPH_EDGES.TO, right_index=True
            ),
        ]
    ).sort_index()

    if all_reaction_edges_df.shape[0] != decorated_all_reaction_edges_df.shape[0]:
        msg = (
            "'decorated_all_reaction_edges_df' and 'all_reaction_edges_df' should\n"
            "have the same number of rows but they did not"
        )

        raise ValueError(msg)

    logger.info(f"Done preparing {wiring_approach} graph")

    return decorated_all_reaction_edges_df


def add_graph_weights(
    napistu_graph: NapistuGraph,
    reaction_attrs: dict,
    weighting_strategy: str = NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
) -> NapistuGraph:
    """
    Add Graph Weights to a NapistuGraph using a specified weighting strategy.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A graphical network of molecules/reactions (nodes) and edges linking them (subclass of igraph.Graph).
    reaction_attrs : dict
        An optional dict of reaction attributes.
    weighting_strategy : str, optional
        A network weighting strategy. Options:
            - 'unweighted': all weights (and upstream_weight for directed graphs) are set to 1.
            - 'topology': weight edges by the degree of the source nodes favoring nodes emerging from nodes with few connections.
            - 'mixed': transform edges with a quantitative score based on reaction_attrs; and set edges without quantitative score as a source-specific weight.

    Returns
    -------
    NapistuGraph
        The weighted NapistuGraph.

    Raises
    ------
    ValueError
        If weighting_strategy is not valid.
    """

    napistu_graph_updated = copy.deepcopy(napistu_graph)

    ng_utils._validate_entity_attrs(reaction_attrs)

    if weighting_strategy not in VALID_WEIGHTING_STRATEGIES:
        raise ValueError(
            f"weighting_strategy was {weighting_strategy} and must be one of: "
            f"{', '.join(VALID_WEIGHTING_STRATEGIES)}"
        )

    topology_weighted_graph = napistu_graph_updated.add_topology_weights()

    if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY:
        # count parents and children and create weights based on them

        topology_weighted_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] = (
            topology_weighted_graph.es["topo_weights"]
        )
        if napistu_graph_updated.is_directed():
            topology_weighted_graph.es[NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT] = (
                topology_weighted_graph.es["upstream_topo_weights"]
            )

        return topology_weighted_graph

    if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED:
        # set weights as a constant
        topology_weighted_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] = 1
        if napistu_graph_updated.is_directed():
            topology_weighted_graph.es[NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT] = 1
        return topology_weighted_graph

    if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.MIXED:
        return _add_graph_weights_mixed(topology_weighted_graph, reaction_attrs)

    raise ValueError(f"No logic implemented for {weighting_strategy}")


def _create_napistu_graph_bipartite(sbml_dfs: sbml_dfs_core.SBML_dfs) -> pd.DataFrame:
    """
    Turn an sbml_dfs model into a bipartite graph linking molecules to reactions.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing the model data.

    Returns
    -------
    pd.DataFrame
        DataFrame representing the bipartite network edges.
    """

    # setup edges
    network_edges = (
        sbml_dfs.reaction_species.reset_index()[
            [SBML_DFS.R_ID, SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM]
        ]
        # rename species and reactions to reflect from -> to edges
        .rename(
            columns={
                SBML_DFS.SC_ID: NAPISTU_GRAPH_NODE_TYPES.SPECIES,
                SBML_DFS.R_ID: NAPISTU_GRAPH_NODE_TYPES.REACTION,
            }
        )
    )
    # add back an r_id variable so that each edge is annotated by a reaction
    network_edges[NAPISTU_GRAPH_EDGES.R_ID] = network_edges[
        NAPISTU_GRAPH_NODE_TYPES.REACTION
    ]

    # add edge weights
    cspecies_features = sbml_dfs.get_cspecies_features().drop(
        columns=[
            SBML_DFS_METHOD_DEFS.SC_DEGREE,
            SBML_DFS_METHOD_DEFS.SC_CHILDREN,
            SBML_DFS_METHOD_DEFS.SC_PARENTS,
        ]
    )
    network_edges = network_edges.merge(
        cspecies_features, left_on=NAPISTU_GRAPH_NODE_TYPES.SPECIES, right_index=True
    )

    # Ensure species_type is properly named for edge attributes
    if SBML_DFS_METHOD_DEFS.SPECIES_TYPE in network_edges.columns:
        network_edges[NAPISTU_GRAPH_EDGES.SPECIES_TYPE] = network_edges[
            SBML_DFS_METHOD_DEFS.SPECIES_TYPE
        ]

    # if directed then flip substrates and modifiers to the origin edge
    edge_vars = network_edges.columns.tolist()

    origins = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] <= 0]
    origin_edges = origins.loc[:, [edge_vars[1], edge_vars[0]] + edge_vars[2:]].rename(
        columns={
            NAPISTU_GRAPH_NODE_TYPES.SPECIES: NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_NODE_TYPES.REACTION: NAPISTU_GRAPH_EDGES.TO,
        }
    )

    dests = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] > 0]
    dest_edges = dests.rename(
        columns={
            NAPISTU_GRAPH_NODE_TYPES.REACTION: NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_NODE_TYPES.SPECIES: NAPISTU_GRAPH_EDGES.TO,
        }
    )

    network_edges = pd.concat([origin_edges, dest_edges])

    return network_edges


def _add_graph_weights_mixed(
    napistu_graph: NapistuGraph, reaction_attrs: dict
) -> NapistuGraph:
    """
    Weight a NapistuGraph using a mixed approach combining source-specific weights and existing edge weights.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        The network to weight (subclass of igraph.Graph).
    reaction_attrs : dict
        Dictionary of reaction attributes to use for weighting.

    Returns
    -------
    NapistuGraph
        The weighted NapistuGraph.
    """

    edges_df = napistu_graph.get_edge_dataframe()

    calibrated_edges = ng_utils.apply_weight_transformations(edges_df, reaction_attrs)
    calibrated_edges = _create_source_weights(calibrated_edges, "source_wt")

    score_vars = list(reaction_attrs.keys())
    score_vars.append("source_wt")

    logger.info(f"Creating mixed scores based on {', '.join(score_vars)}")

    calibrated_edges[NAPISTU_GRAPH_EDGES.WEIGHT] = calibrated_edges[score_vars].min(
        axis=1
    )

    napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] = calibrated_edges[
        NAPISTU_GRAPH_EDGES.WEIGHT
    ]
    if napistu_graph.is_directed():
        napistu_graph.es[NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT] = calibrated_edges[
            NAPISTU_GRAPH_EDGES.WEIGHT
        ]

    # add other attributes and update transformed attributes
    napistu_graph.es["source_wt"] = calibrated_edges["source_wt"]
    for k in reaction_attrs.keys():
        napistu_graph.es[k] = calibrated_edges[k]

    return napistu_graph


def _add_edge_attr_to_vertex_graph(
    napistu_graph: NapistuGraph,
    edge_attr_list: list,
    shared_node_key: str = "r_id",
) -> NapistuGraph:
    """
    Merge edge attribute(s) from edge_attr_list to vertices of a NapistuGraph.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A graph generated by create_napistu_graph() (subclass of igraph.Graph).
    edge_attr_list : list
        A list containing attributes to pull out of edges, then to add to vertices.
    shared_node_key : str, optional
        Key in edge that is shared with vertex, to map edge ids to corresponding vertex ids. Default is "r_id".

    Returns
    -------
    NapistuGraph
        The input NapistuGraph with additional vertex attributes added from edge attributes.
    """

    if len(edge_attr_list) == 0:
        logger.warning(
            "No edge attributes were passed, " "thus return the input graph."
        )
        return napistu_graph

    graph_vertex_df = napistu_graph.get_vertex_dataframe()
    graph_edge_df = napistu_graph.get_edge_dataframe()

    if shared_node_key not in graph_edge_df.columns.to_list():
        logger.warning(
            f"{shared_node_key} is not in the current edge attributes. "
            "shared_node_key must be an existing edge attribute"
        )
        return napistu_graph

    graph_edge_df_sub = graph_edge_df.loc[:, [shared_node_key] + edge_attr_list].copy()

    # check whether duplicated edge ids by shared_node_key have the same attribute values.
    # If not, give warning, and keep the first value. (which can be improved later)
    check_edgeid_attr_unique = (
        graph_edge_df_sub.groupby(shared_node_key)[edge_attr_list].nunique() == 1
    )

    # check any False in check_edgeid_attr_unique's columns, if so, get the column names
    bool_edgeid_attr_unique = (check_edgeid_attr_unique.isin([False])).any()  # type: ignore

    non_unique_indices = [
        i for i, value in enumerate(bool_edgeid_attr_unique.to_list()) if value
    ]

    # if edge ids with duplicated shared_node_key have more than 1 unique values
    # for attributes of interest
    non_unique_egde_attr = bool_edgeid_attr_unique.index[non_unique_indices].to_list()

    if len(non_unique_egde_attr) == 0:
        logger.info("Per duplicated edge ids, attributes have only 1 unique value.")
    else:
        logger.warning(
            f"Per duplicated edge ids, attributes: {non_unique_egde_attr} "
            "contain more than 1 unique values"
        )

    # remove duplicated edge attribute values
    graph_edge_df_sub_no_duplicate = graph_edge_df_sub.drop_duplicates(
        subset=shared_node_key, keep="first"
    )

    # rename shared_node_key to vertex key 'name'
    # as in net_create.create_napistu_graph(), vertex_name_attr is set to 'name'
    graph_edge_df_sub_no_duplicate = graph_edge_df_sub_no_duplicate.rename(
        columns={shared_node_key: "name"},
    )

    # merge edge attributes in graph_edge_df_sub_no_duplicate to vertex_df,
    # by shared key 'name'
    graph_vertex_df_w_edge_attr = pd.merge(
        graph_vertex_df,
        graph_edge_df_sub_no_duplicate,
        on="name",
        how="outer",
    )

    logger.info(f"Adding {edge_attr_list} to vertex attributes")
    # Warning for NaN values in vertex attributes:
    if graph_vertex_df_w_edge_attr.isnull().values.any():
        logger.warning(
            "NaN values are present in the newly added vertex attributes. "
            "Please assign proper values to those vertex attributes."
        )

    # assign the edge_attrs from edge_attr_list to napistu_graph's vertices:
    # keep the same edge attribute names:
    for col_name in edge_attr_list:
        napistu_graph.vs[col_name] = graph_vertex_df_w_edge_attr[col_name]

    return napistu_graph


def _create_source_weights(
    edges_df: pd.DataFrame,
    source_wt_var: str = "source_wt",
    source_vars_dict: dict = SOURCE_VARS_DICT,
    source_wt_default: int = 1,
) -> pd.DataFrame:
    """
    Create weights based on an edge's source.

    Parameters
    ----------
    edges_df : pd.DataFrame
        The edges dataframe to add the source weights to.
    source_wt_var : str, optional
        The name of the column to store the source weights. Default is "source_wt".
    source_vars_dict : dict, optional
        Dictionary with keys indicating edge attributes and values indicating the weight to assign to that attribute. Default is SOURCE_VARS_DICT.
    source_wt_default : int, optional
        The default weight to assign to an edge if no other weight attribute is found. Default is 1.

    Returns
    -------
    pd.DataFrame
        The edges dataframe with the source weights added.
    """

    logger.warning(
        "_create_source_weights should be reimplemented once https://github.com/calico/pathadex-data/issues/95 "
        "is fixed. The current implementation is quite limited."
    )

    # currently, we will look for values of source_indicator_var which are non NA and set them to
    # source_indicator_match_score and setting entries which are NA as source_indicator_nonmatch_score.
    #
    # this is a simple way of flagging string vs. non-string scores

    included_weight_vars = set(source_vars_dict.keys()).intersection(
        set(edges_df.columns)
    )
    if len(included_weight_vars) == 0:
        logger.warning(
            f"No edge attributes were found which match those in source_vars_dict: {', '.join(source_vars_dict.keys())}"
        )
        edges_df[source_wt_var] = source_wt_default
        return edges_df

    edges_df_source_wts = edges_df[list(included_weight_vars)].copy()
    for wt in list(included_weight_vars):
        edges_df_source_wts[wt] = [
            source_wt_default if x is True else source_vars_dict[wt]
            for x in edges_df[wt].isna()
        ]

    source_wt_edges_df = edges_df.join(
        edges_df_source_wts.max(axis=1).rename(source_wt_var)
    )

    return source_wt_edges_df


def _augment_network_nodes(
    network_nodes: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_graph_attrs: dict = dict(),
    custom_transformations: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add species-level attributes, expand network_nodes with s_id and c_id and then map to species-level attributes by s_id.

    This function merges species-level attributes from sbml_dfs into the provided network_nodes DataFrame,
    using the mapping in species_graph_attrs. Optionally, custom transformation functions can be provided
    to transform the attributes as they are added.

    Parameters
    ----------
    network_nodes : pd.DataFrame
        DataFrame of network nodes. Must include columns 'name', 'node_name', and 'node_type'.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing species data.
    species_graph_attrs : dict, optional
        Dictionary specifying which attributes to pull from species_data and how to transform them.
        The structure should be {attribute_name: {"table": ..., "variable": ..., "trans": ...}}.
    custom_transformations : dict, optional
        Dictionary mapping transformation names to functions. If provided, these will be checked
        before built-in transformations. Example: {"square": lambda x: x**2}

    Returns
    -------
    pd.DataFrame
        The input network_nodes DataFrame with additional columns for each extracted and transformed attribute.

    Raises
    ------
    ValueError
        If required attributes are missing from network_nodes.
    """

    NETWORK_NODE_VARS = {
        NAPISTU_GRAPH_VERTICES.NAME,
        NAPISTU_GRAPH_VERTICES.NODE_NAME,
        NAPISTU_GRAPH_VERTICES.NODE_TYPE,
    }

    missing_required_network_nodes_attrs = NETWORK_NODE_VARS.difference(
        set(network_nodes.columns.tolist())
    )
    print(
        f"DEBUG: _augment_network_nodes - missing_required_network_nodes_attrs: {missing_required_network_nodes_attrs}"
    )

    if len(missing_required_network_nodes_attrs) > 0:
        raise ValueError(
            f"{len(missing_required_network_nodes_attrs)} required attributes were missing "
            "from network_nodes: "
            f"{', '.join(missing_required_network_nodes_attrs)}"
        )

    # include matching s_ids and c_ids of sc_ids
    network_nodes_sid = utils._merge_and_log_overwrites(
        network_nodes,
        sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID, SBML_DFS.C_ID]],
        "network nodes",
        left_on=NAPISTU_GRAPH_VERTICES.NAME,
        right_index=True,
        how="left",
    )

    # assign species_data related attributes to s_id
    species_graph_data = ng_utils.pluck_entity_data(
        sbml_dfs,
        species_graph_attrs,
        "species",
        custom_transformations=custom_transformations,
    )

    if species_graph_data is not None:
        # add species_graph_data to the network_nodes df, based on s_id
        network_nodes_wdata = utils._merge_and_log_overwrites(
            network_nodes_sid,
            species_graph_data,
            "species graph data",
            left_on=SBML_DFS.S_ID,
            right_index=True,
            how="left",
        )
    else:
        network_nodes_wdata = network_nodes_sid

    # Note: multiple sc_ids with the same s_id will be assign with the same species_graph_data

    network_nodes_wdata = network_nodes_wdata.fillna(int(0)).drop(
        columns=["s_id", "c_id"]
    )

    return network_nodes_wdata


def _augment_network_edges(
    network_edges: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_graph_attrs: dict = dict(),
    custom_transformations: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add reversibility and other metadata from reactions.

    Parameters
    ----------
    network_edges : pd.DataFrame
        DataFrame of network edges.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing reaction data.
    reaction_graph_attrs : dict, optional
        Dictionary of reaction attributes to add.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.

    Returns
    -------
    pd.DataFrame
        DataFrame of network edges with additional metadata.

    Raises
    ------
    ValueError
        If required attributes are missing from network_edges.
    """

    missing_required_network_edges_attrs = (
        NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS.difference(
            set(network_edges.columns.tolist())
        )
    )
    if len(missing_required_network_edges_attrs) > 0:
        raise ValueError(
            f"{len(missing_required_network_edges_attrs)} required attributes were missing "
            "from network_edges: "
            f"{', '.join(missing_required_network_edges_attrs)}"
        )

    network_edges = (
        network_edges[list(NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS)]
        # add reaction-level attributes
        .merge(
            sbml_dfs.reactions[SBML_DFS.R_ISREVERSIBLE],
            left_on=SBML_DFS.R_ID,
            right_index=True,
        )
    )

    # add other attributes based on reactions data
    reaction_graph_data = ng_utils.pluck_entity_data(
        sbml_dfs,
        reaction_graph_attrs,
        SBML_DFS.REACTIONS,
        custom_transformations=custom_transformations,
    )
    if reaction_graph_data is not None:
        network_edges = network_edges.merge(
            reaction_graph_data, left_on=SBML_DFS.R_ID, right_index=True, how="left"
        )

    return network_edges


def _reverse_network_edges(augmented_network_edges: pd.DataFrame) -> pd.DataFrame:
    """
    Flip reversible reactions to derive the reverse reaction.

    Parameters
    ----------
    augmented_network_edges : pd.DataFrame
        DataFrame of network edges with metadata.

    Returns
    -------
    pd.DataFrame
        DataFrame with reversed edges for reversible reactions.

    Raises
    ------
    ValueError
        If required variables are missing or if the transformation fails.
    """

    # validate inputs
    required_vars = {NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO}
    missing_required_vars = required_vars.difference(
        set(augmented_network_edges.columns.tolist())
    )

    if len(missing_required_vars) > 0:
        raise ValueError(
            "augmented_network_edges is missing required variables: "
            f"{', '.join(missing_required_vars)}"
        )

    # Check if direction already exists
    if NAPISTU_GRAPH_EDGES.DIRECTION in augmented_network_edges.columns:
        logger.warning(
            f"{NAPISTU_GRAPH_EDGES.DIRECTION} field already exists in augmented_network_edges. "
            "This is unexpected and may indicate an issue in the graph creation process."
        )

    # select all edges derived from reversible reactions
    reversible_reaction_edges = augmented_network_edges[
        augmented_network_edges[NAPISTU_GRAPH_EDGES.R_ISREVERSIBLE]
    ]

    r_reaction_edges = (
        # ignore edges which start in a regulator or catalyst; even for a reversible reaction it
        # doesn't make sense for a regulator to be impacted by a target
        reversible_reaction_edges[
            ~reversible_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM].isin(
                [
                    MINI_SBO_FROM_NAME[x]
                    for x in SBO_MODIFIER_NAMES.union({SBOTERM_NAMES.CATALYST})
                ]
            )
        ]
        # flip parent and child attributes
        .rename(
            {
                NAPISTU_GRAPH_EDGES.FROM: NAPISTU_GRAPH_EDGES.TO,
                NAPISTU_GRAPH_EDGES.TO: NAPISTU_GRAPH_EDGES.FROM,
                NAPISTU_GRAPH_EDGES.SC_CHILDREN: NAPISTU_GRAPH_EDGES.SC_PARENTS,
                NAPISTU_GRAPH_EDGES.SC_PARENTS: NAPISTU_GRAPH_EDGES.SC_CHILDREN,
            },
            axis=1,
        )
    )

    # switch substrates and products
    r_reaction_edges[NAPISTU_GRAPH_EDGES.STOICHIOMETRY] = r_reaction_edges[
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY
    ].apply(
        # the ifelse statement prevents 0 being converted to -0 ...
        lambda x: -1 * x if x != 0 else 0
    )

    transformed_r_reaction_edges = pd.concat(
        [
            (
                r_reaction_edges[
                    r_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM]
                    == MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]
                ].assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT])
            ),
            (
                r_reaction_edges[
                    r_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM]
                    == MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]
                ].assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT])
            ),
            r_reaction_edges[
                ~r_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM].isin(
                    [
                        MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT],
                        MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT],
                    ]
                )
            ],
        ]
    )

    if transformed_r_reaction_edges.shape[0] != r_reaction_edges.shape[0]:
        raise ValueError(
            "transformed_r_reaction_edges and r_reaction_edges must have the same number of rows"
        )

    return transformed_r_reaction_edges.assign(
        **{NAPISTU_GRAPH_EDGES.DIRECTION: NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE}
    )
