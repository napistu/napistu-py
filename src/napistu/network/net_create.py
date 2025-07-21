from __future__ import annotations

import copy
import logging
import random
from typing import Optional

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel

from napistu import sbml_dfs_core
from napistu import utils
from napistu.network import net_create_utils
from napistu.network.ng_core import NapistuGraph


from napistu.constants import (
    MINI_SBO_FROM_NAME,
    SBO_MODIFIER_NAMES,
    SBOTERM_NAMES,
    SBML_DFS,
    ENTITIES_W_DATA,
)

from napistu.network.constants import (
    NAPISTU_GRAPH_VERTICES,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_EDGE_DIRECTIONS,
    NAPISTU_GRAPH_NODE_TYPES,
    GRAPH_WIRING_APPROACHES,
    NAPISTU_WEIGHTING_STRATEGIES,
    VALID_GRAPH_WIRING_APPROACHES,
    VALID_WEIGHTING_STRATEGIES,
    DEFAULT_WT_TRANS,
    DEFINED_WEIGHT_TRANSFORMATION,
    SCORE_CALIBRATION_POINTS_DICT,
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
        _validate_entity_attrs(
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
            - 'calibrated': transform edges with a quantitative score based on reaction_attrs and combine them with topology scores to generate a consensus.
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

    logger.info(
        "Adding additional attributes to edges, e.g., # of children and parents."
    )

    # add compartmentalized species summaries to weight edges
    cspecies_features = sbml_dfs.get_cspecies_features()

    # calculate undirected and directed degrees (i.e., # of parents and children)
    # based on a network's edgelist. this used when the network representation is
    # not the bipartite network which can be trivially obtained from the pathway
    # specification
    unique_edges = (
        all_reaction_edges_df.groupby(
            [NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO]
        )
        .first()
        .reset_index()
    )

    # children
    n_children = (
        unique_edges[NAPISTU_GRAPH_EDGES.FROM]
        .value_counts()
        # rename values to the child name
        .to_frame(name=NAPISTU_GRAPH_EDGES.SC_CHILDREN)
        .reset_index()
        .rename(
            {
                NAPISTU_GRAPH_EDGES.FROM: SBML_DFS.SC_ID,
            },
            axis=1,
        )
    )

    # parents
    n_parents = (
        unique_edges[NAPISTU_GRAPH_EDGES.TO]
        .value_counts()
        # rename values to the parent name
        .to_frame(name=NAPISTU_GRAPH_EDGES.SC_PARENTS)
        .reset_index()
        .rename(
            {
                NAPISTU_GRAPH_EDGES.TO: SBML_DFS.SC_ID,
            },
            axis=1,
        )
    )

    graph_degree_by_edgelist = n_children.merge(n_parents, how="outer").fillna(int(0))

    graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_DEGREE] = (
        graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_CHILDREN]
        + graph_degree_by_edgelist[NAPISTU_GRAPH_EDGES.SC_PARENTS]
    )
    graph_degree_by_edgelist = (
        graph_degree_by_edgelist[
            ~graph_degree_by_edgelist[SBML_DFS.SC_ID].str.contains("R[0-9]{8}")
        ]
        .set_index(SBML_DFS.SC_ID)
        .sort_index()
    )

    cspecies_features = (
        cspecies_features.drop(
            [
                NAPISTU_GRAPH_EDGES.SC_DEGREE,
                NAPISTU_GRAPH_EDGES.SC_CHILDREN,
                NAPISTU_GRAPH_EDGES.SC_PARENTS,
            ],
            axis=1,
        )
        .join(graph_degree_by_edgelist)
        .fillna(int(0))
    )

    is_from_reaction = all_reaction_edges_df[NAPISTU_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index.tolist()
    )
    is_from_reaction = all_reaction_edges_df[NAPISTU_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index
    )
    # add substrate weight whenever "from" edge is a molecule
    # and product weight when the "from" edge is a reaction
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


def pluck_entity_data(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    graph_attrs: dict[str, dict],
    data_type: str,
    custom_transformations: Optional[dict[str, callable]] = None,
) -> pd.DataFrame | None:
    """
    Pluck Entity Attributes from an sbml_dfs based on a set of tables and variables to look for.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A mechanistic model.
    graph_attrs : dict
        A dictionary of species/reaction attributes to pull out. If the requested
        data_type ("species" or "reactions") is not present as a key, or if the value
        is an empty dict, this function will return None (no error).
    data_type : str
        "species" or "reactions" to pull out species_data or reactions_data.
    custom_transformations : dict[str, callable], optional
        A dictionary mapping transformation names to functions. If provided, these
        will be checked before built-in transformations. Example:
            custom_transformations = {"square": lambda x: x**2}

    Returns
    -------
    pd.DataFrame or None
        A table where all extracted attributes are merged based on a common index or None
        if no attributes were extracted. If the requested data_type is not present in
        graph_attrs, or if the attribute dict is empty, returns None. This is intended
        to allow optional annotation blocks.

    Raises
    ------
    ValueError
        If data_type is not valid or if requested tables/variables are missing.
    """

    if data_type not in ENTITIES_W_DATA:
        raise ValueError(
            f'"data_type" was {data_type} and must be in {", ".join(ENTITIES_W_DATA)}'
        )

    if data_type not in graph_attrs.keys():
        logger.info(
            f'No {data_type} annotations provided in "graph_attrs"; returning None'
        )
        return None

    entity_attrs = graph_attrs[data_type]
    # validating dict
    _validate_entity_attrs(entity_attrs, custom_transformations=custom_transformations)

    if len(entity_attrs) == 0:
        logger.info(
            f'No attributes defined for "{data_type}" in graph_attrs; returning None'
        )
        return None

    data_type_attr = data_type + "_data"
    entity_data_tbls = getattr(sbml_dfs, data_type_attr)

    data_list = list()
    for k, v in entity_attrs.items():
        # v["table"] is always present if entity_attrs is non-empty and validated
        if v["table"] not in entity_data_tbls.keys():
            raise ValueError(
                f"{v['table']} was defined as a table in \"graph_attrs\" but "
                f'it is not present in the "{data_type_attr}" of the sbml_dfs'
            )

        if v["variable"] not in entity_data_tbls[v["table"]].columns.tolist():
            raise ValueError(
                f"{v['variable']} was defined as a variable in \"graph_attrs\" but "
                f"it is not present in the {v['table']} of the \"{data_type_attr}\" of "
                "the sbml_dfs"
            )

        entity_series = entity_data_tbls[v["table"]][v["variable"]].rename(k)
        trans_name = v.get("trans", DEFAULT_WT_TRANS)
        # Look up transformation
        if custom_transformations and trans_name in custom_transformations:
            trans_fxn = custom_transformations[trans_name]
        elif trans_name in DEFINED_WEIGHT_TRANSFORMATION:
            trans_fxn = globals()[DEFINED_WEIGHT_TRANSFORMATION[trans_name]]
        else:
            # This should never be hit if _validate_entity_attrs is called correctly.
            raise ValueError(
                f"Transformation '{trans_name}' not found in custom_transformations or DEFINED_WEIGHT_TRANSFORMATION."
            )
        entity_series = entity_series.apply(trans_fxn)
        data_list.append(entity_series)

    if len(data_list) == 0:
        return None

    return pd.concat(data_list, axis=1)


def apply_weight_transformations(
    edges_df: pd.DataFrame, reaction_attrs: dict, custom_transformations: dict = None
):
    """
    Apply Weight Transformations to edge attributes.

    Parameters
    ----------
    edges_df : pd.DataFrame
        A table of edges and their attributes extracted from a cpr_graph.
    reaction_attrs : dict
        A dictionary of attributes identifying weighting attributes within
        an sbml_df's reaction_data, how they will be named in edges_df (the keys),
        and how they should be transformed (the "trans" aliases).
    custom_transformations : dict, optional
        A dictionary mapping transformation names to functions. If provided, these
        will be checked before built-in transformations.

    Returns
    -------
    pd.DataFrame
        edges_df with weight variables transformed.

    Raises
    ------
    ValueError
        If a weighting variable is missing or transformation is not found.
    """

    _validate_entity_attrs(
        reaction_attrs, custom_transformations=custom_transformations
    )

    transformed_edges_df = copy.deepcopy(edges_df)
    for k, v in reaction_attrs.items():
        if k not in transformed_edges_df.columns:
            raise ValueError(f"A weighting variable {k} was missing from edges_df")

        trans_name = v["trans"]
        # Look up transformation
        if custom_transformations and trans_name in custom_transformations:
            trans_fxn = custom_transformations[trans_name]
        elif trans_name in DEFINED_WEIGHT_TRANSFORMATION:
            trans_fxn = globals()[DEFINED_WEIGHT_TRANSFORMATION[trans_name]]
        else:
            # This should never be hit if _validate_entity_attrs is called correctly.
            raise ValueError(
                f"Transformation '{trans_name}' not found in custom_transformations or DEFINED_WEIGHT_TRANSFORMATION."
            )

        transformed_edges_df[k] = transformed_edges_df[k].apply(trans_fxn)

    return transformed_edges_df


def summarize_weight_calibration(
    napistu_graph: NapistuGraph, reaction_attrs: dict
) -> None:
    """
    Summarize Weight Calibration for a network with multiple sources for edge weights.

    Parameters
    ----------
    napistu_graph : NapistuGraph
        A graph where edge weights have already been calibrated.
    reaction_attrs : dict
        A dictionary summarizing the types of weights that exist and how they are transformed for calibration.

    Returns
    -------
    None
    """

    score_calibration_df = pd.DataFrame(SCORE_CALIBRATION_POINTS_DICT)
    score_calibration_df_calibrated = apply_weight_transformations(
        score_calibration_df, reaction_attrs
    )

    calibrated_edges = napistu_graph.get_edge_dataframe()

    _summarize_weight_calibration_table(
        calibrated_edges, score_calibration_df, score_calibration_df_calibrated
    )

    _summarize_weight_calibration_plots(
        calibrated_edges, score_calibration_df_calibrated
    )

    return None


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
            - 'calibrated': transform edges with a quantitative score based on reaction_attrs and combine them with topology scores to generate a consensus.

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

    _validate_entity_attrs(reaction_attrs)

    if weighting_strategy not in VALID_WEIGHTING_STRATEGIES:
        raise ValueError(
            f"weighting_strategy was {weighting_strategy} and must be one of: "
            f"{', '.join(VALID_WEIGHTING_STRATEGIES)}"
        )

    # count parents and children and create weights based on them
    topology_weighted_graph = _create_topology_weights(napistu_graph_updated)

    if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.TOPOLOGY:
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

    if weighting_strategy == NAPISTU_WEIGHTING_STRATEGIES.CALIBRATED:
        return _add_graph_weights_calibration(topology_weighted_graph, reaction_attrs)

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
    cspecies_features = sbml_dfs.get_cspecies_features()
    network_edges = network_edges.merge(
        cspecies_features, left_on=NAPISTU_GRAPH_NODE_TYPES.SPECIES, right_index=True
    )

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

    calibrated_edges = apply_weight_transformations(edges_df, reaction_attrs)
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


def _add_graph_weights_calibration(
    napistu_graph: NapistuGraph, reaction_attrs: dict
) -> NapistuGraph:
    """
    Weight a NapistuGraph using a calibrated strategy which aims to roughly align qualitatively similar weights from different sources.

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

    calibrated_edges = apply_weight_transformations(edges_df, reaction_attrs)

    score_vars = list(reaction_attrs.keys())
    score_vars.append("topo_weights")

    logger.info(f"Creating calibrated scores based on {', '.join(score_vars)}")
    napistu_graph.es[NAPISTU_GRAPH_EDGES.WEIGHT] = calibrated_edges[score_vars].min(
        axis=1
    )

    if napistu_graph.is_directed():
        score_vars = list(reaction_attrs.keys())
        score_vars.append("upstream_topo_weights")
        napistu_graph.es[NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT] = calibrated_edges[
            score_vars
        ].min(axis=1)

    # add other attributes and update transformed attributes
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


def _summarize_weight_calibration_table(
    calibrated_edges: pd.DataFrame,
    score_calibration_df: pd.DataFrame,
    score_calibration_df_calibrated: pd.DataFrame,
):
    """
    Create a table comparing edge weights from multiple sources.

    Parameters
    ----------
    calibrated_edges : pd.DataFrame
        DataFrame of calibrated edge weights.
    score_calibration_df : pd.DataFrame
        DataFrame of raw calibration points.
    score_calibration_df_calibrated : pd.DataFrame
        DataFrame of calibrated calibration points.

    Returns
    -------
    pd.DataFrame
        Styled DataFrame summarizing calibration points and quantiles.
    """

    # generate a table summarizing different scoring measures
    #
    # a set of calibration points defined in DEFINED_WEIGHT_TRANSFORMATION which map
    # onto what we might consider strong versus dubious edges are compared to the
    # observed scores to see whether these calibration points generally map onto
    # the expected quantiles of the score distribution.
    #
    # different scores are also compared to see whether there calibrations are generally
    # aligned. that is to say a strong weight based on one scoring measure would receive
    # a similar quantitative score to a strong score for another measure.

    score_calibration_long_raw = (
        score_calibration_df.reset_index()
        .rename({"index": "edge_strength"}, axis=1)
        .melt(
            id_vars="edge_strength", var_name="weight_measure", value_name="raw_weight"
        )
    )

    score_calibration_long_calibrated = (
        score_calibration_df_calibrated.reset_index()
        .rename({"index": "edge_strength"}, axis=1)
        .melt(
            id_vars="edge_strength",
            var_name="weight_measure",
            value_name="trans_weight",
        )
    )

    score_calibration_table_long = score_calibration_long_raw.merge(
        score_calibration_long_calibrated
    )

    # compare calibration points to the quantiles of the observed score distributions
    score_quantiles = list()
    for ind, row in score_calibration_table_long.iterrows():
        score_quantiles.append(
            1
            - np.mean(
                calibrated_edges[row["weight_measure"]].dropna() >= row["trans_weight"]
            )
        )
    score_calibration_table_long["quantile_of_score_dist"] = score_quantiles

    return utils.style_df(score_calibration_table_long, headers="keys")


def _summarize_weight_calibration_plots(
    calibrated_edges: pd.DataFrame, score_calibration_df_calibrated: pd.DataFrame
) -> None:
    """
    Create plots summarizing the relationships between different scoring measures.

    Parameters
    ----------
    calibrated_edges : pd.DataFrame
        DataFrame of calibrated edge weights.
    score_calibration_df_calibrated : pd.DataFrame
        DataFrame of calibrated calibration points.

    Returns
    -------
    None
    """

    # set up a 2 x 1 plot
    f, (ax1, ax2) = plt.subplots(1, 2)

    calibrated_edges[["topo_weights", "string_wt"]].plot(
        kind="hist", bins=50, alpha=0.5, ax=ax1
    )
    ax1.set_title("Distribution of scores\npost calibration")

    score_calibration_df_calibrated.plot("weights", "string_wt", kind="scatter", ax=ax2)

    for k, v in score_calibration_df_calibrated.iterrows():
        ax2.annotate(k, v)
    ax2.axline((0, 0), slope=1.0, color="C0", label="by slope")
    ax2.set_title("Comparing STRING and\nTopology calibration points")

    return None


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


def _wt_transformation_identity(x):
    """
    Identity transformation for weights.

    Parameters
    ----------
    x : any
        Input value.

    Returns
    -------
    any
        The input value unchanged.
    """
    return x


def _wt_transformation_string(x):
    """
    Map STRING scores to a similar scale as topology weights.

    Parameters
    ----------
    x : float
        STRING score.

    Returns
    -------
    float
        Transformed STRING score.
    """
    return 250000 / np.power(x, 1.7)


def _wt_transformation_string_inv(x):
    """
    Map STRING scores so they work with source weights.

    Parameters
    ----------
    x : float
        STRING score.

    Returns
    -------
    float
        Inverse transformed STRING score.
    """
    # string scores are bounded on [0, 1000]
    # and score/1000 is roughly a probability that
    # there is a real interaction (physical, genetic, ...)
    # reported string scores are currently on [150, 1000]
    # so this transformation will map these onto {6.67, 1}
    return 1 / (x / 1000)


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
    REQUIRED_NETWORK_NODE_ATTRS = {
        "name",
        "node_name",
        "node_type",
    }

    missing_required_network_nodes_attrs = REQUIRED_NETWORK_NODE_ATTRS.difference(
        set(network_nodes.columns.tolist())
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
        sbml_dfs.compartmentalized_species[["s_id", "c_id"]],
        "network nodes",
        left_on="name",
        right_index=True,
        how="left",
    )

    # assign species_data related attributes to s_id
    species_graph_data = pluck_entity_data(
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
            left_on="s_id",
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
    REQUIRED_NETWORK_EDGE_ATTRS = {
        "from",
        "to",
        "stoichiometry",
        "sbo_term",
        "sc_degree",
        "sc_children",
        "sc_parents",
        "species_type",
        "r_id",
    }

    missing_required_network_edges_attrs = REQUIRED_NETWORK_EDGE_ATTRS.difference(
        set(network_edges.columns.tolist())
    )
    if len(missing_required_network_edges_attrs) > 0:
        raise ValueError(
            f"{len(missing_required_network_edges_attrs)} required attributes were missing "
            "from network_edges: "
            f"{', '.join(missing_required_network_edges_attrs)}"
        )

    network_edges = (
        network_edges[list(REQUIRED_NETWORK_EDGE_ATTRS)]
        # add reaction-level attributes
        .merge(
            sbml_dfs.reactions[SBML_DFS.R_ISREVERSIBLE],
            left_on=SBML_DFS.R_ID,
            right_index=True,
        )
    )

    # add other attributes based on reactions data
    reaction_graph_data = pluck_entity_data(
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


def _create_topology_weights(
    napistu_graph: ig.Graph,
    base_score: float = 2,
    protein_multiplier: int = 1,
    metabolite_multiplier: int = 3,
    unknown_multiplier: int = 10,
    scale_multiplier_by_meandegree: bool = True,
) -> ig.Graph:
    """
    Create Topology Weights for a network based on its topology.

    Edges downstream of nodes with many connections receive a higher weight suggesting that any one
    of them is less likely to be regulatory. This is a simple and clearly flawed heuristic which can be
    combined with more principled weighting schemes.

    Parameters
    ----------
    napistu_graph : ig.Graph
        A graph containing connections between molecules, proteins, and reactions.
    base_score : float, optional
        Offset which will be added to all weights. Default is 2.
    protein_multiplier : int, optional
        Multiplier for non-metabolite species. Default is 1.
    metabolite_multiplier : int, optional
        Multiplier for metabolites. Default is 3.
    unknown_multiplier : int, optional
        Multiplier for species without any identifier. Default is 10.
    scale_multiplier_by_meandegree : bool, optional
        If True, multipliers will be rescaled by the average number of connections a node has. Default is True.

    Returns
    -------
    ig.Graph
        Graph with added topology weights.

    Raises
    ------
    ValueError
        If required attributes are missing or if parameters are invalid.
    """

    # check for required attribute before proceeding

    required_attrs = {
        NAPISTU_GRAPH_EDGES.SC_DEGREE,
        NAPISTU_GRAPH_EDGES.SC_CHILDREN,
        NAPISTU_GRAPH_EDGES.SC_PARENTS,
        NAPISTU_GRAPH_EDGES.SPECIES_TYPE,
    }

    missing_required_attrs = required_attrs.difference(
        set(napistu_graph.es.attributes())
    )
    if len(missing_required_attrs) != 0:
        raise ValueError(
            f"model is missing {len(missing_required_attrs)} required attributes: {', '.join(missing_required_attrs)}"
        )

    if base_score < 0:
        raise ValueError(f"base_score was {base_score} and must be non-negative")
    if protein_multiplier > unknown_multiplier:
        raise ValueError(
            f"protein_multiplier was {protein_multiplier} and unknown_multiplier "
            f"was {unknown_multiplier}. unknown_multiplier must be greater than "
            "protein_multiplier"
        )
    if metabolite_multiplier > unknown_multiplier:
        raise ValueError(
            f"protein_multiplier was {metabolite_multiplier} and unknown_multiplier "
            f"was {unknown_multiplier}. unknown_multiplier must be greater than "
            "protein_multiplier"
        )

    # create a new weight variable

    weight_table = pd.DataFrame(
        {
            NAPISTU_GRAPH_EDGES.SC_DEGREE: napistu_graph.es[
                NAPISTU_GRAPH_EDGES.SC_DEGREE
            ],
            NAPISTU_GRAPH_EDGES.SC_CHILDREN: napistu_graph.es[
                NAPISTU_GRAPH_EDGES.SC_CHILDREN
            ],
            NAPISTU_GRAPH_EDGES.SC_PARENTS: napistu_graph.es[
                NAPISTU_GRAPH_EDGES.SC_PARENTS
            ],
            NAPISTU_GRAPH_EDGES.SPECIES_TYPE: napistu_graph.es[
                NAPISTU_GRAPH_EDGES.SPECIES_TYPE
            ],
        }
    )

    lookup_multiplier_dict = {
        "protein": protein_multiplier,
        "metabolite": metabolite_multiplier,
        "unknown": unknown_multiplier,
    }
    weight_table["multiplier"] = weight_table["species_type"].map(
        lookup_multiplier_dict
    )

    # calculate mean degree
    # since topology weights will differ based on the structure of the network
    # and it would be nice to have a consistent notion of edge weights and path weights
    # for interpretability and filtering, we can rescale topology weights by the
    # average degree of nodes
    if scale_multiplier_by_meandegree:
        mean_degree = len(napistu_graph.es) / len(napistu_graph.vs)
        if not napistu_graph.is_directed():
            # for a directed network in- and out-degree are separately treated while
            # an undirected network's degree will be the sum of these two measures.
            mean_degree = mean_degree * 2

        weight_table["multiplier"] = weight_table["multiplier"] / mean_degree

    if napistu_graph.is_directed():
        weight_table["connection_weight"] = weight_table[
            NAPISTU_GRAPH_EDGES.SC_CHILDREN
        ]
    else:
        weight_table["connection_weight"] = weight_table[NAPISTU_GRAPH_EDGES.SC_DEGREE]

    # weight traveling through a species based on
    # - a constant
    # - how plausibly that species type mediates a change
    # - the number of connections that the node can bridge to
    weight_table["topo_weights"] = [
        base_score + (x * y)
        for x, y in zip(weight_table["multiplier"], weight_table["connection_weight"])
    ]
    napistu_graph.es["topo_weights"] = weight_table["topo_weights"]

    # if directed and we want to use travel upstream define a corresponding weighting scheme
    if napistu_graph.is_directed():
        weight_table["upstream_topo_weights"] = [
            base_score + (x * y)
            for x, y in zip(weight_table["multiplier"], weight_table["sc_parents"])
        ]
        napistu_graph.es["upstream_topo_weights"] = weight_table[
            "upstream_topo_weights"
        ]

    return napistu_graph


def _validate_entity_attrs(
    entity_attrs: dict,
    validate_transformations: bool = True,
    custom_transformations: Optional[dict] = None,
) -> None:
    """
    Validate that graph attributes are a valid format.

    Parameters
    ----------
    entity_attrs : dict
        Dictionary of entity attributes to validate.
    validate_transformations : bool, optional
        Whether to validate transformation names, by default True.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions, by default None. Keys are transformation names, values are transformation functions.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If entity_attrs is not a dictionary.
    ValueError
        If a transformation is not found in DEFINED_WEIGHT_TRANSFORMATION or custom_transformations.
    """
    assert isinstance(entity_attrs, dict), "entity_attrs must be a dictionary"

    for k, v in entity_attrs.items():
        # check structure against pydantic config
        validated_attrs = _EntityAttrValidator(**v).model_dump()

        if validate_transformations:
            trans_name = validated_attrs.get("trans", DEFAULT_WT_TRANS)
            valid_trans = set(DEFINED_WEIGHT_TRANSFORMATION.keys())
            if custom_transformations:
                valid_trans = valid_trans.union(set(custom_transformations.keys()))
            if trans_name not in valid_trans:
                raise ValueError(
                    f"transformation '{trans_name}' was not defined as an alias in "
                    "DEFINED_WEIGHT_TRANSFORMATION or custom_transformations. The defined transformations "
                    f"are {', '.join(sorted(valid_trans))}"
                )

    return None


class _EntityAttrValidator(BaseModel):
    table: str
    variable: str
    trans: Optional[str] = DEFAULT_WT_TRANS
