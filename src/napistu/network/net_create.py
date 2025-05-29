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

from napistu.constants import DEFAULT_WT_TRANS
from napistu.constants import DEFINED_WEIGHT_TRANSFORMATION
from napistu.constants import MINI_SBO_FROM_NAME
from napistu.constants import MINI_SBO_TO_NAME
from napistu.constants import SBML_DFS
from napistu.constants import SBO_MODIFIER_NAMES
from napistu.constants import SCORE_CALIBRATION_POINTS_DICT
from napistu.constants import ENTITIES_W_DATA
from napistu.constants import SOURCE_VARS_DICT

from napistu.network.constants import CPR_GRAPH_NODES
from napistu.network.constants import CPR_GRAPH_EDGES
from napistu.network.constants import CPR_GRAPH_EDGE_DIRECTIONS
from napistu.network.constants import CPR_GRAPH_REQUIRED_EDGE_VARS
from napistu.network.constants import CPR_GRAPH_NODE_TYPES
from napistu.network.constants import CPR_GRAPH_TYPES
from napistu.network.constants import CPR_WEIGHTING_STRATEGIES
from napistu.network.constants import SBOTERM_NAMES
from napistu.network.constants import REGULATORY_GRAPH_HIERARCHY
from napistu.network.constants import SURROGATE_GRAPH_HIERARCHY
from napistu.network.constants import VALID_CPR_GRAPH_TYPES
from napistu.network.constants import VALID_WEIGHTING_STRATEGIES

logger = logging.getLogger(__name__)


def create_cpr_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_graph_attrs: Optional[dict] = None,
    directed: bool = True,
    edge_reversed: bool = False,
    graph_type: str = CPR_GRAPH_TYPES.BIPARTITE,
    verbose: bool = False,
    custom_transformations: Optional[dict] = None,
) -> ig.Graph:
    """
    Create CPR Graph

    Create an igraph network from a mechanistic network using one of a set of graph_types.

    Parameters
    ----------
    sbml_dfs : SBML_dfs
        A model formed by aggregating pathways
    reaction_graph_attrs: dict
        Dictionary containing attributes to pull out of reaction_data and
        a weighting scheme for the graph
    directed : bool
        Should a directed (True) or undirected graph be made (False)
    edge_reversed : bool
        Should the directions of edges be reversed or not (False)
    graph_type : str
        Type of graph to create, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - reguatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products
            - surrogate: non-enzymatic modifiers -> substrates -> enzymes -> reaction -> products.
              In this representation enzymes are effective standing in for their reaction (eventhough the enzyme is
              not modified by a substrate per-se).
    verbose : bool
        Extra reporting
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.

    Returns:
    ----------
    An Igraph network
    """

    if reaction_graph_attrs is None:
        reaction_graph_attrs = {}

    if not isinstance(sbml_dfs, sbml_dfs_core.SBML_dfs):
        raise TypeError(
            f"sbml_dfs must be a sbml_dfs_core.SBML_dfs, but was {type(sbml_dfs)}"
        )

    if not isinstance(reaction_graph_attrs, dict):
        raise TypeError(
            f"reaction_graph_attrs must be a dict, but was {type(reaction_graph_attrs)}"
        )

    if not isinstance(directed, bool):
        raise TypeError(f"directed must be a bool, but was {type(directed)}")

    if not isinstance(edge_reversed, bool):
        raise TypeError(f"edge_reverse must be a bool, but was {type(edge_reversed)}")

    if not isinstance(graph_type, str):
        raise TypeError(f"graph_type must be a str, but was {type(verbose)}")

    if graph_type not in VALID_CPR_GRAPH_TYPES:
        raise ValueError(
            f"graph_type is not a valid value ({graph_type}), valid values are {','.join(VALID_CPR_GRAPH_TYPES)}"
        )

    if not isinstance(verbose, bool):
        raise TypeError(f"verbose must be a bool, but was {type(verbose)}")

    # fail fast in reaction_graph_attrs is not properly formatted
    for k in reaction_graph_attrs.keys():
        _validate_entity_attrs(reaction_graph_attrs[k])

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
        .assign(node_type=CPR_GRAPH_NODE_TYPES.SPECIES)
    )
    network_nodes.append(
        working_sbml_dfs.reactions.reset_index()[[SBML_DFS.R_ID, SBML_DFS.R_NAME]]
        .rename(columns={SBML_DFS.R_ID: "node_id", SBML_DFS.R_NAME: "node_name"})
        .assign(node_type=CPR_GRAPH_NODE_TYPES.REACTION)
    )

    # rename nodes to name since it is treated specially
    network_nodes_df = pd.concat(network_nodes).rename(
        columns={"node_id": CPR_GRAPH_NODES.NAME}
    )

    logger.info(f"Formatting edges as a {graph_type} graph")

    if graph_type == CPR_GRAPH_TYPES.BIPARTITE:
        network_edges = _create_cpr_graph_bipartite(working_sbml_dfs)
    elif graph_type in [CPR_GRAPH_TYPES.REGULATORY, CPR_GRAPH_TYPES.SURROGATE]:
        # pass graph_type so that an appropriate tiered schema can be used.
        network_edges = _create_cpr_graph_tiered(working_sbml_dfs, graph_type)
    else:
        raise NotImplementedError("Invalid graph_type")

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
                    direction=CPR_GRAPH_EDGE_DIRECTIONS.FORWARD
                ),
                # create reverse edges for reversibile reactions
                _reverse_network_edges(augmented_network_edges),
            ]
        )
    else:
        directed_network_edges = augmented_network_edges.assign(
            direction=CPR_GRAPH_EDGE_DIRECTIONS.UNDIRECTED
        )

    # de-duplicate edges
    unique_edges = (
        directed_network_edges.groupby([CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO])
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
                [CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO]
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

    # reverse edge directions if edge_reversed is True:

    if edge_reversed:
        rev_unique_edges_df = unique_edges.copy()
        rev_unique_edges_df[CPR_GRAPH_EDGES.FROM] = unique_edges[CPR_GRAPH_EDGES.TO]
        rev_unique_edges_df[CPR_GRAPH_EDGES.TO] = unique_edges[CPR_GRAPH_EDGES.FROM]
        rev_unique_edges_df[CPR_GRAPH_EDGES.SC_PARENTS] = unique_edges[
            CPR_GRAPH_EDGES.SC_CHILDREN
        ]
        rev_unique_edges_df[CPR_GRAPH_EDGES.SC_CHILDREN] = unique_edges[
            CPR_GRAPH_EDGES.SC_PARENTS
        ]
        rev_unique_edges_df[CPR_GRAPH_EDGES.STOICHOMETRY] = unique_edges[
            CPR_GRAPH_EDGES.STOICHOMETRY
        ] * (-1)

        rev_unique_edges_df[CPR_GRAPH_EDGES.DIRECTION] = unique_edges[
            CPR_GRAPH_EDGES.DIRECTION
        ].replace(
            {
                CPR_GRAPH_EDGE_DIRECTIONS.REVERSE: CPR_GRAPH_EDGE_DIRECTIONS.FORWARD,
                CPR_GRAPH_EDGE_DIRECTIONS.FORWARD: CPR_GRAPH_EDGE_DIRECTIONS.REVERSE,
            }
        )
    else:
        # unchanged if edge_reversed is False:
        rev_unique_edges_df = unique_edges

    # convert nodes and edgelist into an igraph network

    logger.info("Formatting cpr_graph output")
    cpr_graph = ig.Graph.DictList(
        vertices=network_nodes_df.to_dict("records"),
        edges=rev_unique_edges_df.to_dict("records"),
        directed=directed,
        vertex_name_attr=CPR_GRAPH_NODES.NAME,
        edge_foreign_keys=(CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO),
    )

    return cpr_graph


def process_cpr_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    reaction_graph_attrs: Optional[dict] = None,
    directed: bool = True,
    edge_reversed: bool = False,
    graph_type: str = CPR_GRAPH_TYPES.BIPARTITE,
    weighting_strategy: str = CPR_WEIGHTING_STRATEGIES.UNWEIGHTED,
    verbose: bool = False,
    custom_transformations: dict = None,
) -> ig.Graph:
    """
    Process Consensus Graph

    Setup an igraph network and then add weights and other maleable attributes.

    Args:
        sbml_dfs (SBML_dfs): A model formed by aggregating pathways
        reaction_graph_attrs (dict): Dictionary containing attributes to pull out of reaction_data and
            a weighting scheme for the graph
        directed (bool): Should a directed (True) or undirected graph be made (False)
        edge_reversed (bool): Should directions of edges be reversed (False)
        graph_type (str): Type of graph to create, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - reguatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products
        weighting_strategy (str) : a network weighting strategy with options:
            - unweighted: all weights (and upstream_weights for directed graphs) are set to 1.
            - topology: weight edges by the degree of the source nodes favoring nodes emerging from nodes
                with few connections.
            - mixed: transform edges with a quantitative score based on reaction_attrs; and set edges
                without quantitative score as a source-specific weight.
            - calibrated: transforme edges with a quantitative score based on reaction_attrs and combine them
                with topology scores to generate a consensus.
        verbose (bool): Extra reporting
        custom_transformations (dict, optional):
            Dictionary of custom transformation functions to use for attribute transformation.

    Returns:
        weighted_graph (ig.Graph): An Igraph network
    """

    if reaction_graph_attrs is None:
        reaction_graph_attrs = {}

    logging.info("Constructing network")
    cpr_graph = create_cpr_graph(
        sbml_dfs,
        reaction_graph_attrs,
        directed=directed,
        edge_reversed=edge_reversed,
        graph_type=graph_type,
        verbose=verbose,
        custom_transformations=custom_transformations,
    )

    if "reactions" in reaction_graph_attrs.keys():
        reaction_attrs = reaction_graph_attrs["reactions"]
    else:
        reaction_attrs = dict()

    logging.info(f"Adding edge weights with an {weighting_strategy} strategy")

    weighted_cpr_graph = add_graph_weights(
        cpr_graph=cpr_graph,
        reaction_attrs=reaction_attrs,
        weighting_strategy=weighting_strategy,
    )

    return weighted_cpr_graph


def pluck_entity_data(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    graph_attrs: dict[str, dict],
    data_type: str,
    custom_transformations: Optional[dict[str, callable]] = None,
) -> pd.DataFrame | None:
    """
    Pluck Entity Attributes

    Pull species or reaction attributes out of an sbml_dfs based on a set of
      tables and variables to look for.

    Parameters:
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A mechanistic model
    graph_attrs: dict
        A dictionary of species/reaction attributes to pull out. If the requested
        data_type ("species" or "reactions") is not present as a key, or if the value
        is an empty dict, this function will return None (no error).
    data_type: str
        "species" or "reactions" to pull out species_data or reactions_data
    custom_transformations: dict[str, callable], optional
        A dictionary mapping transformation names to functions. If provided, these
        will be checked before built-in transformations. Example:
            custom_transformations = {"square": lambda x: x**2}

    Returns:
        A table where all extracted attributes are merged based on a common index or None
        if no attributes were extracted. If the requested data_type is not present in
        graph_attrs, or if the attribute dict is empty, returns None. This is intended
        to allow optional annotation blocks.

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
    Apply Weight Transformations

    Args:
        edges_df (pd.DataFrame): a table of edges and their attributes extracted
            from a cpr_grpah.
        reaction_attrs (dict):
            A dictionary of attributes identifying weighting attributes within
            an sbml_df's reaction_data, how they will be named in edges_df (the keys),
            and how they should be transformed (the "trans" aliases")
        custom_transformations (dict, optional):
            A dictionary mapping transformation names to functions. If provided, these
            will be checked before built-in transformations.

    Returns:
        transformed_edges_df (pd.DataFrame): edges_df with weight variables transformed.

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


def summarize_weight_calibration(cpr_graph: ig.Graph, reaction_attrs: dict) -> None:
    """
    Summarize Weight Calibration

    For a network with multiple sources for edge weights summarize the alignment of
    different weighting schemes and how they map onto our notion of "good" versus
    "dubious" weights.

    Args:
        cpr_graph (ig.Graph): A graph where edge weights have already been calibrated.
        reaction_attrs (dict): a dictionary summarizing the types of weights that
            exist and how they are transformed for calibration.

    Returns:
        None

    """

    score_calibration_df = pd.DataFrame(SCORE_CALIBRATION_POINTS_DICT)
    score_calibration_df_calibrated = apply_weight_transformations(
        score_calibration_df, reaction_attrs
    )

    calibrated_edges = cpr_graph.get_edge_dataframe()

    _summarize_weight_calibration_table(
        calibrated_edges, score_calibration_df, score_calibration_df_calibrated
    )

    _summarize_weight_calibration_plots(
        calibrated_edges, score_calibration_df_calibrated
    )

    return None


def add_graph_weights(
    cpr_graph: ig.Graph,
    reaction_attrs: dict,
    weighting_strategy: str = CPR_WEIGHTING_STRATEGIES.UNWEIGHTED,
) -> ig.Graph:
    """
    Add Graph Weights

    Apply a weighting strategy to generate edge weights on a graph. For directed graphs "upstream_weights" will
    be generated as well which should be used when searching for a node's ancestors.

    Args:
        cpr_graph (ig.Graph): a graphical network of molecules/reactions (nodes) and edges linking them.
        reaction_attrs (dict): an optional dict
        weighting_strategy: a network weighting strategy with options:
            - unweighted: all weights (and upstream_weights for directed graphs) are set to 1.
            - topology: weight edges by the degree of the source nodes favoring nodes emerging from nodes
                with few connections.
            - mixed: transform edges with a quantitative score based on reaction_attrs; and set edges
                without quantitative score as a source-specific weight.
            - calibrated: transforme edges with a quantitative score based on reaction_attrs and combine them
                with topology scores to generate a consensus.

    """

    cpr_graph_updated = copy.deepcopy(cpr_graph)

    _validate_entity_attrs(reaction_attrs)

    if weighting_strategy not in VALID_WEIGHTING_STRATEGIES:
        raise ValueError(
            f"weighting_strategy was {weighting_strategy} and must be one of: "
            f"{', '.join(VALID_WEIGHTING_STRATEGIES)}"
        )

    # count parents and children and create weights based on them
    topology_weighted_graph = _create_topology_weights(cpr_graph_updated)

    if weighting_strategy == CPR_WEIGHTING_STRATEGIES.TOPOLOGY:
        topology_weighted_graph.es[CPR_GRAPH_EDGES.WEIGHTS] = (
            topology_weighted_graph.es["topo_weights"]
        )
        if cpr_graph_updated.is_directed():
            topology_weighted_graph.es[CPR_GRAPH_EDGES.UPSTREAM_WEIGHTS] = (
                topology_weighted_graph.es["upstream_topo_weights"]
            )

        return topology_weighted_graph

    if weighting_strategy == CPR_WEIGHTING_STRATEGIES.UNWEIGHTED:
        # set weights as a constant
        topology_weighted_graph.es[CPR_GRAPH_EDGES.WEIGHTS] = 1
        if cpr_graph_updated.is_directed():
            topology_weighted_graph.es[CPR_GRAPH_EDGES.UPSTREAM_WEIGHTS] = 1
        return topology_weighted_graph

    if weighting_strategy == CPR_WEIGHTING_STRATEGIES.MIXED:
        return _add_graph_weights_mixed(topology_weighted_graph, reaction_attrs)

    if weighting_strategy == CPR_WEIGHTING_STRATEGIES.CALIBRATED:
        return _add_graph_weights_calibration(topology_weighted_graph, reaction_attrs)

    raise ValueError(f"No logic implemented for {weighting_strategy}")


def _create_cpr_graph_bipartite(sbml_dfs: sbml_dfs_core.SBML_dfs) -> pd.DataFrame:
    """Turn an sbml_dfs model into a bipartite graph linking molecules to reactions."""

    # setup edges
    network_edges = (
        sbml_dfs.reaction_species.reset_index()[
            [SBML_DFS.R_ID, SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM]
        ]
        # rename species and reactions to reflect from -> to edges
        .rename(
            columns={
                SBML_DFS.SC_ID: CPR_GRAPH_NODE_TYPES.SPECIES,
                SBML_DFS.R_ID: CPR_GRAPH_NODE_TYPES.REACTION,
            }
        )
    )
    # add back an r_id variable so that each edge is annotated by a reaction
    network_edges[CPR_GRAPH_EDGES.R_ID] = network_edges[CPR_GRAPH_NODE_TYPES.REACTION]

    # add edge weights
    cspecies_features = sbml_dfs.get_cspecies_features()
    network_edges = network_edges.merge(
        cspecies_features, left_on=CPR_GRAPH_NODE_TYPES.SPECIES, right_index=True
    )

    # if directed then flip substrates and modifiers to the origin edge
    edge_vars = network_edges.columns.tolist()

    origins = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] <= 0]
    origin_edges = origins.loc[:, [edge_vars[1], edge_vars[0]] + edge_vars[2:]].rename(
        columns={
            CPR_GRAPH_NODE_TYPES.SPECIES: CPR_GRAPH_EDGES.FROM,
            CPR_GRAPH_NODE_TYPES.REACTION: CPR_GRAPH_EDGES.TO,
        }
    )

    dests = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] > 0]
    dest_edges = dests.rename(
        columns={
            CPR_GRAPH_NODE_TYPES.REACTION: CPR_GRAPH_EDGES.FROM,
            CPR_GRAPH_NODE_TYPES.SPECIES: CPR_GRAPH_EDGES.TO,
        }
    )

    network_edges = pd.concat([origin_edges, dest_edges])

    return network_edges


def _create_cpr_graph_tiered(
    sbml_dfs: sbml_dfs_core.SBML_dfs, graph_type: str
) -> pd.DataFrame:
    """Turn an sbml_dfs model into a tiered graph which links upstream entities to downstream ones."""

    # check whether all expect SBO terms are present
    invalid_sbo_terms = sbml_dfs.reaction_species[
        ~sbml_dfs.reaction_species[SBML_DFS.SBO_TERM].isin(MINI_SBO_TO_NAME.keys())
    ]

    if invalid_sbo_terms.shape[0] != 0:
        invalid_counts = invalid_sbo_terms.value_counts(SBML_DFS.SBO_TERM).to_frame("N")
        if not isinstance(invalid_counts, pd.DataFrame):
            raise TypeError("invalid_counts must be a pandas DataFrame")
        logger.warning(utils.style_df(invalid_counts, headers="keys"))  # type: ignore
        raise ValueError("Some reaction species have unusable SBO terms")

    # load and validate the schema of graph_type
    graph_hierarchy_df = _create_graph_hierarchy_df(graph_type)

    # organize reaction species for defining connections
    sorted_reaction_species = sbml_dfs.reaction_species.set_index(
        [SBML_DFS.R_ID, SBML_DFS.SBO_TERM]
    ).sort_index()

    logger.info(
        f"Formatting {sorted_reaction_species.shape[0]} reactions species as "
        "tiered edges."
    )

    # infer tiered edges in each reaction
    all_reaction_edges = [
        _format_tiered_reaction_species(
            r, sorted_reaction_species, sbml_dfs, graph_hierarchy_df
        )
        for r in sorted_reaction_species.index.get_level_values(SBML_DFS.R_ID).unique()
    ]
    all_reaction_edges_df = pd.concat(all_reaction_edges).reset_index(drop=True)

    # test for reactions missing substrates
    r_id_list = sorted_reaction_species.index.get_level_values(0).unique()
    r_id_reactant_only = [
        x for x in r_id_list if len(sorted_reaction_species.loc[x]) == 1
    ]

    if len(r_id_reactant_only) > 0:
        logger.warning(f"{len(r_id_reactant_only)} reactions are missing substrates")
        all_reaction_edges_df_pre = all_reaction_edges_df.copy()
        all_reaction_edges_df = all_reaction_edges_df_pre[
            ~all_reaction_edges_df_pre[SBML_DFS.R_ID].isin(r_id_reactant_only)
        ]

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
        all_reaction_edges_df.groupby([CPR_GRAPH_EDGES.FROM, CPR_GRAPH_EDGES.TO])
        .first()
        .reset_index()
    )

    # children
    n_children = (
        unique_edges[CPR_GRAPH_EDGES.FROM]
        .value_counts()
        # rename values to the child name
        .to_frame(name=CPR_GRAPH_EDGES.SC_CHILDREN)
        .reset_index()
        .rename(
            {
                CPR_GRAPH_EDGES.FROM: SBML_DFS.SC_ID,
            },
            axis=1,
        )
    )

    # parents
    n_parents = (
        unique_edges[CPR_GRAPH_EDGES.TO]
        .value_counts()
        # rename values to the parent name
        .to_frame(name=CPR_GRAPH_EDGES.SC_PARENTS)
        .reset_index()
        .rename(
            {
                CPR_GRAPH_EDGES.TO: SBML_DFS.SC_ID,
            },
            axis=1,
        )
    )

    graph_degree_by_edgelist = n_children.merge(n_parents, how="outer").fillna(int(0))

    graph_degree_by_edgelist[CPR_GRAPH_EDGES.SC_DEGREE] = (
        graph_degree_by_edgelist[CPR_GRAPH_EDGES.SC_CHILDREN]
        + graph_degree_by_edgelist[CPR_GRAPH_EDGES.SC_PARENTS]
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
                CPR_GRAPH_EDGES.SC_DEGREE,
                CPR_GRAPH_EDGES.SC_CHILDREN,
                CPR_GRAPH_EDGES.SC_PARENTS,
            ],
            axis=1,
        )
        .join(graph_degree_by_edgelist)
        .fillna(int(0))
    )

    is_from_reaction = all_reaction_edges_df[CPR_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index.tolist()
    )
    is_from_reaction = all_reaction_edges_df[CPR_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index
    )
    # add substrate weight whenever "from" edge is a molecule
    # and product weight when the "from" edge is a reaction
    decorated_all_reaction_edges_df = pd.concat(
        [
            all_reaction_edges_df[~is_from_reaction].merge(
                cspecies_features, left_on=CPR_GRAPH_EDGES.FROM, right_index=True
            ),
            all_reaction_edges_df[is_from_reaction].merge(
                cspecies_features, left_on=CPR_GRAPH_EDGES.TO, right_index=True
            ),
        ]
    ).sort_index()

    if all_reaction_edges_df.shape[0] != decorated_all_reaction_edges_df.shape[0]:
        msg = (
            "'decorated_all_reaction_edges_df' and 'all_reaction_edges_df' should\n"
            "have the same number of rows but they did not"
        )

        raise ValueError(msg)

    logger.info(f"Done preparing {graph_type} graph")

    return decorated_all_reaction_edges_df


def _format_tiered_reaction_species(
    r_id: str,
    sorted_reaction_species: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    graph_hierarchy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format Tiered Reaction Species

    Refactor a reaction's species into tiered edges between substrates, products, enzymes and allosteric regulators.
    """

    rxn_species = sorted_reaction_species.loc[r_id]
    if not isinstance(rxn_species, pd.DataFrame):
        raise TypeError("rxn_species must be a pandas DataFrame")
    if list(rxn_species.index.names) != [SBML_DFS.SBO_TERM]:
        raise ValueError("rxn_species index names must be [SBML_DFS.SBO_TERM]")
    if rxn_species.columns.tolist() != [SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY]:
        raise ValueError(
            "rxn_species columns must be [SBML_DFS.SC_ID, SBML_DFS.STOICHIOMETRY]"
        )

    rxn_sbo_terms = set(rxn_species.index.unique())
    # map to common names
    rxn_sbo_names = {MINI_SBO_TO_NAME[x] for x in rxn_sbo_terms}

    # is the reaction a general purpose interaction
    if len(rxn_sbo_names) == 1:
        if list(rxn_sbo_names)[0] == SBOTERM_NAMES.INTERACTOR:
            # further validation happens in the function - e.g., exactly two interactors
            return _format_interactors_for_tiered_graph(r_id, rxn_species, sbml_dfs)

    if SBOTERM_NAMES.INTERACTOR in rxn_sbo_names:
        logger.warning(
            f"Invalid combinations of SBO_terms in {str(r_id)} : {sbml_dfs.reactions.loc[r_id][SBML_DFS.R_NAME]}. "
            "If interactors are present then there can't be any other types of reaction species. "
            f"The following roles were defined: {', '.join(rxn_sbo_names)}"
        )

    # reorganize molecules and the reaction itself into tiers
    entities_ordered_by_tier = (
        pd.concat(
            [
                (
                    rxn_species.reset_index()
                    .rename({SBML_DFS.SC_ID: "entity_id"}, axis=1)
                    .merge(graph_hierarchy_df)
                ),
                graph_hierarchy_df[
                    graph_hierarchy_df[CPR_GRAPH_EDGES.SBO_NAME]
                    == CPR_GRAPH_NODE_TYPES.REACTION
                ].assign(entity_id=r_id, r_id=r_id),
            ]
        )
        .sort_values(["tier"])
        .set_index("tier")
    )
    ordered_tiers = entities_ordered_by_tier.index.get_level_values("tier").unique()

    if len(ordered_tiers) <= 1:
        raise ValueError("ordered_tiers must have more than one element")

    # which tier is the reaction?
    reaction_tier = graph_hierarchy_df["tier"][
        graph_hierarchy_df[CPR_GRAPH_EDGES.SBO_NAME] == CPR_GRAPH_NODE_TYPES.REACTION
    ].tolist()[0]

    rxn_edges = list()
    past_reaction = False
    for i in range(0, len(ordered_tiers) - 1):
        formatted_tier_combo = _format_tier_combo(
            entities_ordered_by_tier.loc[[ordered_tiers[i]]],
            entities_ordered_by_tier.loc[[ordered_tiers[i + 1]]],
            past_reaction,
        )

        if ordered_tiers[i + 1] == reaction_tier:
            past_reaction = True

        rxn_edges.append(formatted_tier_combo)

    rxn_edges_df = (
        pd.concat(rxn_edges)[
            [
                CPR_GRAPH_EDGES.FROM,
                CPR_GRAPH_EDGES.TO,
                CPR_GRAPH_EDGES.STOICHIOMETRY,
                CPR_GRAPH_EDGES.SBO_TERM,
            ]
        ]
        .reset_index(drop=True)
        .assign(r_id=r_id)
    )

    return rxn_edges_df


def _format_tier_combo(
    upstream_tier: pd.DataFrame, downstream_tier: pd.DataFrame, past_reaction: bool
) -> pd.DataFrame:
    """
    Format Tier Combo

    Create a set of edges crossing two tiers of a tiered graph. This will involve an
      all x all combination of entries. Tiers form an ordering along the molecular entities
      in a reaction plus a tier for the reaction itself. Attributes such as stoichiometry
      and sbo_term will be passed from the tier which is furthest from the reaction tier
      to ensure that each tier of molecular data applies its attributes to a single set of
      edges while the "reaction" tier does not. Reaction entities have neither a
      stoichiometery or sbo_term annotation.

    Args:
        upstream_tier (pd.DataFrame): A table containing upstream entities in a reaction,
            e.g., regulators.
        downstream_tier (pd.DataFrame): A table containing downstream entities in a reaction,
            e.g., catalysts.
        past_reaction (bool): if True then attributes will be taken from downstream_tier and
            if False they will come from upstream_tier.

    Returns:
        formatted_tier_combo (pd.DataFrame): A table of edges containing (from, to, stoichiometry, sbo_term, r_id). The
        number of edges is the product of the number of entities in the upstream tier
        times the number in the downstream tier.

    """

    upstream_fields = ["entity_id", SBML_DFS.STOICHIOMETRY, SBML_DFS.SBO_TERM]
    downstream_fields = ["entity_id"]

    if past_reaction:
        # swap fields
        upstream_fields, downstream_fields = downstream_fields, upstream_fields

    formatted_tier_combo = (
        upstream_tier[upstream_fields]
        .rename({"entity_id": CPR_GRAPH_EDGES.FROM}, axis=1)
        .assign(_joiner=1)
    ).merge(
        (
            downstream_tier[downstream_fields]
            .rename({"entity_id": CPR_GRAPH_EDGES.TO}, axis=1)
            .assign(_joiner=1)
        ),
        left_on="_joiner",
        right_on="_joiner",
    )

    return formatted_tier_combo


def _create_graph_hierarchy_df(graph_type: str) -> pd.DataFrame:
    """
    Create Graph Hierarchy DataFrame

    Format a graph hierarchy list of lists and a pd.DataFrame

    Args:
        graph_type (str):
            The type of tiered graph to work with. Each type has its own specification in constants.py.

    Returns:
        A pandas DataFrame with sbo_name, tier, and sbo_term.

    """

    if graph_type == CPR_GRAPH_TYPES.REGULATORY:
        sbo_names_hierarchy = REGULATORY_GRAPH_HIERARCHY
    elif graph_type == CPR_GRAPH_TYPES.SURROGATE:
        sbo_names_hierarchy = SURROGATE_GRAPH_HIERARCHY
    else:
        raise NotImplementedError(f"{graph_type} is not a valid graph_type")

    # format as a DF
    graph_hierarchy_df = pd.concat(
        [
            pd.DataFrame({"sbo_name": sbo_names_hierarchy[i]}).assign(tier=i)
            for i in range(0, len(sbo_names_hierarchy))
        ]
    ).reset_index(drop=True)
    graph_hierarchy_df[SBML_DFS.SBO_TERM] = graph_hierarchy_df["sbo_name"].apply(
        lambda x: MINI_SBO_FROM_NAME[x] if x != CPR_GRAPH_NODE_TYPES.REACTION else ""
    )

    # ensure that the output is expected
    utils.match_pd_vars(
        graph_hierarchy_df,
        req_vars={CPR_GRAPH_EDGES.SBO_NAME, "tier", SBML_DFS.SBO_TERM},
        allow_series=False,
    ).assert_present()

    return graph_hierarchy_df


def _add_graph_weights_mixed(cpr_graph: ig.Graph, reaction_attrs: dict) -> ig.Graph:
    """Weight a graph using a mixed approach combining source-specific weights and existing edge weights."""

    edges_df = cpr_graph.get_edge_dataframe()

    calibrated_edges = apply_weight_transformations(edges_df, reaction_attrs)
    calibrated_edges = _create_source_weights(calibrated_edges, "source_wt")

    score_vars = list(reaction_attrs.keys())
    score_vars.append("source_wt")

    logger.info(f"Creating mixed scores based on {', '.join(score_vars)}")

    calibrated_edges["weights"] = calibrated_edges[score_vars].min(axis=1)

    cpr_graph.es[CPR_GRAPH_EDGES.WEIGHTS] = calibrated_edges[CPR_GRAPH_EDGES.WEIGHTS]
    if cpr_graph.is_directed():
        cpr_graph.es[CPR_GRAPH_EDGES.UPSTREAM_WEIGHTS] = calibrated_edges[
            CPR_GRAPH_EDGES.WEIGHTS
        ]

    # add other attributes and update transformed attributes
    cpr_graph.es["source_wt"] = calibrated_edges["source_wt"]
    for k in reaction_attrs.keys():
        cpr_graph.es[k] = calibrated_edges[k]

    return cpr_graph


def _add_graph_weights_calibration(
    cpr_graph: ig.Graph, reaction_attrs: dict
) -> ig.Graph:
    """Weight a graph using a calibrated strategy which aims to roughly align qualiatively similar weights from different sources."""

    edges_df = cpr_graph.get_edge_dataframe()

    calibrated_edges = apply_weight_transformations(edges_df, reaction_attrs)

    score_vars = list(reaction_attrs.keys())
    score_vars.append("topo_weights")

    logger.info(f"Creating calibrated scores based on {', '.join(score_vars)}")
    cpr_graph.es["weights"] = calibrated_edges[score_vars].min(axis=1)

    if cpr_graph.is_directed():
        score_vars = list(reaction_attrs.keys())
        score_vars.append("upstream_topo_weights")
        cpr_graph.es["upstream_weights"] = calibrated_edges[score_vars].min(axis=1)

    # add other attributes and update transformed attributes
    for k in reaction_attrs.keys():
        cpr_graph.es[k] = calibrated_edges[k]

    return cpr_graph


def _add_edge_attr_to_vertex_graph(
    cpr_graph: ig.Graph,
    edge_attr_list: list,
    shared_node_key: str = "r_id",
) -> ig.Graph:
    """
    Merge edge attribute(s) from edge_attr_list to vetices of an igraph

    Parameters
    ----------
    cpr_graph : iGraph
        A graph generated by create_cpr_graph()
    edge_attr_list: list
        A list containing attributes to pull out of edges, then to add to vertices
    shared_node_key : str
        key in edge that is shared with vertex, to map edge ids to corresponding vertex ids

    Returns:
    ----------
    An Igraph network
    """

    if len(edge_attr_list) == 0:
        logger.warning(
            "No edge attributes were passed, " "thus return the input graph."
        )
        return cpr_graph

    graph_vertex_df = cpr_graph.get_vertex_dataframe()
    graph_edge_df = cpr_graph.get_edge_dataframe()

    if shared_node_key not in graph_edge_df.columns.to_list():
        logger.warning(
            f"{shared_node_key} is not in the current edge attributes. "
            "shared_node_key must be an existing edge attribute"
        )
        return cpr_graph

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
    # as in net_create.create_cpr_graph(), vertex_name_attr is set to 'name'
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

    # assign the edge_attrs from edge_attr_list to cpr_graph's vertices:
    # keep the same edge attribute names:
    for col_name in edge_attr_list:
        cpr_graph.vs[col_name] = graph_vertex_df_w_edge_attr[col_name]

    return cpr_graph


def _summarize_weight_calibration_table(
    calibrated_edges: pd.DataFrame,
    score_calibration_df: pd.DataFrame,
    score_calibration_df_calibrated: pd.DataFrame,
):
    """Create a table comparing edge weights from multiple sources."""

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
    """Create a couple of plots summarizing the relationships between different scoring measures."""

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
    """ "
    Create Source Weights

    Create weights based on an edges source. This is a simple but crude way of allowing different
    data sources to have different support if we think that some are more trustworthly than others.

    Args:
        edges_df: pd.DataFrame
            The edges dataframe to add the source weights to.
        source_wt_var: str
            The name of the column to store the source weights.
        source_vars_dict: dict
            Dictionary with keys indicating edge attributes and values indicating the weight to assign
            to that attribute. This value is generally the largest weight that can be assigned to an
            edge so that the numeric weight is chosen over the default.
        source_wt_default: int
            The default weight to assign to an edge if no other weight attribute is found.

    Returns:
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
    """Identity"""
    return x


def _wt_transformation_string(x):
    """Map STRING scores to a similar scale as topology weights."""

    return 250000 / np.power(x, 1.7)


def _wt_transformation_string_inv(x):
    """Map STRING scores so they work with source weights."""

    # string scores are bounded on [0, 1000]
    # and score/1000 is roughly a probability that
    # there is a real interaction (physical, genetic, ...)
    # reported string scores are currently on [150, 1000]
    # so this transformation will map these onto {6.67, 1}

    return 1 / (x / 1000)


def _format_interactors_for_tiered_graph(
    r_id: str, rxn_species: pd.DataFrame, sbml_dfs: sbml_dfs_core.SBML_dfs
) -> pd.DataFrame:
    """Format an undirected interactions for tiered graph so interactions are linked even though they would be on the same tier."""

    interactor_data = rxn_species.loc[MINI_SBO_FROM_NAME["interactor"]]
    if interactor_data.shape[0] != 2:
        raise ValueError(
            f"{interactor_data.shape[0]} interactors present for {str(r_id)} : "
            f"{sbml_dfs.reactions.loc[r_id]['r_name']}. "
            "Reactions with interactors must have exactly two interactors"
        )

    if not (interactor_data["stoichiometry"] == 0).any():
        raise ValueError(
            f"Interactors had non-zero stoichiometry for {str(r_id)} : {sbml_dfs.reactions.loc[r_id]['r_name']}. "
            "If stoichiometry is important for this reaction then it should use other SBO terms "
            "(e.g., substrate and product)."
        )

    # set the first entry as "from" and second as "to" if stoi is zero.
    # the reverse reaction will generally be added later because these
    # reactions should be reversible

    return pd.DataFrame(
        {
            "from": interactor_data["sc_id"].iloc[0],
            "to": interactor_data["sc_id"].iloc[1],
            "sbo_term": MINI_SBO_FROM_NAME["interactor"],
            "stoichiometry": 0,
            "r_id": r_id,
        },
        index=[0],
    )


def _add_graph_species_attribute(
    cpr_graph: ig.Graph,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    species_graph_attrs: dict,
    custom_transformations: Optional[dict] = None,
) -> ig.Graph:
    """
    Add meta-data from species_data to existing igraph's vertices.

    This function augments the vertices of an igraph network with additional attributes
    derived from the species-level data in the provided SBML_dfs object. The attributes
    to add are specified in the species_graph_attrs dictionary, and can be transformed
    using either built-in or user-supplied transformation functions.

    Parameters
    ----------
    cpr_graph : ig.Graph
        The igraph network to augment.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing species data.
    species_graph_attrs : dict
        Dictionary specifying which attributes to pull from species_data and how to transform them.
        The structure should be {attribute_name: {"table": ..., "variable": ..., "trans": ...}}.
    custom_transformations : dict, optional
        Dictionary mapping transformation names to functions. If provided, these will be checked
        before built-in transformations. Example: {"square": lambda x: x**2}

    Returns
    -------
    ig.Graph
        The input igraph network with additional vertex attributes added from species_data.
    """
    if not isinstance(species_graph_attrs, dict):
        raise TypeError(
            f"species_graph_attrs must be a dict, but was {type(species_graph_attrs)}"
        )

    # fail fast if species_graph_attrs is not properly formatted
    # also flatten attribute list to be added to vertex nodes
    sp_graph_key_list = []
    sp_node_attr_list = []
    for k in species_graph_attrs.keys():
        _validate_entity_attrs(
            species_graph_attrs[k], custom_transformations=custom_transformations
        )

        sp_graph_key_list.append(k)
        sp_node_attr_list.append(list(species_graph_attrs[k].keys()))

    # flatten sp_node_attr_list
    flat_sp_node_attr_list = [item for items in sp_node_attr_list for item in items]

    logger.info("Adding meta-data from species_data")

    curr_network_nodes_df = cpr_graph.get_vertex_dataframe()

    # add species-level attributes to nodes dataframe
    augmented_network_nodes_df = _augment_network_nodes(
        curr_network_nodes_df,
        sbml_dfs,
        species_graph_attrs,
        custom_transformations=custom_transformations,
    )

    for vs_attr in flat_sp_node_attr_list:
        # in case more than one vs_attr in the flat_sp_node_attr_list
        logger.info(f"Adding new attribute {vs_attr} to vertices")
        cpr_graph.vs[vs_attr] = augmented_network_nodes_df[vs_attr].values

    return cpr_graph


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
    species_graph_attrs : dict
        Dictionary specifying which attributes to pull from species_data and how to transform them.
        The structure should be {attribute_name: {"table": ..., "variable": ..., "trans": ...}}.
    custom_transformations : dict, optional
        Dictionary mapping transformation names to functions. If provided, these will be checked
        before built-in transformations. Example: {"square": lambda x: x**2}

    Returns
    -------
    pd.DataFrame
        The input network_nodes DataFrame with additional columns for each extracted and transformed attribute.
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
    # (the index of network_nodes df) in network_nodes df
    network_nodes_sid = pd.merge(
        network_nodes,
        sbml_dfs.compartmentalized_species[["s_id", "c_id"]],
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
        network_nodes_wdata = network_nodes_sid.merge(
            species_graph_data, left_on="s_id", right_index=True, how="left"
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
    """Add reversibility and other metadata from reactions.

    Parameters
    ----------
    network_edges : pd.DataFrame
        DataFrame of network edges.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing reaction data.
    reaction_graph_attrs : dict
        Dictionary of reaction attributes to add.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.
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
    """Flip reversible reactions to derive the reverse reaction."""

    # validate inputs
    missing_required_vars = CPR_GRAPH_REQUIRED_EDGE_VARS.difference(
        set(augmented_network_edges.columns.tolist())
    )

    if len(missing_required_vars) > 0:
        raise ValueError(
            "augmented_network_edges is missing required variables: "
            f"{', '.join(missing_required_vars)}"
        )

    # select all edges derived from reversible reactions
    reversible_reaction_edges = augmented_network_edges[
        augmented_network_edges[CPR_GRAPH_EDGES.R_ISREVERSIBLE]
    ]

    r_reaction_edges = (
        # ignore edges which start in a regulator or catalyst; even for a reversible reaction it
        # doesn't make sense for a regulator to be impacted by a target
        reversible_reaction_edges[
            ~reversible_reaction_edges[CPR_GRAPH_EDGES.SBO_TERM].isin(
                [
                    MINI_SBO_FROM_NAME[x]
                    for x in SBO_MODIFIER_NAMES.union({SBOTERM_NAMES.CATALYST})
                ]
            )
        ]
        # flip parent and child attributes
        .rename(
            {
                CPR_GRAPH_EDGES.FROM: CPR_GRAPH_EDGES.TO,
                CPR_GRAPH_EDGES.TO: CPR_GRAPH_EDGES.FROM,
                CPR_GRAPH_EDGES.SC_CHILDREN: CPR_GRAPH_EDGES.SC_PARENTS,
                CPR_GRAPH_EDGES.SC_PARENTS: CPR_GRAPH_EDGES.SC_CHILDREN,
            },
            axis=1,
        )
    )

    # switch substrates and products
    r_reaction_edges[CPR_GRAPH_EDGES.STOICHIOMETRY] = r_reaction_edges[
        CPR_GRAPH_EDGES.STOICHIOMETRY
    ].apply(
        # the ifelse statement prevents 0 being converted to -0 ...
        lambda x: -1 * x if x != 0 else 0
    )

    transformed_r_reaction_edges = pd.concat(
        [
            (
                r_reaction_edges[
                    r_reaction_edges[CPR_GRAPH_EDGES.SBO_TERM]
                    == MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]
                ].assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT])
            ),
            (
                r_reaction_edges[
                    r_reaction_edges[CPR_GRAPH_EDGES.SBO_TERM]
                    == MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]
                ].assign(sbo_term=MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT])
            ),
            r_reaction_edges[
                ~r_reaction_edges[CPR_GRAPH_EDGES.SBO_TERM].isin(
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
        direction=CPR_GRAPH_EDGE_DIRECTIONS.REVERSE
    )


def _create_topology_weights(
    cpr_graph: ig.Graph,
    base_score: float = 2,
    protein_multiplier: int = 1,
    metabolite_multiplier: int = 3,
    unknown_multiplier: int = 10,
    scale_multiplier_by_meandegree: bool = True,
) -> ig.Graph:
    """
    Create Topology Weights

    Add weights to a network based on its topology. Edges downstream of nodes
    with many connections receive a higher weight suggesting that any one
    of them is less likely to be regulatory. This is a simple and clearly
    flawed heuristic which can be combined with more principled weighting
    schemes.

    Args:
        cpr_graph (ig.Graph): a graph containing connections between molecules, proteins, and reactions.
        base_score (float): offset which will be added to all weights.
        protein_multiplier (int): multiplier for non-metabolite species (lower weight paths will tend to be selected).
        metabolite_multiplier (int): multiplier for metabolites [defined a species with a ChEBI ID).
        unknown_multiplier (int): multiplier for species without any identifier. See sbml_dfs_core.species_type_types.
        scale_multiplier_by_meandegree (bool): if True then multipliers will be rescaled by the average number of
            connections a node has (i.e., its degree) so that weights will be relatively similar regardless of network
            size and sparsity.

    Returns:
        cpr_graph (ig.Graph): graph with added topology weights

    """

    # check for required attribute before proceeding

    required_attrs = {
        CPR_GRAPH_EDGES.SC_DEGREE,
        CPR_GRAPH_EDGES.SC_CHILDREN,
        CPR_GRAPH_EDGES.SC_PARENTS,
        CPR_GRAPH_EDGES.SPECIES_TYPE,
    }

    missing_required_attrs = required_attrs.difference(set(cpr_graph.es.attributes()))
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
            CPR_GRAPH_EDGES.SC_DEGREE: cpr_graph.es[CPR_GRAPH_EDGES.SC_DEGREE],
            CPR_GRAPH_EDGES.SC_CHILDREN: cpr_graph.es[CPR_GRAPH_EDGES.SC_CHILDREN],
            CPR_GRAPH_EDGES.SC_PARENTS: cpr_graph.es[CPR_GRAPH_EDGES.SC_PARENTS],
            CPR_GRAPH_EDGES.SPECIES_TYPE: cpr_graph.es[CPR_GRAPH_EDGES.SPECIES_TYPE],
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
        mean_degree = len(cpr_graph.es) / len(cpr_graph.vs)
        if not cpr_graph.is_directed():
            # for a directed network in- and out-degree are separately treated while
            # an undirected network's degree will be the sum of these two measures.
            mean_degree = mean_degree * 2

        weight_table["multiplier"] = weight_table["multiplier"] / mean_degree

    if cpr_graph.is_directed():
        weight_table["connection_weight"] = weight_table[CPR_GRAPH_EDGES.SC_CHILDREN]
    else:
        weight_table["connection_weight"] = weight_table[CPR_GRAPH_EDGES.SC_DEGREE]

    # weight traveling through a species based on
    # - a constant
    # - how plausibly that species type mediates a change
    # - the number of connections that the node can bridge to
    weight_table["topo_weights"] = [
        base_score + (x * y)
        for x, y in zip(weight_table["multiplier"], weight_table["connection_weight"])
    ]
    cpr_graph.es["topo_weights"] = weight_table["topo_weights"]

    # if directed and we want to use travel upstream define a corresponding weighting scheme
    if cpr_graph.is_directed():
        weight_table["upstream_topo_weights"] = [
            base_score + (x * y)
            for x, y in zip(weight_table["multiplier"], weight_table["sc_parents"])
        ]
        cpr_graph.es["upstream_topo_weights"] = weight_table["upstream_topo_weights"]

    return cpr_graph


def _validate_entity_attrs(
    entity_attrs: dict,
    validate_transformations: bool = True,
    custom_transformations: Optional[dict] = None,
) -> None:
    """Validate that graph attributes are a valid format."""

    assert isinstance(entity_attrs, dict)
    for v in entity_attrs.values():
        # check structure against pydantic config
        entity_attrs = _EntityAttrValidator(**v).model_dump()

        if validate_transformations:
            trans_name = v["trans"]
            valid_trans = set(DEFINED_WEIGHT_TRANSFORMATION.keys())
            if custom_transformations:
                valid_trans = valid_trans.union(set(custom_transformations.keys()))
            if trans_name not in valid_trans:
                raise ValueError(
                    f"transformation {trans_name} was not defined as an alias in "
                    "DEFINED_WEIGHT_TRANSFORMATION or custom_transformations. The defined transformations "
                    f"are {', '.join(valid_trans)}"
                )

    return None


class _EntityAttrValidator(BaseModel):
    table: str
    variable: str
    trans: Optional[str] = DEFAULT_WT_TRANS
