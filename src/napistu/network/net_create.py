from __future__ import annotations

import copy
import logging
from typing import Optional

import igraph as ig
import pandas as pd

from napistu import sbml_dfs_core, utils
from napistu.constants import (
    MINI_SBO_FROM_NAME,
    SBML_DFS,
    SBML_DFS_METHOD_DEFS,
    SBO_MODIFIER_NAMES,
    SBOTERM_NAMES,
)
from napistu.network import net_create_utils, ng_utils
from napistu.network.constants import (
    DROP_REACTIONS_WHEN,
    GRAPH_WIRING_APPROACHES,
    NAPISTU_GRAPH_EDGE_DIRECTIONS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
    NAPISTU_WEIGHTING_STRATEGIES,
    VALID_GRAPH_WIRING_APPROACHES,
)
from napistu.network.ng_core import (
    NapistuGraph,
    _apply_edge_reversal_mapping,
    _handle_special_reversal_cases,
)

logger = logging.getLogger(__name__)


def create_napistu_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    directed: bool = True,
    wiring_approach: str = GRAPH_WIRING_APPROACHES.REGULATORY,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
    deduplicate_edges: bool = True,
    verbose: bool = False,
) -> NapistuGraph:
    """
    Create a NapistuGraph network from a mechanistic network using one of a set of wiring approaches.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A model formed by aggregating pathways.
    directed : bool, optional
        Whether to create a directed (True) or undirected (False) graph. Default is True.
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
    deduplicate_edges : bool, optional
        Whether to deduplicate edges with the same FROM -> TO pair, keeping only the first occurrence.
        Default is True for backwards compatibility.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    NapistuGraph
        A NapistuGraph network (subclass of igraph.Graph).

    Raises
    ------
    ValueError
        If wiring_approach is not valid or if required attributes are missing.
    """

    if wiring_approach not in VALID_GRAPH_WIRING_APPROACHES + ["bipartite_og"]:
        raise ValueError(
            f"wiring_approach is not a valid value ({wiring_approach}), valid values are {','.join(VALID_GRAPH_WIRING_APPROACHES)}"
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

    logger.debug("DEBUG: creating compartmentalized species features")

    cspecies_features = working_sbml_dfs.get_cspecies_features().drop(
        columns=[
            SBML_DFS_METHOD_DEFS.SC_DEGREE,
            SBML_DFS_METHOD_DEFS.SC_CHILDREN,
            SBML_DFS_METHOD_DEFS.SC_PARENTS,
        ]
    )

    logger.info(
        "Organizing all network nodes (compartmentalized species and reactions)"
    )

    species_vertices = (
        working_sbml_dfs.compartmentalized_species.reset_index()[
            [SBML_DFS.SC_ID, SBML_DFS.SC_NAME]
        ]
        .rename(
            columns={
                SBML_DFS.SC_ID: NAPISTU_GRAPH_VERTICES.NAME,
                SBML_DFS.SC_NAME: NAPISTU_GRAPH_VERTICES.NODE_NAME,
            }
        )
        .assign(**{NAPISTU_GRAPH_VERTICES.NODE_TYPE: NAPISTU_GRAPH_NODE_TYPES.SPECIES})
        .merge(
            cspecies_features,
            left_on=NAPISTU_GRAPH_VERTICES.NAME,
            right_index=True,
            how="left",
        )
        .merge(
            working_sbml_dfs.compartmentalized_species[[SBML_DFS.S_ID, SBML_DFS.C_ID]],
            left_on=NAPISTU_GRAPH_VERTICES.NAME,
            right_index=True,
            how="left",
        )
    )

    reaction_vertices = (
        working_sbml_dfs.reactions.reset_index()[[SBML_DFS.R_ID, SBML_DFS.R_NAME]]
        .rename(
            columns={
                SBML_DFS.R_ID: NAPISTU_GRAPH_VERTICES.NAME,
                SBML_DFS.R_NAME: NAPISTU_GRAPH_VERTICES.NODE_NAME,
            }
        )
        .assign(**{NAPISTU_GRAPH_VERTICES.NODE_TYPE: NAPISTU_GRAPH_NODE_TYPES.REACTION})
    )

    network_nodes_df = pd.concat([species_vertices, reaction_vertices])

    logger.info(f"Formatting edges as a {wiring_approach} graph")

    if wiring_approach == "bipartite_og":
        network_edges = _create_napistu_graph_bipartite(working_sbml_dfs)
    elif wiring_approach in VALID_GRAPH_WIRING_APPROACHES:
        # pass wiring_approach so that an appropriate tiered schema can be used.
        network_edges = net_create_utils.wire_reaction_species(
            working_sbml_dfs.reaction_species, wiring_approach, drop_reactions_when
        )
    else:
        raise NotImplementedError("Invalid wiring_approach")

    logger.info("Adding reversibility and other meta-data from reactions_data")
    augmented_network_edges = _augment_network_edges(
        network_edges,
        working_sbml_dfs,
        cspecies_features,
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

    # convert nodes and edgelist into an igraph network
    logger.info("Formatting NapistuGraph output")
    napistu_ig_graph = ig.Graph.DictList(
        vertices=network_nodes_df.to_dict("records"),
        edges=directed_network_edges.to_dict("records"),
        directed=directed,
        vertex_name_attr=NAPISTU_GRAPH_VERTICES.NAME,
        edge_foreign_keys=(NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO),
    )

    # Always return NapistuGraph
    napistu_graph = NapistuGraph.from_igraph(
        napistu_ig_graph, wiring_approach=wiring_approach
    )

    # validate assumptions about the graph structure
    napistu_graph.validate()

    # remove singleton nodes (mostly reactions that are not part of any interaction)
    napistu_graph.remove_isolated_vertices()

    # de-duplicate edges if requested
    if deduplicate_edges:
        napistu_graph.deduplicate_edges(verbose=verbose)

    return napistu_graph


def process_napistu_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    directed: bool = True,
    wiring_approach: str = GRAPH_WIRING_APPROACHES.BIPARTITE,
    weighting_strategy: str = NAPISTU_WEIGHTING_STRATEGIES.UNWEIGHTED,
    reaction_graph_attrs: Optional[dict] = None,
    custom_transformations: dict = None,
    deduplicate_edges: bool = True,
    drop_reactions_when: str = DROP_REACTIONS_WHEN.SAME_TIER,
    verbose: bool = False,
) -> NapistuGraph:
    """
    Process Consensus Graph.

    Sets up a NapistuGraph network and then adds weights and other malleable attributes.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A model formed by aggregating pathways.
    directed : bool, optional
        Whether to create a directed (True) or undirected (False) graph. Default is True.
    wiring_approach : str, optional
        Type of graph to create. See `create_napistu_graph` for valid values.
    weighting_strategy : str, optional
        A network weighting strategy. Options:
            - 'unweighted': all weights (and upstream_weight for directed graphs) are set to 1.
            - 'topology': weight edges by the degree of the source nodes favoring nodes with few connections.
            - 'mixed': transform edges with a quantitative score based on reaction_attrs; and set edges without quantitative score as a source-specific weight.
    reaction_graph_attrs : dict, optional
        Dictionary containing attributes to pull out of reaction_data and a weighting scheme for the graph.
    custom_transformations : dict, optional
        Dictionary of custom transformation functions to use for attribute transformation.
    deduplicate_edges : bool, optional
        Whether to deduplicate edges with the same FROM -> TO pair, keeping only the first occurrence.
        Default is True for backwards compatibility.
    drop_reactions_when : str, optional
        The condition under which to drop reactions as a network vertex. Valid values are:
            - 'same_tier': drop reactions when all participants are on the same tier of a wiring hierarchy
            - 'edgelist': drop reactions when the reaction species are only 2 (1 reactant + 1 product)
            - 'always': drop reactions regardless of tiers
        Default is 'same_tier'.
    verbose : bool, optional
        Extra reporting. Default is False.

    Returns
    -------
    NapistuGraph
        A weighted NapistuGraph network (subclass of igraph.Graph).
    """

    if reaction_graph_attrs is None:
        reaction_graph_attrs = {}

    # fail fast if reaction_graph_attrs is pathological
    for k in reaction_graph_attrs.keys():
        ng_utils._validate_entity_attrs(
            reaction_graph_attrs[k], custom_transformations=custom_transformations
        )

    logging.info("Constructing network")
    napistu_graph = create_napistu_graph(
        sbml_dfs,
        directed=directed,
        wiring_approach=wiring_approach,
        drop_reactions_when=drop_reactions_when,
        deduplicate_edges=deduplicate_edges,
        verbose=verbose,
    )

    # pull out the requested attributes
    napistu_graph.set_graph_attrs(reaction_graph_attrs)
    napistu_graph.add_edge_data(sbml_dfs)
    napistu_graph.transform_edges(custom_transformations=custom_transformations)

    if SBML_DFS.REACTIONS in reaction_graph_attrs.keys():
        reaction_attrs = reaction_graph_attrs[SBML_DFS.REACTIONS]
    else:
        reaction_attrs = dict()

    logging.info(f"Adding edge weights with an {weighting_strategy} strategy")

    napistu_graph.set_weights(
        weight_by=list(reaction_attrs.keys()),
        weighting_strategy=weighting_strategy,
    )

    return napistu_graph


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

    # if directed then flip substrates and modifiers to the origin edge
    edge_vars = network_edges.columns.tolist()

    origins = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] <= 0]
    origin_edges = origins.loc[:, [edge_vars[1], edge_vars[0]] + edge_vars[2:]].rename(
        columns={
            NAPISTU_GRAPH_NODE_TYPES.SPECIES: NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_NODE_TYPES.REACTION: NAPISTU_GRAPH_EDGES.TO,
            SBML_DFS.STOICHIOMETRY: NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
            SBML_DFS.SBO_TERM: NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
        }
    )

    dests = network_edges[network_edges[SBML_DFS.STOICHIOMETRY] > 0]
    dest_edges = dests.rename(
        columns={
            NAPISTU_GRAPH_NODE_TYPES.REACTION: NAPISTU_GRAPH_EDGES.FROM,
            NAPISTU_GRAPH_NODE_TYPES.SPECIES: NAPISTU_GRAPH_EDGES.TO,
            SBML_DFS.STOICHIOMETRY: NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
            SBML_DFS.SBO_TERM: NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
        }
    )

    # Concatenate first (pd.concat fills missing columns with NaN)
    network_edges = pd.concat([origin_edges, dest_edges], ignore_index=True)

    # Replace NaN with None (from pd.concat) to match tiered approach behavior
    for col in [
        NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM,
        NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_DOWNSTREAM,
        NAPISTU_GRAPH_EDGES.STOICHIOMETRY_UPSTREAM,
    ]:
        if col in network_edges.columns:
            network_edges[col] = network_edges[col].where(
                network_edges[col].notna(), None
            )

    return network_edges


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
        SBML_DFS.SPECIES,
        custom_transformations=custom_transformations,
        # to do - separate concern by add data to the graph and then transforming results
        # to better track transformations and allow for rewinding
        transform=True,
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
        columns=[SBML_DFS.S_ID, SBML_DFS.C_ID]
    )

    return network_nodes_wdata


def _augment_network_edges(
    network_edges: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    cspecies_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add reversibility and other metadata from reactions, and species features.

    Parameters
    ----------
    network_edges : pd.DataFrame
        DataFrame of network edges.
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The SBML_dfs object containing reaction data.
    cspecies_features : pd.DataFrame
        DataFrame containing species features to merge with edges.

    Returns
    -------
    pd.DataFrame
        DataFrame of network edges with additional metadata.

    Raises
    ------
    ValueError
        If required attributes are missing from network_edges.
    """

    missing_required_network_edges_attrs = [
        attr
        for attr in NAPISTU_GRAPH_EDGES_FROM_WIRING_VARS
        if attr not in network_edges.columns
    ]
    if len(missing_required_network_edges_attrs) > 0:
        raise ValueError(
            f"{len(missing_required_network_edges_attrs)} required attributes were missing "
            "from network_edges: "
            f"{', '.join(missing_required_network_edges_attrs)}"
        )

    # Determine which edges are from reactions vs species
    is_from_reaction = network_edges[NAPISTU_GRAPH_EDGES.FROM].isin(
        sbml_dfs.reactions.index
    )

    # Merge species features with edges
    # For edges where FROM is a species (not reaction), use FROM node's features
    # For edges where FROM is a reaction, use TO node's features
    augmented_network_edges = (
        pd.concat(
            [
                network_edges[~is_from_reaction].merge(
                    cspecies_features,
                    left_on=NAPISTU_GRAPH_EDGES.FROM,
                    right_index=True,
                ),
                network_edges[is_from_reaction].merge(
                    cspecies_features, left_on=NAPISTU_GRAPH_EDGES.TO, right_index=True
                ),
            ]
        )
        .sort_index()
        .merge(
            sbml_dfs.reactions[SBML_DFS.R_ISREVERSIBLE],
            left_on=SBML_DFS.R_ID,
            right_index=True,
            how="left",
        )
    )

    if augmented_network_edges.shape[0] != network_edges.shape[0]:
        raise ValueError(
            "Augmented_network_edges and network_edges must have the same number of rows. Please contact the developers."
        )

    return augmented_network_edges


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

    # Filter: ignore edges which start in a regulator or catalyst; even for a reversible reaction it
    # doesn't make sense for a regulator to be impacted by a target
    filter_mask = ~reversible_reaction_edges[
        NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM
    ].isin(
        [
            MINI_SBO_FROM_NAME[x]
            for x in SBO_MODIFIER_NAMES.union({SBOTERM_NAMES.CATALYST})
        ]
    )

    r_reaction_edges = reversible_reaction_edges[filter_mask]

    # Apply systematic attribute swapping
    r_reaction_edges = _apply_edge_reversal_mapping(r_reaction_edges)

    # Handle special cases (negate stoichiometries)
    # Note: ignore_direction=True because direction attribute is added later
    r_reaction_edges = _handle_special_reversal_cases(
        r_reaction_edges, ignore_direction=True
    )

    # Transform SBO terms: swap reactant <-> product (specific to network creation)
    reactant_term = MINI_SBO_FROM_NAME[SBOTERM_NAMES.REACTANT]
    product_term = MINI_SBO_FROM_NAME[SBOTERM_NAMES.PRODUCT]

    if NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM in r_reaction_edges.columns:
        r_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM] = r_reaction_edges[
            NAPISTU_GRAPH_EDGES.SBO_TERM_UPSTREAM
        ].replace(
            {
                reactant_term: product_term,
                product_term: reactant_term,
            }
        )

    if NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM in r_reaction_edges.columns:
        r_reaction_edges[NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM] = r_reaction_edges[
            NAPISTU_GRAPH_EDGES.SBO_TERM_DOWNSTREAM
        ].replace(
            {
                reactant_term: product_term,
                product_term: reactant_term,
            }
        )

    # Set direction to REVERSE for these reversed edges
    r_reaction_edges = r_reaction_edges.assign(
        **{NAPISTU_GRAPH_EDGES.DIRECTION: NAPISTU_GRAPH_EDGE_DIRECTIONS.REVERSE}
    )

    return r_reaction_edges
