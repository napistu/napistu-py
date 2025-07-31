"""
Utilities specific to NapistuGraph objects and the wider Napistu ecosystem.

This module contains utilities that are specific to NapistuGraph subclasses
and require knowledge of the Napistu data model (SBML_dfs objects, etc.).
"""

from __future__ import annotations

import copy
import logging
import os
import yaml
from typing import Optional, Union, TYPE_CHECKING

from pydantic import BaseModel
import igraph as ig
import numpy as np
import pandas as pd

from napistu import sbml_dfs_core
from napistu import source
from napistu.network import net_create
from napistu.identifiers import _validate_assets_sbml_ids

if TYPE_CHECKING:
    from napistu.network.ng_core import NapistuGraph
from napistu.constants import (
    ENTITIES_TO_ENTITY_DATA,
    ENTITIES_W_DATA,
    SBML_DFS,
    SOURCE_SPEC,
)
from napistu.network.constants import (
    DEFAULT_WT_TRANS,
    DEFINED_WEIGHT_TRANSFORMATION,
    DISTANCES,
    GRAPH_WIRING_APPROACHES,
    GRAPH_DIRECTEDNESS,
    NAPISTU_GRAPH_EDGES,
    NAPISTU_GRAPH_NODE_TYPES,
    NAPISTU_GRAPH_VERTICES,
)

logger = logging.getLogger(__name__)


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
    vertices: pd.DataFrame,
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    min_pw_size: int = 3,
    source_total_counts: Optional[pd.Series | pd.DataFrame] = None,
    verbose: bool = False,
) -> pd.DataFrame | None:
    """
    Assign edges to a set of sources.

    Parameters
    ----------
    vertices: pd.DataFrame
        A table of vertices.
    sbml_dfs: sbml_dfs_core.SBML_dfs
        A pathway model
    min_pw_size: int
        the minimum size of a pathway to be considered
    source_total_counts: pd.Series | pd.DataFrame
        A series of the total counts of each source or a pd.DataFrame with two columns:
        pathway_id and total_counts.
    verbose: bool
        Whether to print verbose output

    Returns
    -------
    reaction_sources: pd.DataFrame
        A table of reactions and the sources they are assigned to.
    """

    nodes = vertices["node"].tolist()
    present_reactions = sbml_dfs.reactions[sbml_dfs.reactions.index.isin(nodes)]

    if len(present_reactions) == 0:
        return None

    source_df = source.unnest_sources(present_reactions)

    if source_df is None:
        return None
    else:
        if source_total_counts is not None:

            source_total_counts = source._ensure_source_total_counts(
                source_total_counts, verbose=verbose
            )
            defined_source_totals = source_total_counts.index.tolist()

            source_mask = source_df[SOURCE_SPEC.PATHWAY_ID].isin(defined_source_totals)

            if sum(~source_mask) > 0:
                if verbose:
                    dropped_pathways = (
                        source_df[~source_mask][SOURCE_SPEC.PATHWAY_ID]
                        .unique()
                        .tolist()
                    )
                    logger.warning(
                        f"Some pathways in `source_df` are not present in `source_total_counts` ({sum(~source_mask)} entries). Dropping these pathways: {dropped_pathways}."
                    )
                source_df = source_df[source_mask]

            if source_df.shape[0] == 0:
                select_source_total_pathways = defined_source_totals[:5]
                if verbose:
                    logger.warning(
                        f"None of the pathways in `source_df` are present in `source_total_counts ({source_df[SOURCE_SPEC.PATHWAY_ID].unique().tolist()})`. Example pathways in `source_total_counts` are: {select_source_total_pathways}; returning None."
                    )
                return None

        reaction_sources = source.source_set_coverage(
            source_df,
            source_total_counts,
            sbml_dfs,
            min_pw_size=min_pw_size,
            verbose=verbose,
        )
        return reaction_sources.reset_index()[
            [SBML_DFS.R_ID, SOURCE_SPEC.PATHWAY_ID, SOURCE_SPEC.NAME]
        ]


def export_networks(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    model_prefix: str,
    outdir: str,
    directeds: list[bool] = [True, False],
    wiring_approaches: list[str] = [
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
    wiring_approaches : [str]
        Types of graphs to construct, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - regulatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products
            - surrogate regulatory approach but with substrates upstream of enzymes

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
    if not isinstance(wiring_approaches, list):
        raise TypeError(
            f"wiring_approaches must be a list but was a {type(wiring_approaches)}"
        )

    # iterate through provided wiring_approaches and export each type
    for wiring_approach in wiring_approaches:
        for directed in directeds:
            export_pkl_path = _create_network_save_string(
                model_prefix=model_prefix,
                outdir=outdir,
                directed=directed,
                wiring_approach=wiring_approach,
            )
            print(f"Exporting {wiring_approach} network to {export_pkl_path}")

            network_graph = net_create.process_napistu_graph(
                sbml_dfs=sbml_dfs,
                directed=directed,
                wiring_approach=wiring_approach,
                verbose=True,
            )

            network_graph.write_pickle(export_pkl_path)

    return None


def pluck_entity_data(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    entity_attrs: dict[str, list[dict]] | list[dict],
    data_type: str,
    custom_transformations: Optional[dict[str, callable]] = None,
) -> pd.DataFrame | None:
    """
    Pluck Entity Attributes from an sbml_dfs based on a set of tables and variables to look for.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        A mechanistic model.
    entity_attrs : dict[str, list[dict]] | list[dict]
        A list of dicts containing the species/reaction attributes to pull out. Of the form:
        [
            "to_be_created_graph_attr_name": {
                "table": "species/reactions data table",
                "variable": "variable in the data table",
                "trans": "optionally, a transformation to apply to the variable (where applicable)"
            }
        ]

        This can also be a dict of the form but this will result in a deprecation warning:
        {
            "species": << entity attributes list >>
            "reactions" : << entity attributes list >>
        }
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

    if data_type in entity_attrs.keys():
        logger.warning(
            f"The provided entity_attrs is a dict of the form {entity_attrs}. This will be deprecated in a future release. Please provide a species- or reactions-level entity_attrs list of dicts."
        )
        entity_attrs = entity_attrs[data_type]

    # validating dict
    _validate_entity_attrs(entity_attrs, custom_transformations=custom_transformations)

    if len(entity_attrs) == 0:
        logger.warning(
            f"No {data_type} attributes were provided in entity_attrs; returning None"
        )
        return None

    data_type_attr = ENTITIES_TO_ENTITY_DATA[data_type]
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


def read_network_pkl(
    model_prefix: str,
    network_dir: str,
    wiring_approach: str,
    directed: bool = True,
) -> "NapistuGraph":
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
    wiring_approach : [str]
        Type of graphs to read, valid values are:
            - bipartite: substrates and modifiers point to the reaction they drive, this reaction points to products
            - reguatory: non-enzymatic modifiers point to enzymes, enzymes point to substrates and products
            - surrogate regulatory approach but with substrates upstream of enzymes

    Returns
    -------
        network_graph: "NapistuGraph"
    A NapistuGraph network of the pathway

    """
    if not isinstance(model_prefix, str):
        raise TypeError(f"model_prefix was a {type(model_prefix)} and must be a str")
    if not os.path.isdir(network_dir):
        raise FileNotFoundError(f"{network_dir} does not exist")
    if not isinstance(directed, bool):
        raise TypeError(f"directed must be a bool, but was {type(directed)}")
    if not isinstance(wiring_approach, str):
        raise TypeError(
            f"wiring_approach must be a str but was a {type(wiring_approach)}"
        )

    import_pkl_path = _create_network_save_string(
        model_prefix, network_dir, directed, wiring_approach
    )
    if not os.path.isfile(import_pkl_path):
        raise FileNotFoundError(f"{import_pkl_path} does not exist")
    print(f"Importing {wiring_approach} network from {import_pkl_path}")

    network_graph = ig.Graph.Read_Pickle(fname=import_pkl_path)

    return network_graph


def validate_assets(
    sbml_dfs: sbml_dfs_core.SBML_dfs,
    napistu_graph: Optional[Union["NapistuGraph", ig.Graph]] = None,
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
        napistu_graph : "NapistuGraph", optional
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


def read_graph_attrs_spec(graph_attrs_spec_uri: str) -> dict:
    """Read a YAML file containing the specification for adding reaction- and/or species-attributes to a napistu_graph."""
    with open(graph_attrs_spec_uri) as f:
        graph_attrs_spec = yaml.safe_load(f)

    VALID_SPEC_SECTIONS = [SBML_DFS.SPECIES, SBML_DFS.REACTIONS]
    defined_spec_sections = set(graph_attrs_spec.keys()).intersection(
        VALID_SPEC_SECTIONS
    )

    if len(defined_spec_sections) == 0:
        raise ValueError(
            f"The provided graph attributes spec did not contain either of the expected sections: {', '.join(VALID_SPEC_SECTIONS)}"
        )

    if SBML_DFS.REACTIONS in defined_spec_sections:
        net_create._validate_entity_attrs(graph_attrs_spec[SBML_DFS.REACTIONS])

    if SBML_DFS.SPECIES in defined_spec_sections:
        net_create._validate_entity_attrs(graph_attrs_spec["reactions"])

    return graph_attrs_spec


# Internal utility functions
def _create_network_save_string(
    model_prefix: str, outdir: str, directed: bool, wiring_approach: str
) -> str:
    if directed:
        directed_str = GRAPH_DIRECTEDNESS.DIRECTED
    else:
        directed_str = GRAPH_DIRECTEDNESS.UNDIRECTED

    export_pkl_path = os.path.join(
        outdir,
        model_prefix + "_network_" + wiring_approach + "_" + directed_str + ".pkl",
    )

    return export_pkl_path


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


def _validate_assets_sbml_graph(
    sbml_dfs: sbml_dfs_core.SBML_dfs, napistu_graph: Union["NapistuGraph", ig.Graph]
) -> None:
    """
    Check an sbml_dfs model and NapistuGraph for inconsistencies.

    Parameters
    ----------
    sbml_dfs : sbml_dfs_core.SBML_dfs
        The pathway representation.
    napistu_graph : "NapistuGraph"
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
        [SBML_DFS.SC_ID, SBML_DFS.SC_NAME]
    ].merge(
        vertices.query(
            f"{NAPISTU_GRAPH_VERTICES.NODE_TYPE} == '{NAPISTU_GRAPH_NODE_TYPES.SPECIES}'"
        ),
        left_on=[SBML_DFS.SC_ID],
        right_on=[NAPISTU_GRAPH_VERTICES.NAME],
    )
    mismatched_names = [
        f"{x} != {y}"
        for x, y in zip(
            matched_cspecies[SBML_DFS.SC_NAME],
            matched_cspecies[NAPISTU_GRAPH_VERTICES.NODE_NAME],
        )
        if x != y
    ]
    if len(mismatched_names) > 0:
        example_names = mismatched_names[: min(10, len(mismatched_names))]
        raise ValueError(
            f"{len(mismatched_names)} species names do not match between sbml_dfs and napistu_graph: {example_names}"
        )
    return None


def _validate_assets_graph_dist(
    napistu_graph: "NapistuGraph", precomputed_distances: pd.DataFrame
) -> None:
    """
    Check a NapistuGraph and precomputed distances table for inconsistencies.

    Parameters
    ----------
    napistu_graph : "NapistuGraph"
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
        edges[
            [
                NAPISTU_GRAPH_EDGES.FROM,
                NAPISTU_GRAPH_EDGES.TO,
                NAPISTU_GRAPH_EDGES.WEIGHT,
                NAPISTU_GRAPH_EDGES.UPSTREAM_WEIGHT,
            ]
        ],
        left_on=[DISTANCES.SC_ID_ORIGIN, DISTANCES.SC_ID_DEST],
        right_on=[NAPISTU_GRAPH_EDGES.FROM, NAPISTU_GRAPH_EDGES.TO],
    )
    inconsistent_weights = edges_with_distances.query(
        f"{DISTANCES.PATH_WEIGHT} != {NAPISTU_GRAPH_EDGES.WEIGHT}"
    )
    if inconsistent_weights.shape[0] > 0:
        logger.warning(
            f"{inconsistent_weights.shape[0]} edges' weights are inconsistent between",
            "edges in the napistu_graph and length 1 paths in precomputed_distances."
            f"This is {inconsistent_weights.shape[0] / edges_with_distances.shape[0]:.2%} of all edges.",
        )
    return None


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
